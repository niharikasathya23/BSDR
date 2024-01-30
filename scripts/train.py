import torch
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
import sys

from models import *
from datasets import *
from utils.augmentations import get_train_augmentation, get_val_augmentation
from utils.losses import get_loss
from ndr.loss import DisparityLoss
from utils.schedulers import get_scheduler
from utils.optimizers import get_optimizer
from utils.utils import (
    fix_seeds,
    setup_cudnn,
    cleanup_ddp,
    setup_ddp,
    init_scfg,
    init_synth_scfg,
    get_slc,
)
from val import evaluate
from utils.metrics import Metrics, multi_class_FPr_TPr
from utils.weight_loading import load_segmentation_pretrain_weights

import random
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np


def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg["DEVICE"])
    train_cfg, eval_cfg = cfg["TRAIN"], cfg["EVAL"]
    dataset_cfg, model_cfg = cfg["DATASET"], cfg["MODEL"]
    loss_cfg, optim_cfg, sched_cfg = cfg["LOSS"], cfg["OPTIMIZER"], cfg["SCHEDULER"]
    epochs, lr = train_cfg["EPOCHS"], optim_cfg["LR"]

    traintransform = get_train_augmentation(
        train_cfg["IMAGE_SIZE"], seg_fill=dataset_cfg["IGNORE_LABEL"]
    )
    valtransform = get_val_augmentation(eval_cfg["IMAGE_SIZE"])

    trainset = eval(dataset_cfg["NAME"])(
        dataset_cfg["ROOT"], "train", dataset_cfg=dataset_cfg, transform=valtransform
    )  # transform is effectively None
    valset = eval(dataset_cfg["NAME"])(
        dataset_cfg["ROOT"], "val", dataset_cfg=dataset_cfg, transform=valtransform
    )
    # trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', transform=None)
    # valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', transform=None)

    model = eval(model_cfg["NAME"])(
        model_cfg["BACKBONE"],
        trainset.n_classes,
        using_separation_loss=loss_cfg["NAME"] == "SeparationLoss",
    )
    model.init_pretrained(model_cfg["PRETRAINED"])
    model = model.to(device)

    if len(model_cfg["WEIGHTS"]):
        map_location = None
        if not torch.cuda.is_available():
            map_location = torch.device('cpu')

        if train_cfg["SPECIAL_LOAD"]:
            state_dict = load_segmentation_pretrain_weights(model, model_cfg["WEIGHTS"], map_location)
            model.load_state_dict(state_dict)
            print("SUCCESS!")
        else:
            model.load_state_dict(torch.load(model_cfg["WEIGHTS"], map_location=map_location))
        print("Loaded weights:", model_cfg["WEIGHTS"])

    save_module = torch.cuda.device_count() > 1 or train_cfg["DDP"]
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if "FREEZE_DISP" in train_cfg.keys() and train_cfg["FREEZE_DISP"]:
        print("Freezing disp head...")
        if torch.cuda.device_count() > 1:
            for k, v in model.module.disp_head.named_parameters():
                v.requires_grad = False
        else:
            for k, v in model.disp_head.named_parameters():
                v.requires_grad = False
    if "FREEZE_SEG" in train_cfg.keys() and train_cfg["FREEZE_SEG"]:
        print("Freezing seg head...")
        if torch.cuda.device_count() > 1:
            for k, v in model.module.logits_head.named_parameters():
                v.requires_grad = False
        else:
            for k, v in model.logits_head.named_parameters():
                v.requires_grad = False

    if "CALIB" in train_cfg.keys():
        scfg = init_scfg(cfg)
    else:
        scfg = init_synth_scfg()
    slc = get_slc(scfg)

    if train_cfg["DDP"]:
        sampler = DistributedSampler(
            trainset, dist.get_world_size(), dist.get_rank(), shuffle=True
        )
        model = DDP(model, device_ids=[gpu])
    else:
        sampler = RandomSampler(trainset)

    # print('#workers', num_workers)
    num_workers = 20
    if train_cfg["MOSAIC"]:
        assert train_cfg["BATCH_SIZE"] % 4 == 0
        trainloader = DataLoader(
            trainset,
            batch_size=train_cfg["BATCH_SIZE"],
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
            collate_fn=trainset.collate_fn_mosaic,
        )
    else:
        trainloader = DataLoader(
            trainset,
            batch_size=train_cfg["BATCH_SIZE"],
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
        )
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)

    iters_per_epoch = len(trainset) // train_cfg["BATCH_SIZE"]
    print("Trainset:", len(trainset))
    print("train_cfg",train_cfg["BATCH_SIZE"])

    loss_fn = get_loss(loss_cfg["NAME"], trainset.ignore_label, None)
    optimizer = get_optimizer(model, optim_cfg["NAME"], lr, optim_cfg["WEIGHT_DECAY"])
    scheduler = get_scheduler(
        sched_cfg["NAME"],
        optimizer,
        epochs * iters_per_epoch,
        sched_cfg["POWER"],
        iters_per_epoch * sched_cfg["WARMUP"],
        sched_cfg["WARMUP_RATIO"],
    )
    scaler = GradScaler(enabled=train_cfg["AMP"])
    writer = SummaryWriter(str(save_dir / "logs"))

    for epoch in range(epochs):
        model.train()
        if train_cfg["DDP"]:
            sampler.set_epoch(epoch)

        train_loss = 0.0
        pbar = tqdm(
            enumerate(trainloader),
            total=iters_per_epoch,
            desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}",
        )

        random_batch_indices = random.sample(range(0, len(trainloader)), 5)

        batch_idx = 0
        metrics = Metrics(
            trainloader.dataset.n_classes, trainloader.dataset.ignore_label, device
        )

        class_metrics = {
            "macro_TPr": [],
            "macro_FPr": [],
        }

        for iter, (img, lbl, gt_disp, err) in pbar:
            optimizer.zero_grad(set_to_none=True)

            img = img.to(device)
            lbl = lbl.to(device)
            gt_disp = gt_disp.to(device)
            err = err.to(device)

            with autocast(enabled=train_cfg["AMP"]):
                logits = model(img)
                if loss_cfg["NAME"] == "DispLoss":
                    loss = loss_fn(logits[-1], gt_disp, img[:, 1, ...])
                else:
                    # loss = loss_fn(logits, lbl, err, img[:,1,...])
                    loss = loss_fn(logits, lbl, gt_disp, img[:, 1, ...])

            outputs_batch = logits[0]

            classes = nn.functional.softmax(outputs_batch, dim=1)

            metrics.update(classes, lbl)

            seg_pred = torch.argmax(classes, dim=1)
            y_true = (
                lbl.cpu()
                .numpy()
                .reshape(
                    -1,
                )
            )
            y_pred = (
                seg_pred.detach()
                .cpu()
                .numpy()
                .reshape(
                    -1,
                )
            )

            # a specific order instead of the label's order of appearance
            cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
            report = multi_class_FPr_TPr(cnf_matrix, ["background", "person"])

            macro_TPr = np.array([dict_["TPr"] for dict_ in report.values()]).mean()
            macro_FPr = np.array([dict_["FPr"] for dict_ in report.values()]).mean()

            class_metrics["macro_TPr"].append(macro_TPr)
            class_metrics["macro_FPr"].append(macro_FPr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if torch.cuda.is_available():
             torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(
                f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}"
            )
            batch_idx += 1

        train_loss /= iter + 1

        writer.add_scalar("train/loss", train_loss, epoch)

        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()

        for metric, metric_list in class_metrics.items():
            writer.add_scalar(f"train/{metric}", np.array(metric_list).mean(), epoch)

        writer.add_scalar("train/mIoU", miou, epoch)
        writer.add_scalar("train/mAcc", macc, epoch)
        writer.add_scalar("train/mf1", mf1, epoch)

        torch.cuda.empty_cache()

        if (
            (epoch + 1) % train_cfg["EVAL_INTERVAL"] == 0 or (epoch + 1) == epochs
        ) and train_cfg["EVAL_INTERVAL"] != -1:
            acc, macc, f1, mf1, ious, miou = evaluate(
                model,
                valloader,
                device,
                scfg,
                slc,
                writer=writer,
                epoch=epoch,
                training=True,
            )
            writer.add_scalar("val/mIoU", miou, epoch)
            writer.add_scalar("val/mAcc", macc, epoch)
            writer.add_scalar("val/mf1", mf1, epoch)

            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(
                    model.module.state_dict() if save_module else model.state_dict(),
                    save_dir
                    / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth",
                )
            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")
        torch.save(
            model.module.state_dict() if save_module else model.state_dict(),
            save_dir / f"last.pth",
        )

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ["Best mIoU", f"{best_mIoU:.2f}"],
        ["Total Training Time", time.strftime("%H:%M:%S", end)],
    ]
    print(tabulate(table, numalign="right"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Configuration file to use")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    if torch.cuda.is_available():
        gpu = setup_ddp()
    else:
        gpu = torch.device("cpu")

    save_dir = Path(cfg["SAVE_DIR"])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()
