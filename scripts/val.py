import torch
import argparse
import yaml
import math
import pickle
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
import sys
sys.path.insert(0, '.')
from models import *
from ndr.models import Model
from datasets import *
from utils.augmentations import get_val_augmentation
from utils.metrics import Metrics, multi_class_FPr_TPr
from utils.utils import setup_cudnn, init_scfg, init_synth_scfg, get_slc
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.signal import medfilt

import open3d as o3d
import depthai as dai
import warnings

@torch.no_grad()
def disparity_eval(disp, ref, gt_disp, lbl):
    def disp_err(map1, map2, lbl, occ=None):
        if occ is not None:
            vs, us = occ
            map1[vs, us] = 0
            map2[vs, us] = 0

        # import cv2
        # show1 = ((map1*lbl / 90)*255).astype(np.uint8)
        # show2 = ((map2*lbl / 90)*255).astype(np.uint8)
        # show1 = cv2.applyColorMap(show1, cv2.COLORMAP_JET)
        # show2 = cv2.applyColorMap(show2, cv2.COLORMAP_JET)
        # cv2.imshow('map1', show1)
        # cv2.imshow('map2', show2)
        # print(np.mean(np.abs(map1*lbl - map2*lbl)))
        # cv2.waitKey(0)

        # test = np.abs(map1[lbl!=0] - map2[lbl!=0])
        # import matplotlib.pyplot as plt
        # plt.boxplot(test)
        # plt.show()

        # return np.median(np.abs(map1[lbl!=0] - map2[lbl!=0]))
        return np.mean(np.abs(map1[lbl!=0] - map2[lbl!=0]))

    vs, us = np.where(disp == 0)

    mask_all = lbl>0
    mask_person = lbl==1
    mask_object = lbl==2
    occ_mask = disp==0

    gt_disp_mask = gt_disp*mask_all

    gt_mean_person = np.mean(gt_disp[mask_person])
    gt_mean_object = np.mean(gt_disp[mask_object])

    perc_occ_all = np.sum(np.logical_and(occ_mask, mask_all)) / np.sum(mask_all)
    perc_occ_person = np.sum(np.logical_and(occ_mask, mask_person)) / np.sum(mask_person)
    perc_occ_object = np.sum(np.logical_and(occ_mask, mask_object)) / np.sum(mask_object)

    if np.sum([gt_disp_mask != 0]) > 0:

        results = {
            'disp_o': disp_err(disp, gt_disp, mask_all) if np.sum(mask_all)>0 else None,
            'ref_o': disp_err(ref, gt_disp, mask_all) if np.sum(mask_all)>0 else None,
            'disp_o_person': disp_err(disp, gt_disp, mask_person) if np.sum(mask_person)>0 else None,
            'ref_o_person': disp_err(ref, gt_disp, mask_person) if np.sum(mask_person)>0 else None,
            'disp_o_object': disp_err(disp, gt_disp, mask_object) if np.sum(mask_object)>0 else None,
            'ref_o_object': disp_err(ref, gt_disp, mask_object) if np.sum(mask_object)>0 else None,
            'disp_no': disp_err(disp, gt_disp, mask_all, occ=(vs,us)) if np.sum(mask_all)>0 else None,
            'ref_no': disp_err(ref, gt_disp, mask_all, occ=(vs,us)) if np.sum(mask_all)>0 else None,
            'disp_no_person': disp_err(disp, gt_disp, mask_person, occ=(vs,us)) if np.sum(mask_person)>0 else None,
            'ref_no_person': disp_err(ref, gt_disp, mask_person, occ=(vs,us)) if np.sum(mask_person)>0 else None,
            'disp_no_object': disp_err(disp, gt_disp, mask_object, occ=(vs,us)) if np.sum(mask_object)>0 else None,
            'ref_no_object': disp_err(ref, gt_disp, mask_object, occ=(vs,us)) if np.sum(mask_object)>0 else None,
            'perc_occ_all': perc_occ_all if np.sum(mask_all)>0 else None,
            'perc_occ_person': perc_occ_person if np.sum(mask_person)>0 else None,
            'perc_occ_object': perc_occ_object if np.sum(mask_object)>0 else None,
            'gt_mean_person': gt_mean_person if np.sum(mask_person)>0 else None,
            'gt_mean_object': gt_mean_object if np.sum(mask_object)>0 else None
        }

        return results

@torch.no_grad()
def get_point_cloud(slc, disp, mask, focal, baseline):
    # get point cloud
    vs, us = np.where(mask == 1)
    ds = disp[vs, us]
    zs = -(focal*baseline)/ds # -Z to follow right hand rule
    spatials = slc.calc_point_spatials(us, vs, zs)/1000 # div by 1000 to convert mm to m

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(spatials)
    return pcd

@torch.no_grad()
def get_spatials(slc, disp, mask, focal, baseline):
    # get point cloud
    vs, us = np.where(mask == 1)
    ds = disp[vs, us]
    ds[ds == 0] = 1e-16
    zs = -(focal*baseline)/ds # -Z to follow right hand rule
    spatials = slc.calc_point_spatials(us, vs, zs)/1000 # div by 1000 to convert mm to m
    return spatials

@torch.no_grad()
def evaluate_ndr(model, dataloader, device, scfg, slc, writer=None, epoch=0, vis=False):
    print('Evaluating...')
    model.eval()

    class_metrics = {
        "mean_spatial_error": [],
        "mean_spatial_error_disp": [],
        "median_spatial_error": [],
        "median_spatial_error_disp": []
    }
    all_results = {}

    for data in tqdm(dataloader):
        images, labels, gt_disp = data[0], data[1], data[2]
        images = images.to(device)
        labels = labels.to(device)

        gray = images[:,0,...]
        disp = images[:,1,...]
        gray = gray.unsqueeze(1)
        disp = disp.unsqueeze(1)
        gt = gt_disp.unsqueeze(1)
        rgb = torch.cat([gray, gray, gray], dim=1)

        ref = model(rgb, disp)[0]

        disp = disp[0][0].to('cpu').numpy()*256
        gt_disp = gt_disp[0].to('cpu').numpy()
        ref = ref[0].to('cpu').numpy().reshape(disp.shape)

        lbl = labels[0].to('cpu').numpy().astype(np.uint8)
        mask = lbl>0

        gt_spatials = get_spatials(slc, gt_disp, mask, scfg['focal'], scfg['baseline'])
        pred_spatials = get_spatials(slc, ref, mask, scfg['focal'], scfg['baseline'])
        pred_spatials_2 = get_spatials(slc, disp, mask, scfg['focal'], scfg['baseline'])

        distances = np.linalg.norm(gt_spatials - pred_spatials, axis=1)
        distances_2 = np.linalg.norm(gt_spatials - pred_spatials_2, axis=1)

        class_metrics["mean_spatial_error"].append(np.mean(distances))
        class_metrics["mean_spatial_error_disp"].append(np.mean(distances_2))
        class_metrics["median_spatial_error"].append(np.median(distances))
        class_metrics["median_spatial_error_disp"].append(np.median(distances_2))

        # Disparity evaluation
        results = disparity_eval(disp, ref, gt_disp, lbl)
        if results is not None:
            if len(all_results.keys()) == 0:
                all_results = {key:[] for key in results}
            for key in results:
                if results[key] is not None:
                    all_results[key].append(results[key])

    for metric, metric_list in class_metrics.items():
        print(metric, np.array(metric_list).mean())
    for metric, metric_list in all_results.items():
        print(metric, np.array(metric_list).mean())
    print()

@torch.no_grad()
def evaluate(model, dataloader, device, scfg, slc, writer=None, epoch=0, vis=False, training=False):
    print('Evaluating...')
    model.eval()

    ## Segmentation Evaluation ##

    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)

    class_metrics = {
        "macro_TPr": [],
        "macro_FPr": [],

        "background_FPr": [],
        "background_TPr": [],

        "person_FPr": [],
        "person_TPr": [],

        "mean_spatial_error": [],
        "mean_spatial_error_disp": [],
        "median_spatial_error": [],
        "median_spatial_error_disp": []
    }
    all_results = {}

    for images, labels, gt_disp, _ in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)

        if training: output = output[0] # this line allows skipping of slow evaluation

        if isinstance(output, tuple):
            preds = output[0].softmax(dim=1)
        else:
            preds = output.softmax(dim=1)

        metrics.update(preds, labels)

        seg_pred = torch.argmax(preds, dim=1)
        y_true = labels.cpu().numpy().reshape(-1,)
        y_pred = seg_pred.detach().cpu().numpy().reshape(-1,)

        # [0, 1] = ['background', 'person'], sklearn's confusion_matrix needs this to keep
        # a specific order instead of the label's order of appearance
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        report = multi_class_FPr_TPr(cnf_matrix, ['background', 'person'])

        macro_TPr = np.array([dict_["TPr"] for dict_ in report.values()]).mean()
        macro_FPr = np.array([dict_["FPr"] for dict_ in report.values()]).mean()

        class_metrics["macro_TPr"].append(macro_TPr)
        class_metrics["macro_FPr"].append(macro_FPr)

        for label_class in report.keys():

            for metric in ["TPr", "FPr"]:

                class_metrics[f"{label_class}_{metric}"].append(report[label_class][metric])


        ## 3D Evaluation ##
        if isinstance(output, tuple):

            # Voxel hit/miss
            disp = images[0,1,...].to('cpu').numpy()
            seg = preds.argmax(dim=1).to(int)
            key = seg[0].detach().to('cpu').numpy().astype(np.uint8)
            if cfg['MODEL']['NAME'] == 'BiSeNetv1Disp3':
                ref = np.squeeze(output[1].to('cpu').numpy()).reshape(disp.shape)
            else:
                err = np.squeeze(output[1].to('cpu').numpy())
                ref = disp+err
                ref = ref*cfg['DATASET']['MAX_DISP']
            lbl = labels[0].to('cpu').numpy().astype(np.uint8)
            mask = lbl>0
            gt_disp = gt_disp[0].to('cpu').numpy()

            if cfg['DATASET']['NORM_DISP']: disp = disp*cfg['DATASET']['MAX_DISP']
            if cfg['DATASET']['NORM_GT_DISP']: gt_disp = gt_disp*cfg['DATASET']['MAX_DISP']
            # disp = disp*cfg['DATASET']['MAX_DISP']
            # gt_disp = gt_disp*cfg['DATASET']['MAX_DISP']
            # disp = disp*256

            if vis:
                disp_show = disp*mask
                disp_show = (disp_show*255).astype(np.uint8)
                disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
                ref_show = ref*mask
                if cfg['MODEL']['NAME'] == 'BiSeNetv1Disp3': ref_show = ref_show/cfg['DATASET']['MAX_DISP']
                ref_show = (ref_show*255).astype(np.uint8)
                ref_show = cv2.applyColorMap(ref_show, cv2.COLORMAP_JET)

                seg = np.zeros((key.shape[0],key.shape[1],3)).astype(np.uint8)
                seg[key==1] = [255,0,0]
                seg[key==2] = [0,0,255]

                gray = images[:,0,...].detach().to('cpu').numpy().squeeze()
                if cfg['DATASET']['NORM_IMG']:
                    gray = gray*255
                gray = gray.astype(np.uint8)

                overlay = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.5, seg, 0.5, 0)

                # err1 = np.abs(gt_disp-ref)
                # err2 = np.abs(gt_disp-disp)
                # bad = err1 > err2
                # good = err1 < err2
                # show = disp_show.copy()
                # show[good] = [0,255,0]
                # show[bad] = [0,0,255]
                # show[lbl == 0] = disp_show[lbl == 0]

                cv2.imshow('right', gray)
                cv2.imshow('ref', ref_show)
                cv2.imshow('disp', disp_show)
                cv2.imshow('mask', overlay)
                # cv2.imshow('errors', show)
                cv2.waitKey(0)

            gt_spatials = get_spatials(slc, gt_disp, mask, scfg['focal'], scfg['baseline'])
            pred_spatials = get_spatials(slc, ref, mask, scfg['focal'], scfg['baseline'])
            pred_spatials_2 = get_spatials(slc, disp, mask, scfg['focal'], scfg['baseline'])

            distances = np.linalg.norm(gt_spatials - pred_spatials, axis=1)
            distances_2 = np.linalg.norm(gt_spatials - pred_spatials_2, axis=1)

            class_metrics["mean_spatial_error"].append(np.mean(distances))
            class_metrics["mean_spatial_error_disp"].append(np.mean(distances_2))
            class_metrics["median_spatial_error"].append(np.median(distances))
            class_metrics["median_spatial_error_disp"].append(np.median(distances_2))

            # Disparity evaluation
            # print('disp', disp.min(), disp.max(), np.median(disp))
            # print('ref', ref.min(), ref.max(), np.median(ref))
            # print('gt_disp', gt_disp.min(), gt_disp.max(), np.median(gt_disp))
            results = disparity_eval(disp, ref, gt_disp, lbl)
            if results is not None:
                if len(all_results.keys()) == 0:
                    all_results = {key:[] for key in results}
                for key in results:
                    if results[key] is not None:
                        all_results[key].append(results[key])


    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    # for size in voxel_sizes:
    #     tpr = np.array(class_metrics[f"voxel_TPr@{size}"]).mean()
    #     fpr = np.array(class_metrics[f"voxel_FPr@{size}"]).mean()
    #     fnr = np.array(class_metrics[f"voxel_FNr@{size}"]).mean()
    #     precision = tpr / (tpr+fpr)
    #     recall = tpr / (tpr+fnr)
    #     voxel_f1 = 2*(precision*recall) / (precision+recall)
    #     class_metrics[f"voxel_P@{size}"] = precision
    #     class_metrics[f"voxel_R@{size}"] = recall
    #     class_metrics[f"voxel_F1@{size}"] = voxel_f1

    if writer:
        for metric, metric_list in class_metrics.items():
            writer.add_scalar(f"val/{metric}", np.array(metric_list).mean(), epoch)
    else:
        for metric, metric_list in class_metrics.items():
            print(metric, np.array(metric_list).mean())
        for metric, metric_list in all_results.items():
            print(metric, np.array(metric_list).mean())
        print()

    with open('results.pickle', 'wb') as file:
        pickle.dump(all_results, file)

    return acc, macc, f1, mf1, ious, miou

def main(cfg, vis):
    device = torch.device(cfg['DEVICE'])

    if 'CALIB' in cfg['TRAIN'].keys():
        scfg = init_scfg(cfg)
    else:
        scfg = init_synth_scfg()
    slc = get_slc(scfg)

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', cfg['DATASET'])
    dataloader = DataLoader(dataset, 1, num_workers=1, pin_memory=True, shuffle=True)

    model_path = Path(cfg['TEST']['MODEL_PATH'])
    # if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    if cfg['MODEL']['NAME'] == 'NDR':
        model = Model(cfg['DATASET']['MAX_DISP'])
        model.load_state_dict(torch.load(str(model_path), map_location='cpu')['model_state_dict'])
        model = model.to(device)

        evaluate_ndr(model, dataloader, device, scfg, slc, vis)
    else:
        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        model = model.to(device)

        acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device, scfg, slc, vis=vis)

        table = {
            'Class': list(dataset.CLASSES) + ['Mean'],
            'IoU': ious + [miou],
            'F1': f1 + [mf1],
            'Acc': acc + [macc]
        }

        print(tabulate(table, headers='keys'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/vermeer-Separation.yaml')
    parser.add_argument('-v', '--visualize', action='store_true', help='flag to visualize')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg, args.visualize)
