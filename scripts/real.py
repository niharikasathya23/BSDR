import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import os
import cv2
import json
import glob
from PIL import Image, ImageDraw
import cmapy
from scipy.signal import medfilt

import sys

import math, random
import albumentations as A

sys.path.append("..")
# from datasets.augmentations import get_train_augmentations


def get_train_augmentations():
    base_augmentations = [
        A.CropNonEmptyMaskIfExists(height=384, width=640),
        A.Rotate(limit=20),
        A.HorizontalFlip(p=0.5),
    ]
    right_augmentations = [
        A.GaussNoise(p=0.3),
        # A.InvertImg(p=0.1),
        A.RandomShadow(),
        A.RandomSunFlare(src_radius=200, p=0.1),
        A.RandomRain(p=0.1),
        # A.RandomSnow(p=0.1),
    ]
    disp_augmentations = [A.GaussNoise(var_limit=0.005, p=1)]

    base_aug = A.Compose(
        base_augmentations,
        additional_targets={
            "gt_disp": "image",
        },
    )
    real_aug = A.Compose(
        base_augmentations, additional_targets={"gt_disp": "image", "disp": "image"}
    )

    right_aug = A.Compose(right_augmentations)

    disp_aug = A.Compose(disp_augmentations)

    return base_aug, right_aug, disp_aug, real_aug


class HumanReal(Dataset):
    # CLASSES = ['background', 'human']
    # PALETTE = torch.tensor([[0, 255, 0], [0, 0, 255]])
    CLASSES = ["background", "human", "object"]
    PALETTE = torch.tensor([[0, 255, 0], [0, 0, 255], [0, 255, 0]])

    def __init__(
        self,
        root: str,
        split: str = "train",
        dataset_cfg=None,
        transform=None,
        load_path=False,
    ) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.max_disp = dataset_cfg["MAX_DISP"]
        self.load_path = load_path

        self.root = root
        self.split = split
        with open(self.root + "/loader.json") as f:
            self.mono_paths = json.load(f)[self.split]

        self.gra = dataset_cfg["GT_RIGHT_ALIGN"]
        self.mono_is = "right" if self.gra else "left"
        self.other_is = "left" if self.gra else "right"
        self.exr_disp = dataset_cfg["EXR_DISP"]
        self.load_filtered = dataset_cfg["LOAD_FILTERED"]
        self.median = dataset_cfg["MEDIAN_FILTER"]
        self.norm_img = dataset_cfg["NORM_IMG"]
        self.norm_disp = dataset_cfg["NORM_DISP"]
        self.norm_gt_disp = dataset_cfg["NORM_GT_DISP"]

        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        if dataset_cfg["FILTER_CLOSE"]:
            raise NotImplementedError()
        #     remove_paths = []
        #     for path in self.mono_paths:
        #         num = int(path.split('_')[-1].split('.png')[0])
        #         granule_split = path.split('/')[-1].split(f'_rect_{self.mono_is}')
        #         granule = f"{granule_split[0]}_" if len(granule_split) else ''
        #         gt_disp_path = f"{self.root}/gt_disp/{granule}disp_{num}.exr"
        #         max_disp = np.max(cv2.imread(gt_disp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        #         if max_disp > 190.:
        #             remove_paths.append(path)
        #     for path in remove_paths:
        #         self.mono_paths.remove(path)

        (
            _,
            self.right_transform,
            self.disp_transform,
            self.base_transform,
        ) = get_train_augmentations()
        if self.split != "train":
            self.right_transform, self.base_transform = None, None

    def __len__(self) -> int:
        return len(self.mono_paths)

    def preprocess(self, index: int) -> Tuple[Tensor, Tensor]:
        # get paths
        mono_path = self.mono_paths[index]
        num = int(mono_path.split("_")[-1].split(".png")[0])
        granule_split = mono_path.split("/")[-1].split(f"_rect_{self.mono_is}")
        if len(granule_split):
            granule = f"{granule_split[0]}_"
        else:
            granule = ""
        other_path = (
            f"{self.root}/rect_{self.other_is}/{granule}rect_{self.other_is}_{num}.png"
        )
        if self.load_filtered:
            ann_path = f"{self.root}/ann_filtered/{granule}mask_{num}.png"
        else:
            ann_path = f"{self.root}/ann/{granule}mask_{num}.png"
        gt_disp_path = f"{self.root}/gt_disp/{granule}disp_{num}.exr"
        if self.exr_disp:
            sgbm_path = f"{self.root}/disparity/disp_{num}.exr"
        else:
            sgbm_path = f"{self.root}/disparity/{granule}disparity_{num}.png"

        # load images
        if self.gra:
            left_path = other_path
            right_path = mono_path
        else:
            left_path = mono_path
            right_path = other_path
        # left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

        # disp from depth
        gt_disp = cv2.imread(gt_disp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        # SGBM
        # leftImg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        rightImg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # construct seg
        seg_key = cv2.imread(ann_path)
        seg = np.zeros_like(rightImg)
        seg[seg_key[..., 0] == 255] = 1
        seg[seg_key[..., 2] == 255] = 2

        disp = cv2.imread(sgbm_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if not self.exr_disp:
            disp = disp / 8  # 8 subpixels
        if self.median:
            disp = medfilt(disp, 7)
        if self.norm_disp:
            disp = disp / self.max_disp  # max disparity

        # rightImg = rightImg[8:-8, :]
        # disp = disp[8:-8, :]
        # if not self.load_filtered:
        #     seg = seg[8:-8, :]
        # gt_disp = gt_disp[8:-8, :]
        if self.norm_gt_disp:
            gt_disp = gt_disp / self.max_disp

        if self.base_transform:
            rightImg = cv2.cvtColor(rightImg, cv2.COLOR_GRAY2BGR)
            transformed = self.base_transform(
                image=rightImg, gt_disp=gt_disp, disp=disp, masks=[seg]
            )
            rightImg = transformed["image"]
            gt_disp = transformed["gt_disp"]
            disp = transformed["disp"]
            seg = transformed["masks"][0]

            if self.right_transform:
                transformed = self.right_transform(image=rightImg)
                rightImg = transformed["image"]
            rightImg = cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2GRAY)

        if self.norm_img:
            rightImg = rightImg / 255

        # convert to tensors
        # image = torch.tensor(left).unsqueeze(0)
        image = np.stack([rightImg, disp])
        # image = np.transpose(left, (2,0,1))
        image = torch.tensor(image).float()
        label = torch.tensor(seg).unsqueeze(0)
        gt_disp = torch.tensor(gt_disp).float()

        err = (gt_disp - disp).float()

        if self.load_path:
            return image, label.squeeze().long(), gt_disp, mono_path
        else:
            return image, label.squeeze().long(), gt_disp, err

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        try:
            return self.preprocess(index)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            index = np.random.choice(self.__len__())
            return self.preprocess(index)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Configuration file to use")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    trainset = HumanReal(cfg["DATASET"]["ROOT"], "val", dataset_cfg=cfg["DATASET"])
    trainloader = DataLoader(trainset, batch_size=1, num_workers=0)
    # trainloader = DataLoader(trainset, batch_size=4, num_workers=0, collate_fn=trainset.collate_fn_mosaic)

    for img, lbl, gt_disp, err in trainloader:
        print(lbl.shape)
        print(gt_disp.shape)
        print(err.shape)

        img = img.squeeze().numpy()
        right = img[0, :, :]
        disp = img[1, :, :]
        seg = lbl.squeeze().numpy()
        gt_disp = gt_disp.squeeze().numpy()
        err = err.squeeze().numpy()

        print(np.min(disp), np.max(disp))
        print(np.min(gt_disp), np.max(gt_disp))
        print(np.min(err), np.max(err))

        gt_disp_show = ((gt_disp) * 255).astype(np.uint8)
        gt_disp_show = cv2.applyColorMap(gt_disp_show, cv2.COLORMAP_JET)
        disp_show = ((disp) * 255).astype(np.uint8)
        disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
        err_show = ((err + 1) / 2 * 255).astype(np.uint8)
        err_show = cv2.applyColorMap(err_show, cmapy.cmap("seismic"))
        cv2.imshow("right", (right).astype(np.uint8))
        cv2.imshow("seg", (seg * 255).astype(np.uint8))
        cv2.imshow("gt_disp", gt_disp_show)
        cv2.imshow("disp", disp_show)
        cv2.imshow("err", err_show)
        cv2.waitKey(0)
