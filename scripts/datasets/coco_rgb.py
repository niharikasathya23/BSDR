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
import warnings
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils

import sys
sys.path.append('..')
from datasets.augmentations import get_train_augmentations, generate_polygon
# from augmentations import get_train_augmentations, generate_polygon

class HumanCOCORgb(Dataset):

    CLASSES = ['background', 'human']
    PALETTE = torch.tensor([[0, 255, 0], [0, 0, 255]])

    def __init__(self, root: str, split: str = 'train', dataset_cfg = None, transform = None, load_path = False) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.max_disp = 192
        self.load_path = load_path

        self.root = root
        self.split = split

        # self.coco_paths = sorted(glob.glob(f"{self.root}/{self.split}2017/*.jpg"))

        with open(f"{self.root}/annotations/single_person_splits.json") as file:
            splits = json.load(file)[self.split]
        self.coco_paths = [f"{self.root}/{self.split}2017/{fn}" for fn in splits]

        with open(f"{self.root}/annotations/instances_{self.split}2017.json") as file:
            self.coco = json.load(file)

        self.base_transform, self.right_transform, self.disp_transform = get_train_augmentations()
        if self.split != 'train':
            self.right_transform = None

        self.min_polys = 5
        self.max_polys = 20
        self.max_noise = 0.15

    def __len__(self) -> int:
        return len(self.coco_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:

        path = self.coco_paths[index]
        rgb = cv2.imread(path)

        num = int(path.split('/')[-1].split('.')[0])

        anns = [ann for ann in self.coco['annotations'] if ann['category_id']==1 and ann['image_id']==num]

        ih, iw = rgb.shape[0], rgb.shape[1]
        gt_disp = np.zeros((ih, iw))
        seg = np.zeros((ih, iw))

        for ann in anns:
            if not ann['iscrowd'] and 'segmentation' in ann.keys():
                segmentation = ann['segmentation']
                if isinstance(segmentation, list) and len(segmentation) > 0: # polygon format
                    rles = maskUtils.frPyObjects(segmentation, ih, iw)
                    rle = maskUtils.merge(rles)
                elif isinstance(segmentation, dict) and len(segmentation['counts']) > 0:
                    rle = maskUtils.frPyObjects(segmentation, ih, iw)

                mask = maskUtils.decode(rle)
                seg = seg + mask

        seg[seg > 0] = 1

        if self.base_transform:

            og_h, og_w, _ = rgb.shape
            wratio = 640/og_w
            hratio = 384/og_h
            if wratio > 1 or hratio > 1:
                if wratio > hratio:
                    rgb = cv2.resize(rgb, (round(wratio*og_w), round(wratio*og_h)))
                    seg = cv2.resize(seg, (round(wratio*og_w), round(wratio*og_h)))
                    gt_disp = cv2.resize(gt_disp, (round(wratio*og_w), round(wratio*og_h)))
                else:
                    rgb = cv2.resize(rgb, (round(hratio*og_w), round(hratio*og_h)))
                    seg = cv2.resize(seg, (round(hratio*og_w), round(hratio*og_h)))
                    gt_disp = cv2.resize(gt_disp, (round(hratio*og_w), round(hratio*og_h)))

            transformed = self.base_transform(image=rgb, gt_disp=gt_disp, masks=[seg])
            rgb = transformed['image']
            gt_disp = transformed['gt_disp']
            seg = transformed['masks'][0]

            if self.right_transform:
                transformed = self.right_transform(image=rgb)
                rgb = transformed['image']

        else:
            rgb = cv2.resize(rgb, (640, 384))
            seg = cv2.resize(seg, (640, 384))

        # gt_disp_show = ((gt_disp)*255).astype(np.uint8)
        # gt_disp_show = cv2.applyColorMap(gt_disp_show, cv2.COLORMAP_JET)
        # disp_show = ((disp)*255).astype(np.uint8)
        # disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
        # cv2.imshow('right', right)
        # cv2.imshow('seg', seg*255)
        # cv2.imshow('gt_disp', gt_disp_show)
        # cv2.imshow('disp', disp_show)
        # cv2.waitKey(0)

        # image = np.stack([right, disp])
        image = np.transpose(rgb, (2,0,1))
        image = torch.tensor(image).float()
        label = torch.tensor(seg).unsqueeze(0)
        gt_disp = torch.tensor(gt_disp).float()

        if self.split == "train":
            orig_image = rgb.copy()
            return image, label.squeeze().long(), gt_disp, orig_image

        if self.load_path:
            return image, label.squeeze().long(), gt_disp, path
        else:
            return image, label.squeeze().long(), gt_disp


    @staticmethod
    def collate_fn_mosaic(batch):
        # zipped = zip(*batch)
        # imgs, lbls, gt_disps, _ = zipped
        img_mosaics, lbl_mosaics, gt_disp_mosaics = [], [], []
        for i, (img, lbl, gt_disp, _) in enumerate(batch):
            if i % 4 == 0:
                img_mosaic = np.zeros((3, 384*2, 640*2))
                lbl_mosaic = np.zeros((384*2, 640*2))
                gt_disp_mosaic = np.zeros((384*2, 640*2))
                img_mosaic[:,:384,:640] = img
                lbl_mosaic[:384,:640] = lbl
                gt_disp_mosaic[:384,:640] = gt_disp
            elif i % 4 == 1:
                img_mosaic[:,:384,640:] = img
                lbl_mosaic[:384,640:] = lbl
                gt_disp_mosaic[:384,640:] = gt_disp
            elif i % 4 == 2:
                img_mosaic[:,384:,:640] = img
                lbl_mosaic[384:,:640] = lbl
                gt_disp_mosaic[384:,:640] = gt_disp
            elif i % 4 == 3:
                img_mosaic[:,384:,640:] = img
                lbl_mosaic[384:,640:] = lbl
                gt_disp_mosaic[384:,640:] = gt_disp
                img_mosaics.append(img_mosaic)
                lbl_mosaics.append(lbl_mosaic)
                gt_disp_mosaics.append(gt_disp_mosaic)

        image = np.stack(img_mosaics, 0)
        label = np.stack(lbl_mosaics, 0)
        gt_disp = np.stack(gt_disp_mosaics, 0)

        image = torch.tensor(image).float()
        label = torch.tensor(label).long()
        gt_disp = torch.tensor(gt_disp).float()

        return image, label, gt_disp, image

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    trainset = HumanCOCORgb(cfg['DATASET']['ROOT'], 'train', dataset_cfg=cfg['DATASET'])
    # trainloader = DataLoader(trainset, batch_size=1, num_workers=0)
    trainloader = DataLoader(trainset, batch_size=4, num_workers=0, collate_fn=trainset.collate_fn_mosaic)

    for img, lbl, gt_disp, _ in trainloader:
        print(img.shape)
        print(lbl.shape)
        print(gt_disp.shape)

        img = img.squeeze().numpy()
        img = np.transpose(img, (1,2,0))
        seg = lbl.squeeze().numpy()

        cv2.imshow('rgb', (img).astype(np.uint8))
        cv2.imshow('seg', (seg*255).astype(np.uint8))
        cv2.waitKey(0)
