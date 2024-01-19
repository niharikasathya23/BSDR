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

class HumanCOCO(Dataset):

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

        self.base_transform, self.right_transform, self.disp_transform, _ = get_train_augmentations()
        if self.split != 'train':
            self.right_transform = None

        self.min_polys = 5
        self.max_polys = 20
        self.max_noise = 0.15

    def __len__(self) -> int:
        return len(self.coco_paths)

    def preprocess(self, index: int) -> Tuple[Tensor, Tensor]:
        path = self.coco_paths[index]
        right = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        gt_disp_path = path.replace(f'/{self.split}2017/', '/midas/').replace('.jpg', '.png')
        gt_disp = cv2.imread(gt_disp_path, cv2.IMREAD_UNCHANGED)/10000

        num = int(path.split('/')[-1].split('.')[0])

        anns = [ann for ann in self.coco['annotations'] if ann['category_id']==1 and ann['image_id']==num]

        ih, iw = right.shape[0], right.shape[1]
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

        gt_disp_aug = np.zeros((gt_disp.shape[0], gt_disp.shape[1], 3))
        gt_disp_aug[:,:,0] = gt_disp
        gt_disp_aug[:,:,1] = gt_disp
        gt_disp_aug[:,:,2] = gt_disp
        gt_disp = gt_disp_aug

        if self.base_transform:

            og_h, og_w = right.shape
            wratio = 640/og_w
            hratio = 384/og_h
            if wratio > 1 or hratio > 1:
                if wratio > hratio:
                    right = cv2.resize(right, (round(wratio*og_w), round(wratio*og_h)))
                    seg = cv2.resize(seg, (round(wratio*og_w), round(wratio*og_h)))
                    gt_disp = cv2.resize(gt_disp, (round(wratio*og_w), round(wratio*og_h)))
                else:
                    right = cv2.resize(right, (round(hratio*og_w), round(hratio*og_h)))
                    seg = cv2.resize(seg, (round(hratio*og_w), round(hratio*og_h)))
                    gt_disp = cv2.resize(gt_disp, (round(hratio*og_w), round(hratio*og_h)))

            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            transformed = self.base_transform(image=right, gt_disp=gt_disp, masks=[seg])
            right = transformed['image']
            gt_disp = transformed['gt_disp']
            seg = transformed['masks'][0]

            if self.right_transform:
                transformed = self.right_transform(image=right)
                right = transformed['image']
            right = cv2.cvtColor(transformed['image'], cv2.COLOR_BGR2GRAY)

        else:
            right = cv2.resize(right, (640, 384))
            seg = cv2.resize(seg, (640, 384))
            gt_disp = cv2.resize(gt_disp, (640, 384))

        disp = gt_disp.copy()
        transformed = self.disp_transform(image=disp)
        disp = transformed['image'][:,:,0]
        gt_disp = gt_disp[:,:,0]

        for _ in range(np.random.choice(range(self.min_polys, self.max_polys+1))):
            vs, us = np.where(seg == 1)
            if len(vs) == 0: continue
            idx = np.random.choice(len(vs))
            center = (us[idx], vs[idx])
            poly = generate_polygon(center, 20, np.random.uniform(), np.random.uniform(), 30)
            poly_img = Image.new('L', (640, 384), 0)
            ImageDraw.Draw(poly_img).polygon(poly, outline=1, fill=1)
            poly_mask = np.array(poly_img)

            noise = poly_mask*np.random.uniform(low=-self.max_noise, high=self.max_noise, size=(384,640))
            disp = disp + noise

        # gt_disp_show = ((gt_disp)*255).astype(np.uint8)
        # gt_disp_show = cv2.applyColorMap(gt_disp_show, cv2.COLORMAP_JET)
        # disp_show = ((disp)*255).astype(np.uint8)
        # disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
        # cv2.imshow('right', right)
        # cv2.imshow('seg', seg*255)
        # cv2.imshow('gt_disp', gt_disp_show)
        # cv2.imshow('disp', disp_show)
        # cv2.waitKey(0)

        image = np.stack([right, disp])
        image = torch.tensor(image).float()
        label = torch.tensor(seg).unsqueeze(0)
        gt_disp = torch.tensor(gt_disp).float()

        if self.split == "train":
            orig_image = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            return image, label.squeeze().long(), gt_disp, orig_image

        if self.load_path:
            return image, label.squeeze().long(), gt_disp, path
        else:
            return image, label.squeeze().long(), gt_disp

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:

        try:
            return self.preprocess(index)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            index = np.random.choice(self.__len__())
            return self.preprocess(index)


    @staticmethod
    def collate_fn_mosaic(batch):
        # zipped = zip(*batch)
        # imgs, lbls, gt_disps, _ = zipped
        img_mosaics, lbl_mosaics, gt_disp_mosaics = [], [], []
        for i, (img, lbl, gt_disp, _) in enumerate(batch):
            if i % 4 == 0:
                img_mosaic = np.zeros((2, 384*2, 640*2))
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

    trainset = HumanCOCO(cfg['DATASET']['ROOT'], 'train', dataset_cfg=cfg['DATASET'])
    # trainloader = DataLoader(trainset, batch_size=1, num_workers=0)
    trainloader = DataLoader(trainset, batch_size=4, num_workers=0, collate_fn=trainset.collate_fn_mosaic)

    for img, lbl, gt_disp, _ in trainloader:
        print(img.shape)
        print(lbl.shape)
        print(gt_disp.shape)

        img = img.squeeze().numpy()
        right = img[0,:,:]
        disp = img[1,:,:]
        seg = lbl.squeeze().numpy()
        gt_disp = gt_disp.squeeze().numpy()

        print(np.min(disp), np.max(disp))
        print(np.min(gt_disp), np.max(gt_disp))

        gt_disp_show = ((gt_disp)*255).astype(np.uint8)
        gt_disp_show = cv2.applyColorMap(gt_disp_show, cv2.COLORMAP_JET)
        disp_show = ((disp)*255).astype(np.uint8)
        disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
        cv2.imshow('right', (right).astype(np.uint8))
        cv2.imshow('seg', (seg*255).astype(np.uint8))
        cv2.imshow('gt_disp', gt_disp_show)
        cv2.imshow('disp', disp_show)
        cv2.waitKey(0)
