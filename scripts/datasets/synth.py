import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import os
import cv2
import glob
import json
from PIL import Image
from scipy.signal import medfilt
import cmapy

import sys
sys.path.append('..')
from datasets.augmentations import get_train_augmentations

class HumanSynthetic(Dataset):

    CLASSES = ['background', 'human']
    PALETTE = torch.tensor([[0, 255, 0], [0, 0, 255]])

    def __init__(self, root: str, split: str = 'train',  dataset_cfg = None, transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.max_disp = 192

        self.root = root
        self.split = split
        with open(self.root+'/loader.json') as f:
            self.left_paths = json.load(f)[self.split]

        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

        self.hfov = np.deg2rad(73.5)
        self.sensor_width, self.sensor_height = 1280, 720
        self.focal = self.sensor_width / (2 * np.tan(self.hfov / 2))
        # self.baseline = 0.075 # 7.5 cm
        self.baseline = 0.04 # 4 cm
        self.synth_baseline = 0.075 # 7.5 cm

        _, self.right_transform, self.disp_transform, self.base_transform = get_train_augmentations()
        if self.split != 'train':
            self.right_transform = None

    def __len__(self) -> int:
        return len(self.left_paths)

    def preprocess(self, index: int) -> Tuple[Tensor, Tensor]:

        # get paths
        left_path = self.left_paths[index]
        num = int(left_path.split('/left/rgb_')[-1].split('.jpg')[0])
        right_path = f"{self.root}/right/rgb_{num}.jpg"
        seg_path = f"{self.root}/right_instance/Instance_{num}.png"
        depth_path = f"{self.root}/ScreenCapture/Main Camera (2)_depth_{num-1}.exr"
        sgbm_path = f"{self.root}/SGBM/disp_{num}.exr"

        # load images
        left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)
        seg = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)
        seg = seg>0
        seg = seg[...,0] + seg[...,1] + seg[...,2] # create binary mask
        seg = seg.astype(np.uint8)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0]*1000 # loaded in units of 1000 meters

        # disp from depth
        gt_disp = (self.focal * self.baseline) / depth
        gt_disp = gt_disp / 190.

        # # SGBM
        leftImg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        rightImg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disp = cv2.imread(sgbm_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        # adjust disparity as if it was a 4 cm baseline
        disp = (self.baseline / self.synth_baseline) * disp
        disp = disp / 190. # max disparity

        # rightImg = cv2.resize(rightImg, (640,384))
        # disp = cv2.resize(disp, (640,384))
        # seg = cv2.resize(seg.astype(np.uint8), (640,384))
        # gt_disp = cv2.resize(gt_disp, (640,384))
        # left = cv2.resize(left, (640,384))
        if self.base_transform:
            og_h, og_w = rightImg.shape
            wratio = 640/og_w
            hratio = 384/og_h
            if wratio > 1 or hratio > 1:
                if wratio > hratio:
                    rightImg = cv2.resize(rightImg, (round(wratio*og_w), round(wratio*og_h)))
                    seg = cv2.resize(seg, (round(wratio*og_w), round(wratio*og_h)))
                    gt_disp = cv2.resize(gt_disp, (round(wratio*og_w), round(wratio*og_h)))
                    disp = cv2.resize(disp, (round(wratio*og_w), round(wratio*og_h)))
                else:
                    rightImg = cv2.resize(rightImg, (round(hratio*og_w), round(hratio*og_h)))
                    seg = cv2.resize(seg, (round(hratio*og_w), round(hratio*og_h)))
                    gt_disp = cv2.resize(gt_disp, (round(hratio*og_w), round(hratio*og_h)))
                    disp = cv2.resize(disp, (round(hratio*og_w), round(hratio*og_h)))

            rightImg = cv2.cvtColor(rightImg, cv2.COLOR_GRAY2BGR)
            transformed = self.base_transform(image=rightImg, gt_disp=gt_disp, disp=disp, masks=[seg])
            rightImg = transformed['image']
            gt_disp = transformed['gt_disp']
            disp = transformed['disp']
            seg = transformed['masks'][0]

            if self.right_transform:
                transformed = self.right_transform(image=rightImg)
                rightImg = transformed['image']
            rightImg = cv2.cvtColor(transformed['image'], cv2.COLOR_BGR2GRAY)

        else:
            rightImg = cv2.resize(rightImg, (640, 384))
            seg = cv2.resize(seg, (640, 384))
            gt_disp = cv2.resize(gt_disp, (640, 384))
            disp = cv2.resize(disp, (640, 384))

        # convert to tensors
        # image = torch.tensor(left).unsqueeze(0)
        image = np.stack([rightImg, disp])
        # image = np.transpose(left, (2,0,1))
        image = torch.tensor(image).float()
        label = torch.tensor(seg).unsqueeze(0)
        gt_disp = torch.tensor(gt_disp).float()

        err = (gt_disp - disp).float()

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
    parser.add_argument('--cfg', type=str, help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    trainset = HumanSynthetic(cfg['DATASET']['ROOT'], 'train', dataset_cfg=cfg['DATASET'])
    trainloader = DataLoader(trainset, batch_size=1, num_workers=0)

    for img, lbl, gt_disp, err in trainloader:
        print(lbl.shape)
        print(gt_disp.shape)
        print(err.shape)

        img = img.squeeze().numpy()
        right = img[0,:,:]
        disp = img[1,:,:]
        seg = lbl.squeeze().numpy()
        gt_disp = gt_disp.squeeze().numpy()
        err = err.squeeze().numpy()

        print(np.min(disp), np.max(disp))
        print(np.min(gt_disp), np.max(gt_disp))
        print(np.min(err), np.max(err))

        gt_disp_show = ((gt_disp)*255).astype(np.uint8)
        gt_disp_show = cv2.applyColorMap(gt_disp_show, cv2.COLORMAP_JET)
        disp_show = ((disp)*255).astype(np.uint8)
        disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
        err_show = ((err+1)/2*255).astype(np.uint8)
        err_show = cv2.applyColorMap(err_show, cmapy.cmap('seismic'))
        cv2.imshow('right', (right).astype(np.uint8))
        cv2.imshow('seg', (seg*255).astype(np.uint8))
        cv2.imshow('gt_disp', gt_disp_show)
        cv2.imshow('disp', disp_show)
        cv2.imshow('err', err_show)
        cv2.waitKey(0)
