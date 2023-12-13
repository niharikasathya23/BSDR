import numpy as np
import cv2
import glob
import depthai as dai
import torch

import sys
sys.path.append("utils")
sys.path.append("utils/crestereo")
from crestereo.test_model import inference
from crestereo.nets import Model

class Estimator:

    def __init__(self):
        pass

    def calc_fov_D_H_V(self, f, w, h):
        return np.degrees(2*np.arctan(np.sqrt(w*w+h*h)/(2*f))), np.degrees(2*np.arctan(w/(2*f))), np.degrees(2*np.arctan(h/(2*f)))

    def get_LR_paths(self, dir):
        self.left_paths = sorted(glob.glob(f"{dir}/*left/*.*"))
        self.right_paths = sorted(glob.glob(f"{dir}/*right/*.*"))

    def depth_from_disp(self, disp):
        depth = (self.focal*self.baseline) / disp / 1000 # div by 1000 to turn units to meters
        depth[disp == 0] = 0
        return depth

class CREStereoEstimator(Estimator):

    def __init__(self, dir, max_disp=256, ckpt="utils/crestereo/models/crestereo_eth3d.pth", align="right"):
        self.name = 'crestereo'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_disp = max_disp
        self.get_LR_paths(dir)
        self.model = Model(max_disp=self.max_disp, mixed_precision=False, test_mode=True)
        self.model.load_state_dict(torch.load(ckpt), strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.align = align

    def estimate_depth(self, left_path, right_path, return_disp=False):
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)
        if self.align == "right":
            left_og = left.copy()
            right_og = right.copy()
            left = cv2.flip(right_og, 1)
            right = cv2.flip(left_og, 1)
        pred = inference(left, right, self.model, n_iter=20)
        if self.align == "right":
            pred = cv2.flip(pred, 1)

        if return_disp:
            return pred
        else:
            return self.depth_from_disp(pred)
