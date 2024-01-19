import torch
import numpy as np
import sys
sys.path.append('..')
from models import *

def check_start(param, exclude):
    tests = [param.startswith(start) for start in exclude]
    if np.sum(tests) > 0:
        return False
    else:
        return True

def load_segmentation_pretrain_weights(model, ckpt_file, from_coco=False):
    ckpt = torch.load(ckpt_file)#['model_state_dict']

    exclude = [
        'logits_head.conv_1x1.',
        'context16_head.conv_1x1.',
        'context32_head.conv_1x1.'
    ]

    mdict = model.state_dict()
    for param in ckpt:

        if check_start(param, ['disp_head.']):# and not param.startswith():
            if from_coco and check_start(param, exclude):
                mdict[param] = ckpt[param]
            elif not from_coco:
                mdict[param] = ckpt[param]

    return mdict

if __name__ == "__main__":

    model = BiSeNetv1Disp2('MicroNet-M1', 3)
    ckpt_file = "../runs/coco_best/BiSeNetv1Disp2_MicroNet-M1_HumanCOCO.pth"

    state_dict = load_segmentation_pretrain_weights(model, ckpt_file)
    model.load_state_dict(state_dict)
