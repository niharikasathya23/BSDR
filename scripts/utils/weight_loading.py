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

def load_segmentation_pretrain_weights(model, ckpt_file, map_location=None, from_coco=False):
    ckpt = torch.load(ckpt_file, map_location=map_location)#['model_state_dict']

    exclude = [
        'logits_head.conv_1x1.',
        'context16_head.conv_1x1.',
        'context32_head.conv_1x1.'
    ]

    mdict = model.state_dict()
    for param in ckpt:
        if param in mdict:
            if ckpt[param].shape == mdict[param].shape:
                if from_coco and check_start(param, exclude):
                    mdict[param] = ckpt[param]
                elif not from_coco and not check_start(param, exclude):
                    mdict[param] = ckpt[param]
            else:
                print(f"Skipping {param} due to size mismatch: "
                      f"Checkpoint shape {ckpt[param].shape}, Model shape {mdict[param].shape}")
        else:
            print(f"Skipping {param} as it is not in the model.")

    return mdict

if __name__ == "__main__":

    model = BiSeNetv1Disp2('MicroNet-M1', 3)
    ckpt_file = "../runs/coco_best/BiSeNetv1Disp2_MicroNet-M1_HumanCOCO.pth"

   # Check if CUDA is available and set map_location accordingly
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = load_segmentation_pretrain_weights(model, ckpt_file, map_location=map_location)
    model.load_state_dict(state_dict)
