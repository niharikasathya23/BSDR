import torch
import json
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import yaml
# from datasets.augmentations import get_train_augmentations
# from augmentations import generate_polygon

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='Configuration file to use')
parser.add_argument('--debug', action='store_true', help='Debug')
args = parser.parse_args()

BASELINE = 0.04

with open(args.cfg) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

root = cfg['DATASET']['ROOT']
coco_paths = []
with open(f"{root}/annotations/person_splits.json") as file:
    splits = json.load(file)
# for split in ['train', 'val']:
for split in ['val']:
    coco_paths += [f"{root}/{split}2017/{fn}" for fn in splits[split]]

# source: https://pytorch.org/hub/intelisl_midas_v2/
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = 'cpu'
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

for path in tqdm(coco_paths):

    try:
        img = cv2.imread(path)
        show_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        output[output <= 0] = 1e-16 # some epsilon

        mind = np.min(output)
        maxd = np.max(output)
        write = (((output-mind)/(maxd-mind))*10000).astype(np.uint16)
        if '/train2017/' in path:
            write_path = path.replace('/train2017/', '/midas/').replace('.jpg', '.png')
        elif '/val2017/' in path:
            write_path = path.replace('/val2017/', '/midas/').replace('.jpg', '.png')
        cv2.imwrite(write_path, write)

        if args.debug:
            show = ((output/maxd)*255).astype(np.uint8)
            # show = ((gt_disp/192)*255).astype(np.uint8)
            show = cv2.applyColorMap(show, cv2.COLORMAP_JET)
            cv2.imshow('coco', show_img)
            cv2.imshow('midas', show)
            cv2.waitKey(0)

    except:
        print(f'issue with {path}')
