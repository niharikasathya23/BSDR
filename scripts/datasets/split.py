import numpy as np
import glob, os
import json
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='Configuration file to use')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

with open(args.cfg) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

root = cfg['DATASET']['ROOT']
gra = cfg['DATASET']['GT_RIGHT_ALIGN']
gt_gran = 'right' if gra else 'left'

np.random.seed(args.seed)

splits = {
    'train':[],
    'val':[],
    'test':[] # not implemented for now
}

force_val_path = f"{root}/force_val.txt"
if os.path.exists(force_val_path):
    with open(force_val_path) as f:
         lines = f.readlines()
         force_val = [line[:-1] for line in lines]

    # paths = glob.glob(f"{root}/*{gt_gran}/*.*")
    paths = glob.glob(f"{root}/ann/*.*")

    for path in paths:
        path = path.replace('/ann/', '/rect_right/').replace('_mask_', '_rect_right_')
        granule = path.split('/')[-1].split('_rect')[0]
        if granule in force_val:
            splits['val'].append(path)
        else:
            splits['train'].append(path)

else:
    # paths = sorted(glob.glob(f"{root}/*{gt_gran}/*.*"))
    paths = sorted(glob.glob(f"{root}/ann/*.*"))
    paths = [path.replace('/ann/', '/rect_right/').replace('_mask_', '_rect_right_') for path in paths]
    np.random.shuffle(paths)

    splits['train'] = paths[:int(len(paths)*0.8)]
    splits['val'] = paths[int(len(paths)*0.8):]

with open(root+'/loader.json', 'w') as f:
    json.dump(splits, f)
