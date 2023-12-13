import argparse
import cv2
import numpy as np
import glob
import os
import imageio
from tqdm import tqdm
from utils import CREStereoEstimator

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, type=str, help="Path to extracted frames directory")
parser.add_argument('--debug', action='store_true', help="Visualize debug")
args = parser.parse_args()

ests  = [CREStereoEstimator(args.path)]
est = ests[0]

if not os.path.exists(args.path+'/gt_disp'):
    os.mkdir(args.path+'/gt_disp')

shuffle_idxs = np.random.choice(len(est.left_paths), len(est.left_paths), replace=False)
est.left_paths = list(np.array(est.left_paths)[shuffle_idxs])
est.right_paths = list(np.array(est.right_paths)[shuffle_idxs])

for left_path in tqdm(est.left_paths):
    granule_split = left_path.split('/')[-1].split('_rect_left')
    if len(granule_split):
        granule = f"{granule_split[0]}_"
    else:
        granule = ''

    num = int(left_path.split('_')[-1].split('.png')[0])
    idx = est.left_paths.index(left_path)
    right_path = est.right_paths[idx]

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    denom = 192
    depthMultiplier = 255 / denom

    depths = []
    shows = []
    for ei, est in enumerate(ests):
        disp = est.estimate_depth(left_path, right_path, return_disp=True)
        show = (disp*depthMultiplier).astype(np.uint8)
        show = cv2.applyColorMap(show, cv2.COLORMAP_JET)

        if args.debug:
            cv2.imshow(est.name, show)

    if args.debug:
        cv2.imshow("right", right)
    key = cv2.waitKey()
    if key == ord('q'):
        break

    imageio.imwrite(f"{args.path}/gt_disp/{granule}disp_{num}.exr", disp)
