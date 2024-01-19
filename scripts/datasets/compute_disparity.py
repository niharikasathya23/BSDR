import os
import glob
import cv2
import numpy as np
import argparse
import json
import imageio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help="Directory storing synthetic dataset", required=True)
parser.add_argument('--max_disp', type=int, help="Maximum disparity value", default=192)
parser.add_argument('--debug', action='store_true', help="Flag to debug")
parser.add_argument('--left_align', action='store_true', help="left-align instead of right-align")
args = parser.parse_args()

if not os.path.exists(args.dataset+'/SGBM'):
    os.mkdir(args.dataset+'/SGBM')

left_paths = sorted(glob.glob(f"{args.dataset}/*left/*.*"))

# set up disparity processor
blockSize = 5
K = 32
LRthreshold = 2
stereoProcessor = cv2.StereoSGBM_create(
    minDisparity=1,
    numDisparities=args.max_disp,
    blockSize=blockSize,
    P1=2 * (blockSize ** 2),
    P2=K * (blockSize ** 2),
    disp12MaxDiff=LRthreshold,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

for left_path in tqdm(left_paths):
    num = int(left_path.split('_')[-1].split('.')[0])
    right_path = glob.glob(f"{args.dataset}/*right/*_{num}.*")[0]

    # left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
    # right = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if not args.left_align:
        left_og = left.copy()
        right_og = right.copy()
        left = cv2.flip(right_og.copy(), 1)
        right = cv2.flip(left_og.copy(), 1)

    # SGBM
    leftImg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    rightImg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    padImg = np.zeros(shape=[leftImg.shape[0], args.max_disp], dtype=np.uint8)
    leftImgPadded = cv2.hconcat([padImg, leftImg])
    rightImgPadded = cv2.hconcat([padImg, rightImg])
    disp = stereoProcessor.compute(leftImgPadded, rightImgPadded)
    disp = disp[0:disp.shape[0], args.max_disp:disp.shape[1]]
    subpixelBits = 16.
    disp = (disp / subpixelBits).astype(np.float32)
    if not args.left_align:
        disp = cv2.flip(disp, 1)
        left = left_og
        right = right_og

    if args.debug:
        show = disp.copy().astype(np.uint8)
        show = ((show / args.max_disp)*255).astype(np.uint8)

        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        depth_path = f"{args.dataset}/ScreenCapture/Main Camera (2)_depth_{num-1}.exr"
        hfov = np.deg2rad(73.5)
        sensor_width, sensor_height = 1280, 720
        focal = sensor_width / (2 * np.tan(hfov / 2))
        baseline = 0.075 # 7.5 cm
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0]*1000 # loaded in units of 1000 meters
        gt_disp = (focal * baseline) / depth
        gt_show = gt_disp.copy().astype(np.uint8)
        gt_show = ((gt_show / args.max_disp)*255).astype(np.uint8)

        cv2.imshow("SGBM", show)
        cv2.imshow("GT", gt_show)
        cv2.imshow("left", left)
        cv2.imshow("right", right)

        cv2.waitKey()

    imageio.imwrite(f"{args.dataset}/SGBM/disp_{num}.exr", disp)
    # cv2.imwrite(f"{args.dataset}/SGBM/disp_{num}.png", disp)
