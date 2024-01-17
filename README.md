# Bilateral Segmentation and Disparity Refinement

## Overview

This repository contains code for collecting data, training, evaluating, and running inference using BSDR for a complete machine learning development loop.

Overall status:
- Collecting Data: Done
- Training: TODO
- Evaluating: TODO 
- Inference: TODO

## Repository Structure

## Setup

To install the necessary python libraries, run
```
pip install -r requirements.txt
```
*Note*: we recommend using a virtual environment (venv) or conda. python3.8 or above is required

Thus far, this code has only been tested with ROS2 Foxy. However, feel free to try with a different ROS2 distribution. For Foxy, run
```
source /opt/ros/foxy/setup.bash
```
to activate ROS2 python libraries (specifically `rclpy`).

## Running Application

The applications can be run with `app/main.py`.
```
cd app/
python3 main.py -h
```

### Collection Only

Here is an example of running collection only
```
python3 main.py --log /path/to/save/ros2bag
```

### Inference Only

TODO

### Collection and Inference

TODO

## Pre-processing Collected Data

### 1. Extract images from ROS2 bag

To run this script
```
cd scripts/
python3 read_bag.py --path /path/to/ros2bag --mode extract --out /path/to/save/extracted/frames --skip 16
```
The `--skip` parameter specifies how many frames to skip before saving the frame. Since the ROS2 bag likely has a lot of similar frames, this can be used to reduce the number of images without reducing the amount of information.

### 2. Generating ground truth depth maps with CREStereo

To run this script
```
cd scripts/
python3 generate_gt_disp.py --path /path/to/extracted/frames
```
You can optionally use the `--debug` flag to visualize the CREStereo results and check they are as expected.

### 3. Using SAM 

You will need the sam_vit_h_4b8939.pth model checkpoint (2.4 GB file) which can be downloaded from [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints).


