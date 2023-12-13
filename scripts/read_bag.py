#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import argparse
import time
from tqdm import tqdm

# from rosbags.rosbag1 import Reader
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.image import message_to_cvimage

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, type=str, help="Path to bag")
parser.add_argument('--mode', choices=['show', 'extract', 'mp4'], type=str, help="Mode of log replay")
parser.add_argument('--out', type=str, help="Path to output dir")
parser.add_argument('--skip', type=int, help="Skip this many frames when extracting", default=16)
args = parser.parse_args()

reader = Reader(args.path)
reader.open()

def make_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)

SR_MSGS = [
    '/rect_left',
    '/rect_right',
    '/disparity'
]

topic_to_topic_folder = {
    '/rect_left': 'rect_left',
    '/rect_right': 'rect_right',
    '/disparity': 'disparity'
}

counter = 0

granule = args.path.split('/')[-1].split('.bag')[0]

if args.mode == "extract" and args.out is None:
    raise Exception("Must specify output directory in extract mode!")
elif args.mode == "extract":
    make_directory(args.out)
    for topic in SR_MSGS:
        name = topic_to_topic_folder[topic]
        make_directory(os.path.join(args.out, name))

for conn, timestamp, rawdata in tqdm(reader.messages(connections=reader.connections)):

    if conn.topic in SR_MSGS:

        msg = deserialize_cdr(rawdata, conn.msgtype)

        extracted = message_to_cvimage(msg)

        if conn.topic == "/disparity" and args.mode == "show":
            show_disp = extracted.copy() / 8 # convert to subpixel
            show_disp = (show_disp * (255 / 190)).astype(np.uint8)
            show_disp = cv2.applyColorMap(show_disp, cv2.COLORMAP_JET)

            cv2.imshow(conn.topic, show_disp)

        elif args.mode == "show":
            cv2.imshow(conn.topic, extracted)

        if args.mode == "show": cv2.waitKey()

        if args.mode == "extract":
            if conn.topic == '/disparity':
                counter += 1

            if counter % args.skip == 0:
                name = topic_to_topic_folder[conn.topic]
                out_path = f"{args.out}/{name}/{granule}_{name}_{counter}.png"
                cv2.imwrite(out_path, extracted)
