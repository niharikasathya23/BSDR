import os
import numpy as np
import torch
import cv2
import glob
import argparse

from segment_anything import sam_model_registry, SamPredictor

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, type=str, help="Path to extracted frames directory")
args = parser.parse_args()

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

paths = glob.glob(os.path.join(args.path, "rect_right", "*.png"))
np.random.shuffle(paths)

"""
z = draw human box
x = add human point
c = remove human point
v = draw object box
b = add object point
n = remove object point
"""

drawing = False
new_box, new_pnt = False, False
pnt = np.zeros(2)
box = np.zeros(4)
hum_boxes, obj_boxes = [], []
hum_pnts, obj_pnts = [], []
hum_lbls, obj_lbls = [], []
hum_mask, obj_mask = np.zeros((400,640)).astype(np.uint8), np.zeros((400,640)).astype(np.uint8)
state = -1
def box_fn(event, x, y, flags, param):
    global image, cache, box, pnt, drawing, new_box, new_pnt, state
    if event == cv2.EVENT_LBUTTONDOWN and (state==0 or state==3):
        cache = image.copy()
        box[0] = x
        box[1] = y
        drawing = True
    if event == cv2.EVENT_LBUTTONDOWN and (state in [1,2,4,5]):
        pnt[0] = x
        pnt[1] = y
        new_pnt = True
    if event == cv2.EVENT_LBUTTONUP and (state==0 or state==3):
        box[2] = x
        box[3] = y
        # cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[2]),round(box[3])), (255,0,0), 2)
        drawing = False
        new_box = True
    if drawing:
        image = cache.copy()
        color = (255,0,0) if state==0 else (0,0,255)
        cv2.rectangle(image, (round(box[0]),round(box[1])), (round(x),round(y)), color, 2)

cv2.namedWindow("preview")
cv2.setMouseCallback("preview", box_fn)

def draw_mask(image):
    mask_show = np.zeros((400,640,3)).astype(np.uint8)
    mask_show[hum_mask==1]=[255,0,0]
    mask_show[obj_mask==1]=[0,0,255]
    image = cv2.addWeighted(image, 0.75, mask_show, 0.25, 0)
    return image

def resetImage(image, empty, key):
    image = empty.copy()
    for box in hum_boxes:
        cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[2]),round(box[3])), (255,0,0), 2)
    for box in obj_boxes:
        cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[2]),round(box[3])), (0,0,255), 2)
    image = draw_mask(image)
    org = (0,20)
    if key == ord('z'):
        cv2.putText(image, "Drawing human box", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    elif key == ord('x'):
        cv2.putText(image, "Adding human point", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    elif key == ord('c'):
        cv2.putText(image, "Subtracting human point", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    elif key == ord('v'):
        cv2.putText(image, "Drawing object box", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    elif key == ord('b'):
        cv2.putText(image, "Adding object point", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    elif key == ord('n'):
        cv2.putText(image, "Subtracting object point", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return image

for path in paths:
    write_path = path.replace('/rect_right/', '/ann/').replace('_rect_right_', '_mask_')
    if os.path.exists(write_path):
        continue

    image = cv2.imread(path)
    empty = image.copy()

    predictor.set_image(image)

    while True:

        cv2.imshow("preview", image)

        if new_box and state==0:
            hum_boxes.append(box.copy())
        if new_box and state==3:
            obj_boxes.append(box.copy())
        if new_pnt and state in [1,2]:
            hum_pnts.append(pnt.copy())
            if state==1:
                hum_lbls.append(1)
            else:
                hum_lbls.append(0)
        if new_pnt and state in [4,5]:
            obj_pnts.append(pnt.copy())
            if state==4:
                obj_lbls.append(1)
            else:
                obj_lbls.append(0)
        if new_box or new_pnt:
            if state in [0,1,2]:
                coords = None if len(hum_pnts)==0 else np.array(hum_pnts)
                labels = None if len(hum_lbls)==0 else np.array(hum_lbls)
                masks, _, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=np.array(hum_boxes[0]),
                    multimask_output=False,
                )

                hum_mask = np.zeros((400,640)).astype(np.uint8)
                for mask in masks:
                    hum_mask[mask] = 1

            elif state in [3,4,5]:
                coords = None if len(obj_pnts)==0 else np.array(obj_pnts)
                labels = None if len(obj_lbls)==0 else np.array(obj_lbls)
                masks, _, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=np.array(obj_boxes[0]),
                    multimask_output=False,
                )

                obj_mask = np.zeros((400,640)).astype(np.uint8)
                for mask in masks:
                    obj_mask[mask] = 1

            # image = draw_mask(image)
            image = resetImage(image, empty, key)

            new_box, new_pnt = False, False

        key = cv2.waitKey(1)

        if key == ord('z'):
            state = 0
            image = resetImage(image, empty, key)
        elif key == ord('x'):
            state = 1
            image = resetImage(image, empty, key)
        elif key == ord('c'):
            state = 2
            image = resetImage(image, empty, key)
        elif key == ord('v'):
            state = 3
            image = resetImage(image, empty, key)
        elif key == ord('b'):
            state = 4
            image = resetImage(image, empty, key)
        elif key == ord('n'):
            state = 5
            image = resetImage(image, empty, key)
        elif key == ord('s'):
            # SAVE!
            final_mask = np.zeros((400,640,3)).astype(np.uint8)
            final_mask[hum_mask==1] = [255,0,0]
            final_mask[obj_mask==1] = [0,0,255]
            cv2.imshow('final', final_mask)
            write_path = path.replace('/rect_right/', '/ann/').replace('_rect_right_', '_mask_')
            cv2.imwrite(write_path, final_mask)

            hum_boxes, obj_boxes = [], []
            hum_pnts, obj_pnts = [], []
            hum_lbls, obj_lbls = [], []
            hum_mask, obj_mask = np.zeros((400,640)).astype(np.uint8), np.zeros((400,640)).astype(np.uint8)
            state = -1
            break
