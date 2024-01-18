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

drawing = False
new_box, new_pnt = False, False
box = np.zeros(4)
obj_boxes, obj_pnts, obj_lbls = [], [], []
obj_mask = np.zeros((400,640)).astype(np.uint8)
state = -1

def box_fn(event, x, y, flags, param):
    global image, cache, box, drawing, new_box, state
    if event == cv2.EVENT_LBUTTONDOWN and state == 3:
        # Start drawing a new box
        box[0], box[1] = x, y
        cache = image.copy()
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and state == 3 and drawing:
        # Update the temporary image with the rectangle
        image = cache.copy()
        cv2.rectangle(image, (int(box[0]), int(box[1])), (x, y), (0, 0, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP and state == 3 and drawing:
        # Finish drawing the current box
        box[2], box[3] = x, y
        drawing = False
        new_box = True
        obj_boxes.append(box.copy())  # Append the new box to the list
        box = np.zeros(4)  # Reset the box for the next drawing


cv2.namedWindow("preview")
cv2.setMouseCallback("preview", box_fn)

def draw_mask(image):
    mask_show = np.zeros((400,640,3)).astype(np.uint8)
    mask_show[obj_mask==1]=[0,0,255]
    image = cv2.addWeighted(image, 0.75, mask_show, 0.25, 0)
    return image

def resetImage(image, empty, key):
    image = empty.copy()
    for box in obj_boxes:
        cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[2]),round(box[3])), (0,0,255), 2)
    image = draw_mask(image)
    org = (0,20)
    if key == ord('v'):
        cv2.putText(image, "Drawing object box", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
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

        if new_box and state == 3:
            obj_boxes.append(box.copy())
            new_box = False

        if obj_boxes:
            coords = None
            labels = None
            box_array = np.array(obj_boxes)
            
            # Ensure box_array is 2D even for a single box
            if box_array.ndim == 1:
                box_array = np.expand_dims(box_array, axis=0)

            try:
                masks, _, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=box_array,
                    multimask_output=False,
                )

                obj_mask = np.zeros((400,640)).astype(np.uint8)
                for mask in masks:
                    obj_mask[mask] = 1
            except RuntimeError as e:
                print(f"Error during prediction: {e}")
                # Handle the error or skip this iteration

            image = resetImage(image, empty, key)

        key = cv2.waitKey(1)

        if key == ord('v'):
            state = 3
            image = resetImage(image, empty, key)
        elif key == ord('s'):
            # Save the image
            final_mask = np.zeros((400,640,3)).astype(np.uint8)
            final_mask[obj_mask==1] = [0,0,255]
            cv2.imwrite(write_path, final_mask)
            obj_boxes = []
            obj_mask = np.zeros((400,640)).astype(np.uint8)
            state = -1
            break