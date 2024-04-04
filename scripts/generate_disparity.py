import cv2
import numpy as np
import os
from scipy.signal import medfilt

def compute_disparity_sgbm(left_image_path, right_image_path, disparity_output_path, median_filter=True):
    # Read the left and right images
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Set up the stereo SGBM matcher
    window_size = 5
    min_disp = 0
    num_disp = 64 - min_disp  # Adjust, must be divisible by 16
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the disparity map
    disparity = stereo.compute(left_image, right_image)

    # Apply median filter to smooth the disparity map
    if median_filter:
        disparity = medfilt(disparity, kernel_size=5)

    # Normalize the disparity map for display
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Save the disparity map
    cv2.imwrite(disparity_output_path, disparity_normalized)
    print(f"Disparity map saved to {disparity_output_path}")

# Define the paths for left and right images
base_dir = "/home/nataliya/bsdr/scripts/frames_new"  # Replace with the path to your 'frames_new' directory
left_dir = os.path.join(base_dir, "rect_left")
right_dir = os.path.join(base_dir, "rect_right")

# Directory where the disparity maps will be saved inside frames_new
disparity_dir = os.path.join(base_dir, "disparity")

# Create the output directory if it doesn't exist
if not os.path.exists(disparity_dir):
    os.makedirs(disparity_dir)

# Find pairs of left and right images
left_images = os.listdir(left_dir)
right_images = os.listdir(right_dir)
pairs = zip(sorted(left_images), sorted(right_images))

# Process each pair of images
for left_image, right_image in pairs:
    left_image_path = os.path.join(left_dir, left_image)
    right_image_path = os.path.join(right_dir, right_image)
    output_filename = left_image.replace('rect_left', 'disparity')  # Assuming the left images have 'rect_left' in their filenames
    disparity_output_path = os.path.join(disparity_dir, output_filename)
    
    compute_disparity_sgbm(left_image_path, right_image_path, disparity_output_path)

print("Finished generating disparity maps.")
