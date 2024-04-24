from torch.utils.data import DataLoader
from argparse import ArgumentParser
from config_loader import load_config
from dataset_load_process import HumanReal

import cv2
import cmapy
import numpy as np

"""
This script processes and visualizes stereo image data for evaluation purposes.
It efficiently handles image batches using PyTorch's DataLoader, applies transformations,
and visualizes results such as disparity maps and segmentation masks to aid in the 
development and debugging of stereo vision models.

TO RUN THIS FILE: 

We need kd.yaml and loader.json file for the dataset we want to run real.py on

- Navigate to scripts folder also global search and replace paths(cmd+shift+h) according to your system.
- Go to kd.yaml and change the dataset path
- Run split.py file to generate the loader.jsom for specific dataset which splits the dataset in 80:20 ratio for train and validation. 
Command to run split.py: python3 split.py --cfg configs/kd.yaml
- Run real.py using the command: python3 real.py --cfg configs/kd.yaml

"""

def main(config_path):
    """
    Main function to load dataset, process images, and display visualization.

    Args:
    config_path (str): Path to the configuration YAML file specifying dataset details.
    """
    # Load configuration settings for the dataset
    config = load_config(config_path)

    # Initialize the dataset
    dataset = HumanReal(config["DATASET"]["ROOT"], "val", dataset_cfg=config["DATASET"])

    # Create a DataLoader to handle the dataset
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    # Iterate through the DataLoader to process and visualize data
    for img, lbl, gt_disp, err in loader:
        # Print the shapes of the loaded tensors for debugging
        print("Batch shapes:")
        print("Labels: ", lbl.shape)
        print("Ground Truth Disparity: ", gt_disp.shape)
        print("Error: ", err.shape)

        # Convert the batched tensors to numpy arrays for visualization
        img = img.squeeze().numpy()
        right = img[0, :, :]
        disp = img[1, :, :]
        seg = lbl.squeeze().numpy()
        gt_disp = gt_disp.squeeze().numpy()
        err = err.squeeze().numpy()

        # Display minimum and maximum values for each component for debugging
        print("Data Ranges:")
        print("Disparity - Min: {}, Max: {}".format(np.min(disp), np.max(disp)))
        print("Ground Truth Disparity - Min: {}, Max: {}".format(np.min(gt_disp), np.max(gt_disp)))
        print("Error - Min: {}, Max: {}".format(np.min(err), np.max(err)))

        # Visualize the images using OpenCV
        gt_disp_show = ((gt_disp * 255) / np.max(gt_disp)).astype(np.uint8)
        gt_disp_show = cv2.applyColorMap(gt_disp_show, cv2.COLORMAP_JET)
        disp_show = ((disp * 255)).astype(np.uint8)
        disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
        err_show = ((err + 1) / 2 * 255 / np.max(gt_disp)).astype(np.uint8)
        err_show = cv2.applyColorMap(err_show, cmapy.cmap("seismic"))

        # Display the processed images
        cv2.imshow("Right View", (right * 255).astype(np.uint8))
        cv2.imshow("Segmentation Mask", (seg * 255).astype(np.uint8))
        cv2.imshow("Ground Truth Disparity", gt_disp_show)
        cv2.imshow("Computed Disparity", disp_show)
        cv2.imshow("Disparity Error", err_show)

        cv2.waitKey(0)  # Wait for a key press to proceed to the next image

if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser(description="Load and visualize dataset images and data.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.cfg)  # Execute the main function with the provided config path
