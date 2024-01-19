import numpy as np
import glob, os
import json
import yaml
import argparse

'''
This script is used for handling dataset splitting for training, validation, and test sets. 
It reads configuration parameters, including dataset root and alignment information. 
Depending on the presence of "force_val.txt," it either forces certain samples into validation or shuffles paths randomly. 
The resulting splits (train and validation (80:20)) are saved to JSON file

'''

# PARAMETERS
SEED=42
VALIDATION_RATIO=0.8

# Parse command-line arguments for the configuration file and seed
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, help="Configuration file to use")
parser.add_argument("--seed", type=int, default=SEED)
args = parser.parse_args()

# Load configuration parameters from the specified file
with open(args.cfg) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

# Extract dataset root and alignment information
root = cfg["DATASET"]["ROOT"]
gra = cfg["DATASET"]["GT_RIGHT_ALIGN"]
gt_gran = "right" if gra else "left"


np.random.seed(args.seed)

splits = {"train": [], "val": [], "test": []}  # not implemented for now

# Check for the existence of "force_val.txt"
force_val_path = f"{root}/force_val.txt"
if os.path.exists(force_val_path):
    # Read granules from "force_val.txt"
    with open(force_val_path) as f:
        lines = f.readlines()
        force_val = [line[:-1] for line in lines]

    # paths = glob.glob(f"{root}/*{gt_gran}/*.*")
    paths = glob.glob(f"{root}/ann/*.*")

# Iterate through paths and assign to train or val based on "force_val"
    for path in paths:
        path = path.replace("/ann/", "/rect_right/").replace("_mask_", "_rect_right_")
        granule = path.split("/")[-1].split("_rect")[0]
        if granule in force_val:
            splits["val"].append(path)
        else:
            splits["train"].append(path)


 # If "force_val.txt" doesn't exist, shuffle paths randomly
else:
    # paths = sorted(glob.glob(f"{root}/*{gt_gran}/*.*"))
    paths = sorted(glob.glob(f"{root}/ann/*.*"))
    paths = [
        path.replace("/ann/", "/rect_right/").replace("_mask_", "_rect_right_")
        for path in paths
    ]
    np.random.shuffle(paths)

    splits["train"] = paths[: int(len(paths) * VALIDATION_RATIO)]
    splits["val"] = paths[int(len(paths) * VALIDATION_RATIO) :]

# Save the resulting splits into a JSON file 
with open(root + "/jjloader.json", "w") as f:
    json.dump(splits, f)
