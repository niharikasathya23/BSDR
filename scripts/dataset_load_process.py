import torch
from torch import Tensor  # Add this import for type annotations
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
from scipy.signal import medfilt
from data_augmentations import get_train_augmentations
from typing import Tuple


class HumanReal(Dataset):
    """
    A dataset class for handling and preprocessing stereo image datasets for deep learning tasks.
    This class is configured to manage data involving tasks like image segmentation and
    disparity estimation, with functionalities to apply various image augmentations and
    transformations for enhancing model training and evaluation.

    Attributes:
        root (str): Directory where datasets are stored.
        split (str): Type of dataset split ('train', 'val', 'test').
        dataset_cfg (dict): Configuration for dataset-specific settings like path names and processing parameters.
        transform (callable, optional): Optional transformation to apply to each image pair.
        load_path (bool): If True, returns the file path along with the image data and labels.
    """
    # CLASSES = ['background', 'human']
    # PALETTE = torch.tensor([[0, 255, 0], [0, 0, 255]])
    CLASSES = ["background", "human", "object"]
    PALETTE = torch.tensor([[0, 255, 0], [0, 0, 255], [0, 255, 0]])

    def __init__(self, root: str, split: str = "train", dataset_cfg=None, transform=None, load_path=False):
        """
        Initializes the dataset with necessary directory settings and configuration.

        Parameters:
            root (str): The base directory of the dataset.
            split (str): The dataset split, can be 'train', 'val', or 'test'.
            dataset_cfg (dict): Configuration dictionary specifying dataset parameters.
            transform (callable, optional): Transformation function to apply to each data item.
            load_path (bool): Whether to include the path of the data in the returned items.
        """
        super().__init__()
        assert split in ["train", "val", "test"]
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255  # Label to ignore in calculations, typical in segmentation tasks
        self.max_disp = dataset_cfg.get("MAX_DISP", 192)  # Default max disparity if not specified in cfg
        self.load_path = load_path  # Flag to return image paths for logging/debugging

        self.root = root
        self.split = split
        print("self.root\n", self.root)
        with open(os.path.join(root, "loader.json")) as f:
            self.mono_paths = json.load(f)[self.split] # Load paths for images based on dataset split

        # Configuration settings from dataset_cfg dict
        self.gra = dataset_cfg.get("GT_RIGHT_ALIGN", False)  # Determines if images are right-aligned
        self.mono_is = "right" if self.gra else "left"  # Adjust path based on image alignment
        self.other_is = "left" if self.gra else "right"
        self.exr_disp = dataset_cfg.get("EXR_DISP", False)  # Disparity data in EXR format
        self.load_filtered = dataset_cfg.get("LOAD_FILTERED", False)  # Use filtered annotations
        self.median = dataset_cfg.get("MEDIAN_FILTER", False)  # Apply median filter to disparity
        self.norm_img = dataset_cfg.get("NORM_IMG", True)  # Normalize images
        self.norm_disp = dataset_cfg.get("NORM_DISP", True)  # Normalize disparity
        self.norm_gt_disp = dataset_cfg.get("NORM_GT_DISP", True)  # Normalize ground truth disparity

        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Enable OpenEXR format in OpenCV

        # Retrieve transformation functions for image processing
        self.base_transform, self.right_transform, self.disp_transform, self.real_transform = get_train_augmentations()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.mono_paths)

    def preprocess(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Loads and preprocesses an image pair and associated data at the specified index, including handling disparities
        and segmentation masks.

        Parameters:
            index (int): Index of the data pair to preprocess.

        Returns:
            Tuple[Tensor, Tensor]: Tensors for the preprocessed image data and labels.
        """
        # Retrieve paths for the current index from the dataset path list
        mono_path = self.mono_paths[index]
        num = int(mono_path.split("_")[-1].split(".png")[0])  # Extract frame number from filename
        granule_split = mono_path.split("/")[-1].split(f"_rect_{self.mono_is}")
        granule = f"{granule_split[0]}_" if len(granule_split) else ""
        other_path = f"{self.root}/rect_{self.other_is}/{granule}rect_{self.other_is}_{num}.png"

        # Determine paths based on whether the dataset is filtered
        ann_path = f"{self.root}/ann_filtered/{granule}mask_{num}.png" if self.load_filtered else f"{self.root}/ann/{granule}mask_{num}.png"
        gt_disp_path = f"{self.root}/gt_disp/{granule}disp_{num}.exr"
        sgbm_path = f"{self.root}/disparity/disp_{num}.exr" if self.exr_disp else f"{self.root}/disparity/{granule}disparity_{num}.png"

        # Conditional paths based on alignment configuration
        if self.gra:
            left_path = other_path
            right_path = mono_path
        else:
            left_path = mono_path
            right_path = other_path

        # Load right image and handle failure to load
        image = cv2.imread(right_path)
        if image is None:
            print(f"Failed to load image at {right_path}")
            return None  # Return None if image cannot be loaded to avoid further processing
        right = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and preprocess disparity maps and segmentation masks
        gt_disp = cv2.imread(gt_disp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        rightImg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Load segmentation mask and create segmentation labels
        seg_key = cv2.imread(ann_path)
        seg = np.zeros_like(rightImg)
        seg[seg_key[..., 0] == 255] = 1  # Background
        seg[seg_key[..., 2] == 255] = 2  # Object

        # Load disparity and apply median filtering if configured
        disp = cv2.imread(sgbm_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if not self.exr_disp:
            disp = disp / 8  # Normalize for 8-bit disparity images
        if self.median:
            disp = medfilt(disp, 7)
        if self.norm_disp:
            disp = disp / self.max_disp  # Normalize disparity to a scale

        # Apply base transformations if any are configured
        if self.base_transform:
            transformed = self.base_transform(image=rightImg, gt_disp=gt_disp, disp=disp, masks=[seg])
            rightImg = transformed["image"]
            gt_disp = transformed["gt_disp"]
            disp = transformed["disp"]
            seg = transformed["masks"][0]

        if self.right_transform:
            transformed = self.right_transform(image=rightImg)
            rightImg = transformed["image"]

        if self.norm_img:
            rightImg = rightImg / 255  # Normalize image pixel values

        # Pack the processed images and labels into tensors
        image = np.stack([rightImg, disp])
        image = torch.tensor(image).float()
        label = torch.tensor(seg).unsqueeze(0)
        gt_disp = torch.tensor(gt_disp).float()

        err = (gt_disp - disp).float()

        # Decide whether to return paths with tensors or not
        if self.load_path:
            return image, label.squeeze().long(), gt_disp, mono_path
        else:
            return image, label.squeeze().long(), gt_disp, err
            

    # def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
    #     try:
    #         return self.preprocess(index)
    #     except Exception as e:
    #         print(f"EXCEPTION: {e}")
    #         index = np.random.choice(self.__len__())
    #         return self.preprocess(index)
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
    Retrieves an item from the dataset at a specified index. If preprocessing fails, it retries with a different index.

    Parameters:
        index (int): Index for the data item.

    Returns:
        Tuple[Tensor, Tensor]: Preprocessed data and labels.
    """
        while True:
            try:
                result = self.preprocess(index)
                if result is not None:
                    return result
                else:
                    raise ValueError("Failed to preprocess image")
            except Exception as e:
                print(f"EXCEPTION: {e}")
                index = np.random.choice(self.__len__())