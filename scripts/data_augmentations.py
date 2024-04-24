import albumentations as A  # Importing the albumentations library to use for image augmentations.

def get_train_augmentations():
    """
    Defines and returns various image augmentation pipelines using the albumentations library.
    These augmentations are designed to introduce variability into the training data,
    which can help improve model robustness.

    Returns:
        tuple: A tuple containing four augmentation Compose objects:
               - base_aug: Basic augmentations applied to all images.
               - right_aug: Augmentations specific to right-view images.
               - disp_aug: Augmentations applied to disparity maps.
               - real_aug: Combined basic augmentations applied to both images and disparity maps.
    """
    # Basic augmentations that include cropping, rotating, and flipping
    base_augmentations = [
        A.CropNonEmptyMaskIfExists(height=384, width=640),  # Crop around non-empty mask areas if they exist
        A.Rotate(limit=20),  # Rotate the image within a limit of +/- 20 degrees
        A.HorizontalFlip(p=0.5),  # Apply horizontal flip with a probability of 0.5
    ]

    # Augmentations specific to right-view images including noise and environmental effects
    right_augmentations = [
        A.GaussNoise(p=0.3),  # Apply Gaussian noise with a probability of 0.3
        A.RandomShadow(),  # Add random shadows to simulate varying light conditions
        A.RandomSunFlare(src_radius=200, p=0.1),  # Simulate sun flares with a small probability
        A.RandomRain(p=0.1),  # Add random rain effects with a probability of 0.1
    ]

    # Augmentations specifically designed for modifying disparity maps
    disp_augmentations = [
        A.GaussNoise(var_limit=0.005, p=1),  # Apply Gaussian noise to disparity maps
    ]

    # Compose the basic augmentations with additional targets for disparity
    base_aug = A.Compose(
        base_augmentations,
        additional_targets={"gt_disp": "image"},  # Target for ground truth disparity
    )

    # Compose the basic augmentations with additional targets for both disparity and real images
    real_aug = A.Compose(
        base_augmentations, additional_targets={"gt_disp": "image", "disp": "image"}
    )

    # Compose the right-view specific augmentations
    right_aug = A.Compose(right_augmentations)

    # Compose the disparity specific augmentations
    disp_aug = A.Compose(disp_augmentations)

    return base_aug, right_aug, disp_aug, real_aug
