"""Example custom image transform specifications.

This shows how to create custom transforms with different parameter values
or test specific transform combinations.
"""

from typing import Any

import albumentations as A
import cv2
import numpy as np

# Required: Library name for dependency installation
LIBRARY = "albumentationsx"


# Required: Define how to apply transforms to images
def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    """Apply AlbumentationsX transform to a single image

    Args:
        transform: AlbumentationsX transform instance
        image: numpy array of shape (H, W, C)

    Returns:
        Transformed image as numpy array
    """
    return np.ascontiguousarray(transform(image=image)["image"])


# Required: Transform definitions
TRANSFORMS = [
    # Test different blur kernel sizes
    {
        "name": "GaussianBlur_3x3",
        "transform": A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=1),
    },
    {
        "name": "GaussianBlur_5x5",
        "transform": A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 2.0), p=1),
    },
    {
        "name": "GaussianBlur_7x7",
        "transform": A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.1, 2.0), p=1),
    },
    # Test different rotation angles
    {
        "name": "Rotate_15",
        "transform": A.Rotate(limit=15, interpolation=cv2.INTER_LINEAR, p=1),
    },
    {
        "name": "Rotate_45",
        "transform": A.Rotate(limit=45, interpolation=cv2.INTER_LINEAR, p=1),
    },
    {
        "name": "Rotate_90",
        "transform": A.Rotate(limit=90, interpolation=cv2.INTER_LINEAR, p=1),
    },
    # Test different brightness levels
    {
        "name": "Brightness_0.1",
        "transform": A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=(0.0, 0.0), p=1),
    },
    {
        "name": "Brightness_0.3",
        "transform": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=(0.0, 0.0), p=1),
    },
    {
        "name": "Brightness_0.5",
        "transform": A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=(0.0, 0.0), p=1),
    },
    # Test composed transforms
    {
        "name": "BasicAugmentation",
        "transform": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ],
            p=1,
        ),
    },
    {
        "name": "StrongAugmentation",
        "transform": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ],
            p=1,
        ),
    },
    # Test specific parameter combinations
    {
        "name": "ColorJitter_AllWeak",
        "transform": A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1),
    },
    {
        "name": "ColorJitter_BrightnessOnly",
        "transform": A.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0, p=1),
    },
    {
        "name": "ColorJitter_ContrastOnly",
        "transform": A.ColorJitter(brightness=0, contrast=0.3, saturation=0, hue=0, p=1),
    },
]
