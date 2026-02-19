"""Template for custom video transform specifications.

Define LIBRARY, __call__, and TRANSFORMS.
"""

from typing import Any

import albumentations as A
import numpy as np

# Required: Specify which library (for dependency installation)
LIBRARY = "albumentationsx"  # Options: "albumentationsx", "torchvision", "kornia", "imgaug", "augly"


# Required: Define how to apply transforms to videos
def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    """Apply transform to video using library-specific API

    Args:
        transform: Transform instance from the library
        video: Video data in library-specific format
               - albumentationsx: numpy array (T, H, W, C)
               - torchvision/kornia: torch tensor (T, C, H, W)

    Returns:
        Transformed video in the same format as input
    """
    result = transform(images=video)["images"]
    return np.ascontiguousarray(result)


# Required: List of transforms to benchmark
TRANSFORMS = [
    # Example 1: Simple transform
    {
        "name": "HorizontalFlip",
        "transform": A.HorizontalFlip(p=1),
    },
    # Example 2: Transform with parameters in name
    {
        "name": "GaussianBlur_5x5",
        "transform": A.GaussianBlur(blur_limit=(5, 5), p=1),
    },
    # Example 3: Multiple versions of same transform
    {
        "name": "Rotate_15",
        "transform": A.Rotate(limit=15, p=1),
    },
    {
        "name": "Rotate_45",
        "transform": A.Rotate(limit=45, p=1),
    },
    {
        "name": "Rotate_90",
        "transform": A.Rotate(limit=90, p=1),
    },
    # Example 4: Composed transform
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
]
