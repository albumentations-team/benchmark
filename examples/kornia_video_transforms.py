from typing import Any

import kornia.augmentation as K
import torch

# Required: Library name for dependency installation
LIBRARY = "kornia"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Required: Define how to apply transforms to videos
def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    """Apply kornia transform to video tensor

    Args:
        transform: Kornia augmentation instance
        video: torch.Tensor of shape (T, C, H, W)

    Returns:
        Transformed video as torch.Tensor
    """
    # Move to device
    video = video.to(device)

    # Use float16 on GPU for performance
    if device.type == "cuda":
        video = video.half()

    # Kornia treats time dimension as batch dimension
    # Most transforms should have same_on_batch=True for consistent transform across frames
    return transform(video)


# Required: Transform definitions with explicit names
TRANSFORMS = [
    {
        "name": "RandomGrayscale",
        "transform": K.RandomGrayscale(p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomSolarize_low",
        "transform": K.RandomSolarize(thresholds=0.3, p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomSolarize_high",
        "transform": K.RandomSolarize(thresholds=0.7, p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomHorizontalFlip",
        "transform": K.RandomHorizontalFlip(p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomVerticalFlip",
        "transform": K.RandomVerticalFlip(p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomGaussianBlur_3x3",
        "transform": K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomGaussianBlur_5x5",
        "transform": K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomBrightness_weak",
        "transform": K.RandomBrightness(brightness=(0.9, 1.1), p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomBrightness_strong",
        "transform": K.RandomBrightness(brightness=(0.5, 1.5), p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomContrast_weak",
        "transform": K.RandomContrast(contrast=(0.9, 1.1), p=1, same_on_batch=True).to(device),
    },
    {
        "name": "RandomContrast_strong",
        "transform": K.RandomContrast(contrast=(0.5, 1.5), p=1, same_on_batch=True).to(device),
    },
]
