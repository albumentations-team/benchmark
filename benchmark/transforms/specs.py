from dataclasses import dataclass
from typing import Any


@dataclass
class TransformSpec:
    """Base class that defines exact parameters for each transform"""

    name: str
    params: dict[str, Any]

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"


# Define all transform specifications based on the original implementations
TRANSFORM_SPECS = [
    TransformSpec(
        "HorizontalFlip",
        {
            "p": 1.0,
        },
    ),
    TransformSpec(
        "VerticalFlip",
        {
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Rotate",
        {
            "angle": 45,
            "p": 1.0,
            "interpolation": "nearest",
            "mode": "reflect",  # from imgaug implementation
        },
    ),
    TransformSpec(
        "Affine",
        {
            "angle": 25.0,
            "shift": (50, 50),
            "scale": 2.0,
            "shear": [10.0, 15.0],
            "p": 1.0,
            "interpolation": "bilinear",
            "mode": "reflect",  # from imgaug implementation
        },
    ),
    TransformSpec(
        "Equalize",
        {
            "p": 1.0,
        },
    ),
    TransformSpec(
        "RandomCrop80",
        {
            "height": 80,
            "width": 80,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "RandomResizedCrop",
        {
            "height": 512,
            "width": 512,
            "scale": (0.08, 1.0),
            "ratio": (0.75, 1.3333333333333333),
            "interpolation": "bilinear",
            "p": 1.0,
        },
    ),
    TransformSpec(
        "ShiftRGB",
        {
            "pixel_shift": 100,
            "p": 1.0,
            "per_channel": True,  # from imgaug implementation
        },
    ),
    TransformSpec(
        "Resize",
        {
            "target_size": 512,
            "interpolation": "bilinear",
            "p": 1.0,
        },
    ),
    TransformSpec(
        "RandomGamma",
        {
            "gamma": 120,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Grayscale",
        {
            "p": 1.0,
            "num_output_channels": 3,  # from torchvision implementation
        },
    ),
    TransformSpec(
        "ColorJitter",
        {
            "brightness": 0.5,
            "contrast": 1.5,
            "saturation": 1.5,
            "hue": 0.5,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "PlankianJitter",
        {
            "mode": "blackbody",
            "p": 1.0,
        },
    ),
    TransformSpec(
        "RandomPerspective",
        {
            "scale": (0.05, 0.1),
            "p": 1.0,
            "interpolation": "bilinear",
        },
    ),
    TransformSpec(
        "GaussianBlur",
        {
            "sigma": 2.0,
            "kernel_size": (5, 5),
            "p": 1.0,
        },
    ),
    TransformSpec(
        "MedianBlur",
        {
            "blur_limit": 5,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "MotionBlur",
        {
            "kernel_size": 5,
            "angle": 45,
            "direction": 0.0,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Posterize",
        {
            "bits": 4,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "JpegCompression",
        {
            "quality": 50,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "GaussianNoise",
        {
            "mean": 127,
            "var": 0.010,
            "per_channel": True,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Elastic",
        {
            "alpha": 50.0,
            "sigma": 5.0,
            "interpolation": "bilinear",
            "approximate": False,
            "same_dxdy": True,
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Normalize",
        {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Clahe",
        {
            "clip_limit": (1, 4),
            "tile_grid_size": (8, 8),
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Brightness",
        {
            "brightness_limit": (0.2, 0.2),
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Contrast",
        {
            "contrast_limit": (0.2, 0.2),
            "p": 1.0,
        },
    ),
    TransformSpec(
        "CoarseDropout",
        {
            "hole_height_range": (16, 16),
            "hole_width_range": (16, 16),
            "num_holes_range": (1, 1),
            "p": 1.0,
        },
    ),
    TransformSpec(
        "Blur",
        {
            "radius": 5,
            "p": 1.0,
        },
    ),
]
