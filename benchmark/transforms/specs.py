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
        "Resize",
        {
            "target_size": 512,
            "interpolation": "bilinear",
        },
    ),
    TransformSpec(
        "RandomCrop128",
        {
            "height": 128,
            "width": 128,
        },
    ),
    TransformSpec(
        "RandomResizedCrop",
        {
            "size": (512, 512),
            "width": 512,
            "scale": (0.08, 1.0),
            "ratio": (0.75, 1.3333333333333333),
            "interpolation": "bilinear",
        },
    ),
    TransformSpec(
        "CenterCrop128",
        {
            "height": 128,
            "width": 128,
        },
    ),
    TransformSpec(
        "HorizontalFlip",
        {},
    ),
    TransformSpec(
        "VerticalFlip",
        {},
    ),
    TransformSpec(
        "Pad",
        {
            "padding": 10,
            "fill": 0,
            "border_mode": "constant",
        },
    ),
    TransformSpec(
        "Rotate",
        {
            "angle": 45,
            "interpolation": "nearest",
            "mode": "constant",
            "fill": 0,
        },
    ),
    TransformSpec(
        "Affine",
        {
            "angle": 25.0,
            "shift": (20, 20),
            "scale": 2.0,
            "shear": (10.0, 15.0),
            "interpolation": "bilinear",
            "mode": "constant",
            "fill": 0,
        },
    ),
    TransformSpec(
        "Perspective",
        {
            "scale": (0.05, 0.1),
            "interpolation": "bilinear",
            "fill": 0,
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
        },
    ),
    TransformSpec(
        "ColorJitter",
        {
            "brightness": 0.5,
            "contrast": 1.5,
            "saturation": 1.5,
            "hue": 0.5,
        },
    ),
    TransformSpec(
        "ChannelShuffle",
        {},
    ),
    TransformSpec(
        "Grayscale",
        {
            "num_output_channels": 3,
        },
    ),
    TransformSpec(
        "RGBShift",
        {
            "pixel_shift": 100,
        },
    ),
    TransformSpec(
        "GaussianBlur",
        {
            "sigma": 2.0,
            "kernel_size": (5, 5),
        },
    ),
    TransformSpec(
        "GaussianNoise",
        {
            "mean": 0,
            "std": 0.44,
            "per_channel": True,
        },
    ),
    TransformSpec(
        "Invert",
        {},
    ),
    TransformSpec(
        "Posterize",
        {
            "bits": 4,
        },
    ),
    TransformSpec(
        "Solarize",
        {
            "threshold": 0.5,
        },
    ),
    TransformSpec(
        "Sharpen",
        {
            "alpha": (0.2, 0.5),
            "lightness": (0.5, 1),
        },
    ),
    TransformSpec(
        "AutoContrast",
        {},
    ),
    TransformSpec(
        "Equalize",
        {},
    ),
    TransformSpec(
        "Normalize",
        {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
    ),
    TransformSpec(
        "Erasing",
        {
            "scale": (0.02, 0.33),
            "ratio": (0.3, 3.3),
            "fill": 0,
        },
    ),
    TransformSpec(
        "JpegCompression",
        {
            "quality": 50,
        },
    ),
    TransformSpec(
        "RandomGamma",
        {
            "gamma": 120,
        },
    ),
    TransformSpec(
        "PlankianJitter",
        {
            "mode": "blackbody",
        },
    ),
    TransformSpec(
        "MedianBlur",
        {
            "blur_limit": 5,
        },
    ),
    TransformSpec(
        "MotionBlur",
        {
            "kernel_size": 5,
            "angle_range": (0, 360),
            "direction_range": (-1, 1),
        },
    ),
    TransformSpec(
        "CLAHE",
        {
            "clip_limit": (1, 4),
            "tile_grid_size": (8, 8),
        },
    ),
    TransformSpec(
        "Brightness",
        {
            "brightness_limit": (0.2, 0.2),
        },
    ),
    TransformSpec(
        "Contrast",
        {
            "contrast_limit": (0.2, 0.2),
        },
    ),
    TransformSpec(
        "CoarseDropout",
        {
            "hole_height_range": (0.1, 0.1),
            "hole_width_range": (0.1, 0.1),
            "num_holes_range": (4, 4),
        },
    ),
    TransformSpec(
        "Blur",
        {
            "radius": 5,
            "border_mode": "constant",
        },
    ),
    TransformSpec(
        "HSV",
        {
            "hue": 0.015,
            "saturation": 0.7,
            "value": 0.4,
        },
    ),
    TransformSpec(
        "ChannelDropout",
        {},
    ),
    TransformSpec(
        "LinearIllumination",
        {
            "gain": (0.01, 0.2),
        },
    ),
    TransformSpec(
        "CornerIllumination",
        {
            "gain": (0.01, 0.2),
        },
    ),
    TransformSpec(
        "GaussianIllumination",
        {
            "gain": (0.01, 0.2),
        },
    ),
    TransformSpec(
        "Hue",
        {
            "hue": 20,
        },
    ),
    TransformSpec(
        "PlasmaBrightness",
        {
            "roughness": 0.5,
        },
    ),
    TransformSpec(
        "PlasmaContrast",
        {
            "roughness": 0.5,
        },
    ),
    TransformSpec(
        "PlasmaShadow",
        {
            "roughness": 0.5,
        },
    ),
    TransformSpec(
        "Rain",
        {
            "drop_width": 5,
            "drop_height": 20,
            "brightness_coefficient": 1.0,
        },
    ),
    TransformSpec(
        "SaltAndPepper",
        {
            "amount": (0.01, 0.06),
            "salt_vs_pepper": (0.4, 0.6),
        },
    ),
    TransformSpec(
        "Saturation",
        {
            "saturation_factor": 0.5,
        },
    ),
    TransformSpec(
        "Snow",
        {"snow_point_range": (0.5, 0.5)},
    ),
    TransformSpec(
        "OpticalDistortion",
        {
            "distort_limit": 0.5,
        },
    ),
    TransformSpec(
        "Shear",
        {
            "shear": 10,
        },
    ),
    TransformSpec(
        "ThinPlateSpline",
        {
            "num_control_points": 2,
            "distortion_scale": 0.5,
        },
    ),
]

# Use the same TRANSFORM_SPECS for both images and videos
# No need for separate VIDEO_TRANSFORM_SPECS
