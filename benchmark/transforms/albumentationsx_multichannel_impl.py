r"""AlbumentationsX multi-channel image benchmark spec.

Tests transforms on 9-channel images (3x stacked RGB, simulating e.g. hyperspectral data).
The runner synthesizes these in-memory via make_multichannel_loader; no special dataset needed.

Run with:
    python -m benchmark.cli run \\
        --spec benchmark/transforms/albumentationsx_multichannel_impl.py \\
        --data-dir /path/to/images \\
        --output output/multichannel \\
        --num-channels 9

Excluded (RGB-semantic, meaningless / broken on arbitrary channel counts):
    RGBShift, PlankianJitter, Rain, Snow, PhotoMetricDistort
"""

from typing import Any

import albumentations as A
import cv2
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

LIBRARY = "albumentationsx"

NUM_CHANNELS = 9  # 3 RGB repetitions stacked; must match --num-channels passed to runner


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return np.ascontiguousarray(transform(image=image)["image"])


TRANSFORMS = [
    # --- Geometric (channel-agnostic) ---
    {"name": "HorizontalFlip", "transform": A.HorizontalFlip(p=1)},
    {"name": "VerticalFlip", "transform": A.VerticalFlip(p=1)},
    {
        "name": "Rotate",
        "transform": A.Rotate(limit=45, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1),
    },
    {
        "name": "Affine",
        "transform": A.Affine(
            rotate=25.0,
            translate_px=(20, 20),
            scale=2.0,
            shear=(10.0, 15.0),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1,
        ),
    },
    {
        "name": "Perspective",
        "transform": A.Perspective(scale=(0.05, 0.1), interpolation=cv2.INTER_LINEAR, fill=0, p=1),
    },
    {
        "name": "Shear",
        "transform": A.Affine(shear=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1),
    },
    {
        "name": "ThinPlateSpline",
        "transform": A.ThinPlateSpline(num_control_points=2, scale_range=(0.5, 0.5), p=1),
    },
    {
        "name": "OpticalDistortion",
        "transform": A.OpticalDistortion(distort_limit=0.5, p=1),
    },
    {
        "name": "ElasticTransform",
        "transform": A.ElasticTransform(alpha=50.0, sigma=5.0, interpolation=cv2.INTER_LINEAR, same_dxdy=True, p=1),
    },
    # --- Spatial crop / resize ---
    {"name": "RandomCrop128", "transform": A.RandomCrop(height=128, width=128, pad_if_needed=True, p=1)},
    {"name": "CenterCrop128", "transform": A.CenterCrop(height=128, width=128, pad_if_needed=True, p=1)},
    {
        "name": "RandomResizedCrop",
        "transform": A.RandomResizedCrop(
            size=(512, 512),
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=cv2.INTER_LINEAR,
            p=1,
        ),
    },
    {"name": "Resize", "transform": A.Resize(height=512, width=512, interpolation=cv2.INTER_LINEAR, p=1)},
    {"name": "LongestMaxSize", "transform": A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_LINEAR, p=1)},
    {"name": "SmallestMaxSize", "transform": A.SmallestMaxSize(max_size=512, interpolation=cv2.INTER_LINEAR, p=1)},
    {"name": "Pad", "transform": A.Pad(padding=10, fill=0, border_mode=cv2.BORDER_CONSTANT, p=1)},
    # --- Pixel-level: pure arithmetic (all channels independently, no OpenCV 4ch limit) ---
    {"name": "Invert", "transform": A.InvertImg(p=1)},
    {"name": "Posterize", "transform": A.Posterize(num_bits=4, p=1)},
    {"name": "Solarize", "transform": A.Solarize(threshold_range=(0.5, 0.5), p=1)},
    {"name": "RandomGamma", "transform": A.RandomGamma(gamma_limit=(120, 120), p=1)},
    {
        "name": "GaussianNoise",
        "transform": A.GaussNoise(std_range=(0.44, 0.44), mean_range=(0, 0), per_channel=True, p=1),
    },
    {"name": "SaltAndPepper", "transform": A.SaltAndPepper(amount=(0.01, 0.06), salt_vs_pepper=(0.4, 0.6), p=1)},
    {"name": "Erasing", "transform": A.Erasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), p=1)},
    # --- Blur (OpenCV-backed, requires chunking for > 4 channels) ---
    {"name": "Blur", "transform": A.Blur(blur_limit=(5, 5), p=1)},
    {"name": "GaussianBlur", "transform": A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(2.0, 2.0), p=1)},
    {"name": "MedianBlur", "transform": A.MedianBlur(blur_limit=(5, 5), p=1)},
    {"name": "MotionBlur", "transform": A.MotionBlur(blur_limit=5, angle_range=(0, 360), direction_range=(-1, 1), p=1)},
    {"name": "Sharpen", "transform": A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=1)},
    # --- Histogram / contrast (OpenCV-backed; CLAHE/Equalize require 1ch or 3ch — excluded) ---
    {"name": "AutoContrast", "transform": A.AutoContrast(p=1, method="pil")},
    # --- Brightness / contrast (per-channel arithmetic) ---
    {
        "name": "Brightness",
        "transform": A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0.0, 0.0), p=1),
    },
    {
        "name": "Contrast",
        "transform": A.RandomBrightnessContrast(brightness_limit=(0.0, 0.0), contrast_limit=(0.2, 0.2), p=1),
    },
    # --- Normalize (9-channel mean/std) ---
    {
        "name": "Normalize",
        "transform": A.Normalize(
            mean=(0.485, 0.456, 0.406) * 3,  # repeat per stacked RGB triplet
            std=(0.229, 0.224, 0.225) * 3,
            p=1,
        ),
    },
    # --- Compression (operates on first 3 channels; reveals truncation behavior) ---
    {"name": "JpegCompression", "transform": A.ImageCompression(quality_range=(50, 50), p=1)},
    # --- Channel manipulation ---
    {"name": "ChannelShuffle", "transform": A.ChannelShuffle(p=1)},
    {"name": "ChannelDropout", "transform": A.ChannelDropout(p=1)},
    # --- Grayscale: use method="average" (works with any channel count; weighted_average/from_lab require 3ch) ---
    {"name": "Grayscale", "transform": A.ToGray(num_output_channels=NUM_CHANNELS, method="average", p=1)},
    # --- Illumination ---
    {"name": "LinearIllumination", "transform": A.Illumination(p=1, mode="linear", angle_range=(90, 90))},
    {"name": "CornerIllumination", "transform": A.Illumination(p=1, mode="corner")},
    {"name": "GaussianIllumination", "transform": A.Illumination(p=1, mode="gaussian")},
    {
        "name": "PlasmaBrightness",
        "transform": A.PlasmaBrightnessContrast(p=1, roughness=0.5, contrast_range=(0.0, 0.0)),
    },
    {
        "name": "PlasmaContrast",
        "transform": A.PlasmaBrightnessContrast(p=1, roughness=0.5, brightness_range=(0.0, 0.0)),
    },
    {"name": "PlasmaShadow", "transform": A.PlasmaShadow(p=1, roughness=0.5)},
]
