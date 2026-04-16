r"""Albumentations (MIT) multi-channel image benchmark spec.

Tests transforms on 9-channel images (3x stacked RGB, simulating e.g. hyperspectral data).
The runner synthesizes these in-memory via make_multichannel_loader; no special dataset needed.

Run with:
    python -m benchmark.cli run \\
        --spec benchmark/transforms/albumentations_mit_multichannel_impl.py \\
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

from benchmark.transforms.albumentations_mit_compat import (
    ConstrainedCoarseDropoutWrapper as _ConstrainedCoarseDropoutWrapper,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

LIBRARY = "albumentations_mit"

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
    # --- Additional geometric (channel-agnostic) ---
    {"name": "SquareSymmetry", "transform": A.SquareSymmetry(p=1)},
    {"name": "Transpose", "transform": A.Transpose(p=1)},
    {
        "name": "SafeRotate",
        "transform": A.SafeRotate(
            limit=45,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1,
        ),
    },
    {"name": "RandomRotate90", "transform": A.RandomRotate90(p=1)},
    {"name": "RandomScale", "transform": A.RandomScale(scale_limit=(-0.1, 0.1), interpolation=cv2.INTER_LINEAR, p=1)},
    {
        "name": "ShiftScaleRotate",
        "transform": A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1,
        ),
    },
    {
        "name": "GridDistortion",
        "transform": A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1,
        ),
    },
    {"name": "PiecewiseAffine", "transform": A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=1)},
    {"name": "RandomGridShuffle", "transform": A.RandomGridShuffle(grid=(3, 3), p=1)},
    {"name": "Morphological", "transform": A.Morphological(scale=(2, 3), operation="dilation", p=1)},
    # --- Additional spatial crop / pad ---
    {
        "name": "PadIfNeeded",
        "transform": A.PadIfNeeded(
            min_height=1024,
            min_width=1024,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1,
        ),
    },
    {
        "name": "CropAndPad",
        "transform": A.CropAndPad(
            px=(-10, 20, -10, 20),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1,
        ),
    },
    {
        "name": "RandomSizedCrop",
        "transform": A.RandomSizedCrop(
            min_max_height=(256, 480),
            size=(512, 512),
            interpolation=cv2.INTER_LINEAR,
            p=1,
        ),
    },
    # --- Additional blur ---
    {
        "name": "AdvancedBlur",
        "transform": A.AdvancedBlur(
            blur_limit=(3, 7),
            sigma_x_limit=(0.2, 1.0),
            sigma_y_limit=(0.2, 1.0),
            p=1,
        ),
    },
    {"name": "Defocus", "transform": A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=1)},
    {
        "name": "UnsharpMask",
        "transform": A.UnsharpMask(
            blur_limit=(3, 7),
            sigma_limit=0.0,
            alpha=(0.2, 0.5),
            threshold=10,
            p=1,
        ),
    },
    # --- Additional pixel-level (channel-agnostic arithmetic) ---
    {"name": "Emboss", "transform": A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1)},
    {"name": "MultiplicativeNoise", "transform": A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1)},
    {
        "name": "AdditiveNoise",
        "transform": A.AdditiveNoise(
            noise_type="uniform",
            spatial_mode="per_pixel",
            noise_params={"ranges": [(-0.1, 0.1)]},
            p=1,
        ),
    },
    {"name": "Downscale", "transform": A.Downscale(scale_range=(0.25, 0.25), p=1)},
    {"name": "RingingOvershoot", "transform": A.RingingOvershoot(blur_limit=(7, 15), cutoff=(0.7854, 1.5708), p=1)},
    # --- Additional dropout ---
    {"name": "GridDropout", "transform": A.GridDropout(ratio=0.5, unit_size_range=(2, 10), random_offset=False, p=1)},
    {"name": "PixelDropout", "transform": A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, p=1)},
    {
        "name": "ConstrainedCoarseDropout",
        "transform": _ConstrainedCoarseDropoutWrapper(
            A.ConstrainedCoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.1, 0.2),
                hole_width_range=(0.1, 0.2),
                p=1,
            ),
        ),
    },
    {
        "name": "CoarseDropout",
        "transform": A.CoarseDropout(
            hole_height_range=(0.1, 0.1),
            hole_width_range=(0.1, 0.1),
            num_holes_range=(4, 4),
            p=1,
        ),
    },
]
