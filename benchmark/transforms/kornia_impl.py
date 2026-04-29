"""Kornia implementations of transforms for images in custom format.

Param conversions (additive->multiplicative, degrees->fraction, etc.) follow rules
documented in benchmark.transforms.specs module docstring.
"""

from typing import Any

import kornia
import kornia.augmentation as Kaug
import torch
import torch.nn.functional as F

from benchmark.transforms.registry import build_transforms, register_library
from benchmark.transforms.specs import TransformSpec

# Required: Library name for dependency installation
LIBRARY = "kornia"

# Force single thread for fair comparison with albumentationsx and torchvision
torch.set_num_threads(1)


class _CenterCropWithPad(torch.nn.Module):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.height, self.width = size
        self.crop = Kaug.CenterCrop(size=size, p=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_height = max(0, self.height - height)
        pad_width = max(0, self.width - width)
        if pad_height or pad_width:
            top = pad_height // 2
            bottom = pad_height - top
            left = pad_width // 2
            right = pad_width - left
            image = F.pad(image, (left, right, top, bottom))
        return self.crop(image)


class _RandomJigsawWithPad(torch.nn.Module):
    def __init__(self, grid: tuple[int, int]) -> None:
        super().__init__()
        self.grid = grid
        self.jigsaw = Kaug.RandomJigsaw(grid=grid, p=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        grid_h, grid_w = self.grid
        pad_height = (-height) % grid_h
        pad_width = (-width) % grid_w
        if pad_height or pad_width:
            image = F.pad(image, (0, pad_width, 0, pad_height))
        return self.jigsaw(image)[..., :height, :width]


# Required: Define how to apply transforms to images
def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    """Apply kornia transform to a single image

    Args:
        transform: Kornia augmentation instance
        image: torch.Tensor of shape (C, H, W)

    Returns:
        Transformed image as torch.Tensor
    """
    return transform(image.unsqueeze(0)).squeeze(0)


# Helper function to create transforms from specs
def create_transform(spec: TransformSpec) -> Any | None:
    """Create a Kornia transform from a TransformSpec."""
    params = spec.params

    if spec.name == "ColorJitter":
        return Kaug.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=1,
            same_on_batch=False,
        )
    if spec.name == "ColorJiggle":
        return Kaug.ColorJiggle(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=1,
            same_on_batch=False,
        )
    if spec.name == "AutoContrast":
        return Kaug.RandomAutoContrast(p=1)
    if spec.name == "Blur":
        return Kaug.RandomBoxBlur(
            p=1,
            kernel_size=(params["radius"], params["radius"]),
            border_type=params["border_mode"],
        )
    if spec.name == "Brightness":
        # Kornia brightness is multiplicative (1.0=no change). Spec limit is additive offset,
        # so +0.2 additive -> multiplicative factor (1.2, 1.2).
        limit = params["brightness_limit"]
        offset = float(limit[0] if isinstance(limit, (list, tuple)) else limit)
        return Kaug.RandomBrightness(
            brightness=(1.0 + offset, 1.0 + offset),
            p=1,
        )
    if spec.name == "ChannelDropout":
        return Kaug.RandomChannelDropout(p=1)
    if spec.name == "ChannelShuffle":
        return Kaug.RandomChannelShuffle(p=1)
    if spec.name == "CLAHE":
        # Pass float clip_limit to avoid Kornia's internal Long-dtype sampling bug
        clip = params["clip_limit"]
        clip_float = (float(clip[0]), float(clip[1])) if isinstance(clip, (list, tuple)) else (float(clip), float(clip))
        return Kaug.RandomClahe(
            p=1,
            clip_limit=clip_float,
            grid_size=params["tile_grid_size"],
        )
    if spec.name == "Contrast":
        # Kornia contrast is multiplicative (1.0=no change). Convert additive offset to factor.
        limit = params["contrast_limit"]
        offset = float(limit[0] if isinstance(limit, (list, tuple)) else limit)
        return Kaug.RandomContrast(
            contrast=(1.0 + offset, 1.0 + offset),
            p=1,
        )
    if spec.name == "Equalize":
        return Kaug.RandomEqualize(p=1)
    if spec.name == "RandomGamma":
        gamma = params["gamma"] / 100
        return Kaug.RandomGamma(
            gamma=(gamma, gamma),
            p=1,
        )
    if spec.name == "GaussianBlur":
        return Kaug.RandomGaussianBlur(
            kernel_size=params["kernel_size"],
            sigma=(params["sigma"], params["sigma"]),
            p=1,
        )
    if spec.name == "LinearIllumination":
        return Kaug.RandomLinearIllumination(
            gain=params["gain"],
            p=1,
        )
    if spec.name == "CornerIllumination":
        return Kaug.RandomLinearCornerIllumination(
            gain=params["gain"],
            p=1,
        )
    if spec.name == "GaussianIllumination":
        return Kaug.RandomGaussianIllumination(
            gain=params["gain"],
            p=1,
        )
    if spec.name == "GaussianNoise":
        return Kaug.RandomGaussianNoise(
            mean=params["mean"],
            std=params["std"],
            p=1,
        )
    if spec.name == "Grayscale":
        return Kaug.RandomGrayscale(p=1)
    if spec.name == "Hue":
        # Kornia expects hue in [-0.5, 0.5] (fraction of 360°). Spec is in degrees.
        hue_fraction = params["hue"] / 360.0
        return Kaug.RandomHue(
            hue=(-hue_fraction, hue_fraction),
            p=1,
        )
    if spec.name == "Invert":
        return Kaug.RandomInvert(p=1)
    if spec.name == "JpegCompression":
        return Kaug.RandomJPEG(
            jpeg_quality=params["quality"],
            p=1,
        )
    if spec.name == "MedianBlur":
        kernel_size = params["blur_limit"]
        return Kaug.RandomMedianBlur(
            kernel_size=(kernel_size, kernel_size),
            p=1,
        )
    if spec.name == "MotionBlur":
        return Kaug.RandomMotionBlur(
            kernel_size=params["kernel_size"],
            angle=params["angle_range"],
            direction=params["direction_range"],
            p=1,
        )
    if spec.name == "PlankianJitter":
        return Kaug.RandomPlanckianJitter(
            mode=params["mode"],
            p=1,
        )
    if spec.name == "PlasmaBrightness":
        return Kaug.RandomPlasmaBrightness(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        )
    if spec.name == "PlasmaContrast":
        return Kaug.RandomPlasmaContrast(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        )
    if spec.name == "PlasmaShadow":
        return Kaug.RandomPlasmaShadow(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        )
    if spec.name == "Rain":
        return Kaug.RandomRain(
            drop_width=(params["drop_width"], params["drop_width"]),
            drop_height=(params["drop_height"], params["drop_height"]),
            p=1,
        )
    if spec.name == "RGBShift":
        return Kaug.RandomRGBShift(
            r_shift_limit=params["pixel_shift"] / 255.0,
            g_shift_limit=params["pixel_shift"] / 255.0,
            b_shift_limit=params["pixel_shift"] / 255.0,
            p=1,
        )
    if spec.name == "SaltAndPepper":
        return Kaug.RandomSaltAndPepperNoise(
            amount=params["amount"],
            salt_vs_pepper=params["salt_vs_pepper"],
            p=1,
        )
    if spec.name == "Saturation":
        # Kornia saturation is multiplicative (1.0=no change, >1=more saturated).
        # Spec saturation_factor=0.5 is an additive-style offset -> factor (1.5, 1.5).
        factor = 1.0 + params["saturation_factor"]
        return Kaug.RandomSaturation(
            saturation=(factor, factor),
            p=1,
        )
    if spec.name == "Sharpen":
        # sharpness=2.0 produces noticeable sharpening (default 0.5 barely changes the image)
        return Kaug.RandomSharpness(
            sharpness=2.0,
            p=1,
        )
    if spec.name == "Snow":
        # Pass explicit float brightness to avoid Kornia's internal Long-dtype sampling bug
        return Kaug.RandomSnow(
            snow_coefficient=params["snow_point_range"],
            brightness=(2.0, 2.0),
            p=1,
        )
    if spec.name == "Solarize":
        return Kaug.RandomSolarize(
            thresholds=params["threshold"],
            p=1,
        )
    if spec.name == "CenterCrop128":
        return _CenterCropWithPad(size=(params["height"], params["width"]))
    if spec.name == "Affine":
        angle_degrees = float(params["angle"])
        # Translation in pixels (same as albumentations translate_px)
        tx, ty = float(params["shift"][0]), float(params["shift"][1])
        scale_factor = float(params["scale"])
        shear_value = float(params["shear"]) if isinstance(params["shear"], int | float) else 0.0

        return kornia.geometry.transform.Affine(
            angle=torch.tensor([angle_degrees]),
            translation=torch.tensor([[tx, ty]]),
            scale_factor=torch.tensor([[scale_factor, scale_factor]]),
            shear=torch.tensor([[shear_value, shear_value]]),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    if spec.name == "RandomCrop128":
        return Kaug.RandomCrop(
            size=(params["height"], params["width"]),
            pad_if_needed=True,
            p=1,
        )
    if spec.name == "Elastic":
        return Kaug.RandomElasticTransform(
            p=1,
            sigma=(params["sigma"], params["sigma"]),
            alpha=(params["alpha"], params["alpha"]),
        )
    if spec.name == "Erasing":
        # Pass float value to avoid Kornia's internal Long-dtype sampling bug
        return Kaug.RandomErasing(
            p=1,
            scale=params["scale"],
            ratio=params["ratio"],
            value=float(params["fill"]),
        )
    if spec.name == "OpticalDistortion":
        return kornia.augmentation.RandomFisheye(
            center_x=torch.tensor([-0.3, 0.3]),
            center_y=torch.tensor([-0.3, 0.3]),
            gamma=torch.tensor([0.9, 1.1]),
            p=1,
        )
    if spec.name == "HorizontalFlip":
        return Kaug.RandomHorizontalFlip(p=1)
    if spec.name == "Perspective":
        return Kaug.RandomPerspective(
            distortion_scale=params["scale"][1],
            resample=params["interpolation"],
            p=1,
        )
    if spec.name == "RandomResizedCrop":
        return Kaug.RandomResizedCrop(
            size=params["size"],
            scale=params["scale"],
            ratio=params["ratio"],
            p=1,
        )
    if spec.name == "RandomRotate90":
        return Kaug.RandomRotation90(times=params["times"], p=1)
    if spec.name == "RandomJigsaw":
        return _RandomJigsawWithPad(grid=params["grid"])
    if spec.name == "Rotate":
        return Kaug.RandomRotation(
            degrees=params["angle_range"],
            p=1,
        )
    if spec.name == "Shear":
        return Kaug.RandomShear(
            shear=params["shear"],
            p=1,
        )
    if spec.name == "ThinPlateSpline":
        return Kaug.RandomThinPlateSpline(
            scale=params["distortion_scale"],
            p=1,
        )
    if spec.name == "VerticalFlip":
        return Kaug.RandomVerticalFlip(p=1)
    if spec.name == "Resize":
        return Kaug.Resize(
            size=(params["target_size"], params["target_size"]),
            p=1,
        )
    if spec.name == "Normalize":
        return Kaug.Normalize(
            mean=params["mean"],
            std=params["std"],
            p=1,
        )
    if spec.name == "Posterize":
        return Kaug.RandomPosterize(
            bits=float(params["bits"]),
            p=1,
        )
    if spec.name == "LongestMaxSize":
        return Kaug.LongestMaxSize(
            max_size=params["max_size"],
            resample=params["interpolation"],
            p=1,
        )
    if spec.name == "SmallestMaxSize":
        return Kaug.SmallestMaxSize(
            max_size=params["max_size"],
            resample=params["interpolation"],
            p=1,
        )
    # Skip transforms not supported by kornia
    return None


# Register with the central registry
register_library(LIBRARY, create_image_fn=create_transform)

# Required: Transform definitions from specs
TRANSFORMS = build_transforms(LIBRARY, media="image")
