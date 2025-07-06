"""AlbumentationsX implementations of transforms for videos in custom format."""

from typing import Any

import albumentations as A
import cv2
import numpy as np

from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec

# Ensure single thread
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Required: Library name for dependency installation
LIBRARY = "albumentationsx"


# Required: Define how to apply transforms to videos
def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    """Apply AlbumentationsX transform to video frames

    Args:
        transform: AlbumentationsX transform instance
        video: numpy array of shape (T, H, W, C)

    Returns:
        Transformed video as numpy array
    """
    # AlbumentationsX expects 'images' parameter for video frames
    result = transform(images=video)["images"]
    # Ensure contiguous memory layout for performance
    return np.ascontiguousarray(result)


# Helper function to create transforms from specs
def create_transform(spec: TransformSpec) -> Any:
    """Create an AlbumentationsX transform from a TransformSpec."""
    params = spec.params

    if spec.name == "Resize":
        return A.Resize(
            height=params["target_size"],
            width=params["target_size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "RandomCrop128":
        return A.RandomCrop(
            height=params["height"],
            width=params["width"],
            pad_if_needed=True,
            p=1,
        )
    if spec.name == "RandomResizedCrop":
        return A.RandomResizedCrop(
            size=params["size"],
            scale=params["scale"],
            ratio=params["ratio"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "CenterCrop128":
        return A.CenterCrop(
            height=params["height"],
            width=params["width"],
            pad_if_needed=True,
            p=1,
        )
    if spec.name == "HorizontalFlip":
        return A.HorizontalFlip(p=1)
    if spec.name == "VerticalFlip":
        return A.VerticalFlip(p=1)
    if spec.name == "Pad":
        return A.Pad(
            padding=params["padding"],
            fill=params["fill"],
            border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
            p=1,
        )
    if spec.name == "Rotate":
        return A.Rotate(
            limit=params["angle"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT if params["mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "Affine":
        return A.Affine(
            rotate=params["angle"],
            translate_px=params["shift"],
            scale=params["scale"],
            shear=params["shear"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT if params["mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "Perspective":
        return A.Perspective(
            scale=params["scale"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "Elastic":
        return A.ElasticTransform(
            alpha=params["alpha"],
            sigma=params["sigma"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            approximate=params["approximate"],
            same_dxdy=params["same_dxdy"],
            p=1,
        )
    if spec.name == "ColorJitter":
        return A.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=1,
        )
    if spec.name == "ChannelShuffle":
        return A.ChannelShuffle(p=1)
    if spec.name == "Grayscale":
        return A.ToGray(num_output_channels=params["num_output_channels"], p=1)
    if spec.name == "RGBShift":
        shift = params["pixel_shift"]
        return A.RGBShift(
            r_shift_limit=shift,
            g_shift_limit=shift,
            b_shift_limit=shift,
            p=1,
        )
    if spec.name == "GaussianBlur":
        return A.GaussianBlur(
            blur_limit=params["kernel_size"],
            sigma_limit=(params["sigma"], params["sigma"]),
            p=1,
        )
    if spec.name == "GaussianNoise":
        return A.GaussNoise(
            std_range=(params["std"], params["std"]),
            mean_range=(params["mean"], params["mean"]),
            per_channel=params["per_channel"],
            p=1,
        )
    if spec.name == "Invert":
        return A.InvertImg(p=1)
    if spec.name == "Posterize":
        return A.Posterize(
            num_bits=params["bits"],
            p=1,
        )
    if spec.name == "Solarize":
        return A.Solarize(
            threshold_range=(params["threshold"], params["threshold"]),
            p=1,
        )
    if spec.name == "Sharpen":
        return A.Sharpen(
            alpha=params["alpha"],
            lightness=params["lightness"],
            p=1,
        )
    if spec.name == "AutoContrast":
        return A.AutoContrast(p=1, method="pil")
    if spec.name == "Equalize":
        return A.Equalize(p=1)
    if spec.name == "Normalize":
        return A.Normalize(
            mean=params["mean"],
            std=params["std"],
            p=1,
        )
    if spec.name == "Erasing":
        return A.Erasing(
            scale=params["scale"],
            ratio=params["ratio"],
            p=1,
        )
    if spec.name == "JpegCompression":
        return A.ImageCompression(
            quality_range=(params["quality"], params["quality"]),
            p=1,
        )
    if spec.name == "RandomGamma":
        return A.RandomGamma(
            gamma_limit=(params["gamma"], params["gamma"]),
            p=1,
        )
    if spec.name == "PlankianJitter":
        return A.PlanckianJitter(
            mode=params["mode"],
            p=1,
        )
    if spec.name == "MedianBlur":
        return A.MedianBlur(
            blur_limit=(params["blur_limit"], params["blur_limit"]),
            p=1,
        )
    if spec.name == "MotionBlur":
        return A.MotionBlur(
            blur_limit=params["kernel_size"],
            angle_range=params["angle_range"],
            direction_range=params["direction_range"],
            p=1,
        )
    if spec.name == "CLAHE":
        return A.CLAHE(
            clip_limit=params["clip_limit"],
            tile_grid_size=params["tile_grid_size"],
            p=1,
        )
    if spec.name == "Brightness":
        return A.RandomBrightnessContrast(
            brightness_limit=params["brightness_limit"],
            contrast_limit=(0.0, 0.0),
            p=1,
        )
    if spec.name == "Contrast":
        return A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0),
            contrast_limit=params["contrast_limit"],
            p=1,
        )
    if spec.name == "CoarseDropout":
        return A.CoarseDropout(
            hole_height_range=params["hole_height_range"],
            hole_width_range=params["hole_width_range"],
            num_holes_range=params["num_holes_range"],
            p=1,
        )
    if spec.name == "Blur":
        return A.Blur(
            blur_limit=(params["radius"], params["radius"]),
            p=1,
        )
    if spec.name == "HSV":
        return A.HueSaturationValue(
            hue_shift_limit=params["hue"] * 255,
            sat_shift_limit=params["saturation"] * 255,
            val_shift_limit=params["value"] * 255,
            p=1,
        )
    if spec.name == "ChannelDropout":
        return A.ChannelDropout(p=1)
    if spec.name == "LinearIllumination":
        return A.Illumination(p=1, mode="linear", angle_range=(90, 90))
    if spec.name == "CornerIllumination":
        return A.Illumination(p=1, mode="corner")
    if spec.name == "GaussianIllumination":
        return A.Illumination(p=1, mode="gaussian")
    if spec.name == "Hue":
        return A.HueSaturationValue(
            hue_shift_limit=params["hue"],
            sat_shift_limit=0,
            val_shift_limit=0,
            p=1,
        )
    if spec.name == "PlasmaBrightness":
        return A.PlasmaBrightnessContrast(p=1, roughness=params["roughness"], contrast_range=(0.0, 0.0))
    if spec.name == "PlasmaContrast":
        return A.PlasmaBrightnessContrast(p=1, roughness=params["roughness"], brightness_range=(0.0, 0.0))
    if spec.name == "PlasmaShadow":
        return A.PlasmaShadow(p=1, roughness=params["roughness"])
    if spec.name == "Rain":
        return A.RandomRain(
            p=1,
            drop_width=params["drop_width"],
            brightness_coefficient=params["brightness_coefficient"],
        )
    if spec.name == "SaltAndPepper":
        return A.SaltAndPepper(p=1, amount=params["amount"], salt_vs_pepper=params["salt_vs_pepper"])
    if spec.name == "Saturation":
        sat_shift_limit = params["saturation_factor"] * 255
        return A.HueSaturationValue(p=1, hue_shift_limit=0, sat_shift_limit=sat_shift_limit, val_shift_limit=0)
    if spec.name == "Snow":
        return A.RandomSnow(p=1, snow_point_range=params["snow_point_range"])
    if spec.name == "OpticalDistortion":
        return A.OpticalDistortion(p=1, distort_limit=params["distort_limit"])
    if spec.name == "Shear":
        return A.Affine(
            p=1,
            shear=params["shear"],
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        )
    if spec.name == "ThinPlateSpline":
        return A.ThinPlateSpline(
            p=1,
            num_control_points=params["num_control_points"],
            scale_range=(params["distortion_scale"], params["distortion_scale"]),
        )
    raise ValueError(f"Unknown transform: {spec.name}")


# Required: Transform definitions from specs
TRANSFORMS = [
    {
        "name": spec.name,
        "transform": create_transform(spec),
    }
    for spec in TRANSFORM_SPECS
]
