"""Albumentations (MIT) implementations of transforms for videos in custom format."""

from typing import Any

import albumentations as A
import cv2
import numpy as np

from benchmark.transforms.registry import build_transforms, register_library
from benchmark.transforms.specs import TransformSpec

# Ensure single thread
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Required: Library name for dependency installation
LIBRARY = "albumentations_mit"


# Required: Define how to apply transforms to videos
def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    """Apply Albumentations (MIT) transform to video frames

    Args:
        transform: Albumentations transform instance
        video: numpy array of shape (T, H, W, C)

    Returns:
        Transformed video as numpy array
    """
    # albucore's batch_transform reshapes (T,H,W,C) → (H,W,T*C) for spatial transforms,
    # then calls cv2.warpAffine on the merged array. OpenCV fails when T*C > ~512 channels.
    # Apply frame-by-frame to avoid the XHWC reshape path entirely.
    frames = [transform(image=frame)["image"] for frame in video]
    return np.ascontiguousarray(frames)


# Helper function to create transforms from specs
def create_transform(spec: TransformSpec) -> Any:
    """Create an Albumentations (MIT) transform from a TransformSpec."""
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
    if spec.name == "PhotoMetricDistort":
        return A.ColorJitter(
            brightness=params["brightness_range"],
            contrast=params["contrast_range"],
            saturation=params["saturation_range"],
            hue=params["hue_range"],
            p=1,
        )
    if spec.name == "LongestMaxSize":
        return A.LongestMaxSize(
            max_size=params["max_size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "SmallestMaxSize":
        return A.SmallestMaxSize(
            max_size=params["max_size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    # --- Additional shared transforms ---
    if spec.name == "AdvancedBlur":
        return A.AdvancedBlur(
            blur_limit=params["blur_limit"],
            sigma_x_limit=params["sigma_x_limit"],
            sigma_y_limit=params["sigma_y_limit"],
            p=1,
        )
    if spec.name == "Defocus":
        return A.Defocus(radius=params["radius"], alias_blur=params["alias_blur"], p=1)
    if spec.name == "ZoomBlur":
        return A.ZoomBlur(max_factor=params["max_factor"], p=1)
    if spec.name == "GlassBlur":
        return A.GlassBlur(
            sigma=params["sigma"],
            max_delta=params["max_delta"],
            iterations=params["iterations"],
            p=1,
        )
    if spec.name == "SquareSymmetry":
        return A.SquareSymmetry(p=1)
    if spec.name == "Transpose":
        return A.Transpose(p=1)
    if spec.name == "SafeRotate":
        return A.SafeRotate(
            limit=params["limit"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "RandomRotate90":
        return A.RandomRotate90(p=1)
    if spec.name == "RandomScale":
        return A.RandomScale(
            scale_limit=params["scale_limit"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "ShiftScaleRotate":
        return A.ShiftScaleRotate(
            shift_limit=params["shift_limit"],
            scale_limit=params["scale_limit"],
            rotate_limit=params["rotate_limit"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "GridDistortion":
        return A.GridDistortion(
            num_steps=params["num_steps"],
            distort_limit=params["distort_limit"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "PiecewiseAffine":
        return A.PiecewiseAffine(
            scale=params["scale"],
            nb_rows=params["nb_rows"],
            nb_cols=params["nb_cols"],
            p=1,
        )
    if spec.name == "RandomGridShuffle":
        return A.RandomGridShuffle(grid=params["grid"], p=1)
    if spec.name == "Morphological":
        return A.Morphological(scale=params["scale"], operation=params["operation"], p=1)
    if spec.name == "Downscale":
        return A.Downscale(
            scale_range=params["scale_range"],
            interpolation_pair={
                "downscale": cv2.INTER_NEAREST if params["interpolation_pair"][0] == "nearest" else cv2.INTER_LINEAR,
                "upscale": cv2.INTER_NEAREST if params["interpolation_pair"][1] == "nearest" else cv2.INTER_LINEAR,
            },
            p=1,
        )
    if spec.name == "Emboss":
        return A.Emboss(alpha=params["alpha"], strength=params["strength"], p=1)
    if spec.name == "ChromaticAberration":
        return A.ChromaticAberration(
            primary_distortion_limit=params["primary_distortion_limit"],
            secondary_distortion_limit=params["secondary_distortion_limit"],
            mode=params["mode"],
            p=1,
        )
    if spec.name == "ISONoise":
        return A.ISONoise(color_shift=params["color_shift"], intensity=params["intensity"], p=1)
    if spec.name == "ShotNoise":
        return A.ShotNoise(scale_range=params["scale_range"], p=1)
    if spec.name == "MultiplicativeNoise":
        return A.MultiplicativeNoise(
            multiplier=params["multiplier"],
            per_channel=params["per_channel"],
            p=1,
        )
    if spec.name == "AdditiveNoise":
        lo, hi = params["scale_range"]
        mag = max(abs(lo), abs(hi))
        return A.AdditiveNoise(
            noise_type=params["noise_type"],
            spatial_mode=params["spatial_mode"],
            noise_params={"ranges": [(-mag, mag)]},
            p=1,
        )
    if spec.name == "RandomFog":
        return A.RandomFog(
            fog_coef_range=params["fog_coef_range"],
            alpha_coef=params["alpha_coef"],
            p=1,
        )
    if spec.name == "RandomShadow":
        return A.RandomShadow(
            num_shadows_limit=params["num_shadows_limit"],
            shadow_dimension=params["shadow_dimension"],
            p=1,
        )
    if spec.name == "RandomSunFlare":
        return A.RandomSunFlare(
            flare_roi=params["flare_roi"],
            num_flare_circles_range=params["num_flare_circles_range"],
            p=1,
        )
    if spec.name == "RandomToneCurve":
        return A.RandomToneCurve(scale=params["scale"], p=1)
    if spec.name == "RingingOvershoot":
        return A.RingingOvershoot(blur_limit=params["blur_limit"], cutoff=params["cutoff"], p=1)
    if spec.name == "Spatter":
        return A.Spatter(
            mean=params["mean"],
            std=params["std"],
            gauss_sigma=params["gauss_sigma"],
            intensity=params["intensity"],
            cutout_threshold=params["cutout_threshold"],
            mode=params["mode"],
            p=1,
        )
    if spec.name == "UnsharpMask":
        return A.UnsharpMask(
            blur_limit=params["blur_limit"],
            sigma_limit=params["sigma_limit"],
            alpha=params["alpha"],
            threshold=params["threshold"],
            p=1,
        )
    if spec.name == "FancyPCA":
        return A.FancyPCA(alpha=params["alpha"], p=1)
    if spec.name == "Superpixels":
        return A.Superpixels(
            p_replace=params["p_replace"],
            n_segments=params["n_segments"],
            p=1,
        )
    if spec.name == "ToSepia":
        return A.ToSepia(p=1)
    if spec.name == "RandomGravel":
        return A.RandomGravel(
            gravel_roi=params["gravel_roi"],
            number_of_patches=params["number_of_patches"],
            p=1,
        )
    if spec.name == "GridDropout":
        return A.GridDropout(
            ratio=params["ratio"],
            unit_size_range=params["unit_size_range"],
            holes_number_xy=params["holes_number_xy"],
            random_offset=params["random_offset"],
            p=1,
        )
    if spec.name == "PixelDropout":
        return A.PixelDropout(
            dropout_prob=params["dropout_prob"],
            per_channel=params["per_channel"],
            drop_value=params["drop_value"],
            p=1,
        )
    if spec.name == "ConstrainedCoarseDropout":
        return A.ConstrainedCoarseDropout(
            num_holes_range=params["num_holes_range"],
            hole_height_range=params["hole_height_range"],
            hole_width_range=params["hole_width_range"],
            p=1,
        )
    if spec.name == "PadIfNeeded":
        return A.PadIfNeeded(
            min_height=params["min_height"],
            min_width=params["min_width"],
            border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "CropAndPad":
        return A.CropAndPad(
            px=params["px"],
            border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "RandomSizedCrop":
        return A.RandomSizedCrop(
            min_max_height=params["min_max_height"],
            size=params["size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    raise ValueError(f"Unknown transform: {spec.name}")


# Register with the central registry
register_library(LIBRARY, create_video_fn=create_transform)

# Required: Transform definitions from specs
TRANSFORMS = build_transforms(LIBRARY, media="video")
