from typing import Any

import albumentations as A
import cv2
import numpy as np

# Ensure single thread
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class AlbumentationsVideoImpl:
    """Albumentations implementations of transforms for videos"""

    @staticmethod
    def Resize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Resize(
            height=params["target_size"],
            width=params["target_size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )

    @staticmethod
    def RandomCrop128(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomCrop(
            height=params["height"],
            width=params["width"],
            pad_if_needed=True,
            p=1,
        )

    @staticmethod
    def RandomResizedCrop(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomResizedCrop(
            size=params["size"],
            scale=params["scale"],
            ratio=params["ratio"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )

    @staticmethod
    def CenterCrop128(params: dict[str, Any]) -> A.BasicTransform:
        return A.CenterCrop(
            height=params["height"],
            width=params["width"],
            pad_if_needed=True,
            p=1,
        )

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> A.BasicTransform:
        return A.HorizontalFlip(p=1)

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> A.BasicTransform:
        return A.VerticalFlip(p=1)

    @staticmethod
    def Pad(params: dict[str, Any]) -> A.BasicTransform:
        return A.Pad(
            padding=params["padding"],
            fill=params["fill"],
            border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
            p=1,
        )

    @staticmethod
    def Rotate(params: dict[str, Any]) -> A.BasicTransform:
        return A.Rotate(
            limit=params["angle"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT if params["mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )

    @staticmethod
    def Affine(params: dict[str, Any]) -> A.BasicTransform:
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

    @staticmethod
    def Perspective(params: dict[str, Any]) -> A.BasicTransform:
        return A.Perspective(
            scale=params["scale"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )

    @staticmethod
    def Elastic(params: dict[str, Any]) -> A.BasicTransform:
        return A.ElasticTransform(
            alpha=params["alpha"],
            sigma=params["sigma"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            approximate=params["approximate"],
            same_dxdy=params["same_dxdy"],
            p=1,
        )

    @staticmethod
    def ColorJitter(params: dict[str, Any]) -> A.BasicTransform:
        return A.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=1,
        )

    @staticmethod
    def ChannelShuffle(params: dict[str, Any]) -> A.BasicTransform:
        return A.ChannelShuffle(p=1)

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> A.BasicTransform:
        return A.ToGray(num_output_channels=params["num_output_channels"], p=1)

    @staticmethod
    def RGBShift(params: dict[str, Any]) -> A.BasicTransform:
        shift = params["pixel_shift"]
        return A.RGBShift(
            r_shift_limit=shift,
            g_shift_limit=shift,
            b_shift_limit=shift,
            p=1,
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> A.BasicTransform:
        return A.GaussianBlur(
            blur_limit=params["kernel_size"],  # assuming square kernel
            sigma_limit=(params["sigma"], params["sigma"]),
            p=1,
        )

    @staticmethod
    def GaussianNoise(params: dict[str, Any]) -> A.BasicTransform:
        return A.GaussNoise(
            std_range=(params["std"], params["std"]),
            mean_range=(params["mean"], params["mean"]),
            per_channel=params["per_channel"],
            p=1,
        )

    @staticmethod
    def Invert(params: dict[str, Any]) -> A.BasicTransform:
        return A.InvertImg(p=1)

    @staticmethod
    def Posterize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Posterize(
            num_bits=params["bits"],
            p=1,
        )

    @staticmethod
    def Solarize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Solarize(
            threshold_range=(params["threshold"], params["threshold"]),
            p=1,
        )

    @staticmethod
    def Sharpen(params: dict[str, Any]) -> A.BasicTransform:
        return A.Sharpen(
            alpha=params["alpha"],
            lightness=params["lightness"],
            p=1,
        )

    @staticmethod
    def AutoContrast(params: dict[str, Any]) -> A.BasicTransform:
        return A.AutoContrast(p=1, method="pil")

    @staticmethod
    def Equalize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Equalize(p=1)

    @staticmethod
    def Normalize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Normalize(
            mean=params["mean"],
            std=params["std"],
            p=1,
        )

    @staticmethod
    def Erasing(params: dict[str, Any]) -> A.BasicTransform:
        return A.Erasing(
            scale=params["scale"],
            ratio=params["ratio"],
            p=1,
        )

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> A.BasicTransform:
        return A.ImageCompression(
            quality_range=(params["quality"], params["quality"]),
            p=1,
        )

    @staticmethod
    def RandomGamma(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomGamma(
            gamma_limit=(params["gamma"], params["gamma"]),
            p=1,
        )

    @staticmethod
    def PlankianJitter(params: dict[str, Any]) -> A.BasicTransform:
        return A.PlanckianJitter(
            mode=params["mode"],
            p=1,
        )

    @staticmethod
    def MedianBlur(params: dict[str, Any]) -> A.BasicTransform:
        return A.MedianBlur(
            blur_limit=(params["blur_limit"], params["blur_limit"]),
            p=1,
        )

    @staticmethod
    def MotionBlur(params: dict[str, Any]) -> A.BasicTransform:
        return A.MotionBlur(
            blur_limit=params["kernel_size"],
            angle_range=params["angle_range"],
            direction_range=params["direction_range"],
            p=1,
        )

    @staticmethod
    def CLAHE(params: dict[str, Any]) -> A.BasicTransform:
        return A.CLAHE(
            clip_limit=params["clip_limit"],
            tile_grid_size=params["tile_grid_size"],
            p=1,
        )

    @staticmethod
    def Brightness(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomBrightnessContrast(
            brightness_limit=params["brightness_limit"],
            contrast_limit=(0.0, 0.0),
            p=1,
        )

    @staticmethod
    def Contrast(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0),
            contrast_limit=params["contrast_limit"],
            p=1,
        )

    @staticmethod
    def CoarseDropout(params: dict[str, Any]) -> A.BasicTransform:
        return A.CoarseDropout(
            hole_height_range=params["hole_height_range"],
            hole_width_range=params["hole_width_range"],
            num_holes_range=params["num_holes_range"],
            p=1,
        )

    @staticmethod
    def Blur(params: dict[str, Any]) -> A.BasicTransform:
        return A.Blur(
            blur_limit=(params["radius"], params["radius"]),
            p=1,
        )

    @staticmethod
    def HSV(params: dict[str, Any]) -> A.BasicTransform:
        return A.HueSaturationValue(
            hue_shift_limit=params["hue"] * 255,
            sat_shift_limit=params["saturation"] * 255,
            val_shift_limit=params["value"] * 255,
            p=1,
        )

    @staticmethod
    def ChannelDropout(params: dict[str, Any]) -> A.BasicTransform:
        return A.ChannelDropout(p=1)

    @staticmethod
    def LinearIllumination(params: dict[str, Any]) -> A.BasicTransform:
        return A.Illumination(p=1, mode="linear", angle_range=(90, 90))

    @staticmethod
    def CornerIllumination(params: dict[str, Any]) -> A.BasicTransform:
        return A.Illumination(p=1, mode="corner")

    @staticmethod
    def GaussianIllumination(params: dict[str, Any]) -> A.BasicTransform:
        return A.Illumination(p=1, mode="gaussian")

    @staticmethod
    def Hue(params: dict[str, Any]) -> A.BasicTransform:
        return A.HueSaturationValue(
            hue_shift_limit=params["hue"],
            sat_shift_limit=0,
            val_shift_limit=0,
            p=1,
        )

    @staticmethod
    def PlasmaBrightness(params: dict[str, Any]) -> A.BasicTransform:
        return A.PlasmaBrightnessContrast(p=1, roughness=params["roughness"], contrast_range=(0.0, 0.0))

    @staticmethod
    def PlasmaContrast(params: dict[str, Any]) -> A.BasicTransform:
        return A.PlasmaBrightnessContrast(p=1, roughness=params["roughness"], brightness_range=(0.0, 0.0))

    @staticmethod
    def PlasmaShadow(params: dict[str, Any]) -> A.BasicTransform:
        return A.PlasmaShadow(p=1, roughness=params["roughness"])

    @staticmethod
    def Rain(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomRain(
            p=1,
            drop_width=params["drop_width"],
            brightness_coefficient=params["brightness_coefficient"],
        )

    @staticmethod
    def SaltAndPepper(params: dict[str, Any]) -> A.BasicTransform:
        return A.SaltAndPepper(p=1, amount=params["amount"], salt_vs_pepper=params["salt_vs_pepper"])

    @staticmethod
    def Saturation(params: dict[str, Any]) -> A.BasicTransform:
        sat_shift_limit = params["saturation_factor"] * 255
        return A.HueSaturationValue(p=1, hue_shift_limit=0, sat_shift_limit=sat_shift_limit, val_shift_limit=0)

    @staticmethod
    def Snow(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomSnow(p=1, snow_point_range=params["snow_point_range"])

    @staticmethod
    def OpticalDistortion(params: dict[str, Any]) -> A.BasicTransform:
        return A.OpticalDistortion(p=1, distort_limit=params["distort_limit"])

    @staticmethod
    def Shear(params: dict[str, Any]) -> A.BasicTransform:
        return A.Affine(
            p=1,
            shear=params["shear"],
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        )

    @staticmethod
    def ThinPlateSpline(params: dict[str, Any]) -> A.BasicTransform:
        return A.ThinPlateSpline(
            p=1,
            num_control_points=params["num_control_points"],
            scale_range=(params["distortion_scale"], params["distortion_scale"]),
        )

    @staticmethod
    def __call__(transform: A.BasicTransform, video: np.ndarray) -> np.ndarray:
        """Apply transform to video using the 'images' parameter directly"""
        # Use the transform directly with p=1 to ensure it's applied
        return np.ascontiguousarray(transform(images=video)["images"])
