
from typing import Any

import albumentations as A
import cv2
import numpy as np

# Ensure single thread
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class AlbumentationsImpl:
    """Albumentations implementations of transforms"""

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> A.BasicTransform:
        return A.HorizontalFlip(**params)

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> A.BasicTransform:
        return A.VerticalFlip(**params)

    @staticmethod
    def Rotate(params: dict[str, Any]) -> A.BasicTransform:
        return A.Rotate(
            limit=(params["angle"], params["angle"]),
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=params["p"]
        )

    @staticmethod
    def Affine(params: dict[str, Any]) -> A.BasicTransform:
        return A.Affine(
            rotate=params["angle"],
            translate_percent=[x / 100 for x in params["shift"]],  # convert to percentage
            scale=params["scale"],
            shear=params["shear"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=params["p"]
        )

    @staticmethod
    def Equalize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Equalize(**params)

    @staticmethod
    def RandomCrop64(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomCrop(
            height=params["height"],
            width=params["width"],
            p=params["p"]
        )

    @staticmethod
    def RandomResizedCrop(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomResizedCrop(
            height=params["height"],
            width=params["width"],
            scale=params["scale"],
            ratio=params["ratio"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=params["p"]
        )

    @staticmethod
    def ShiftRGB(params: dict[str, Any]) -> A.BasicTransform:
        shift = params["pixel_shift"]
        return A.RGBShift(
            r_shift_limit=shift,
            g_shift_limit=shift,
            b_shift_limit=shift,
            p=params["p"]
        )

    @staticmethod
    def Resize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Resize(
            height=params["target_size"],
            width=params["target_size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=params["p"]
        )

    @staticmethod
    def RandomGamma(params: dict[str, Any]) -> A.BasicTransform:
        return A.RandomGamma(
            gamma_limit=(params["gamma"], params["gamma"]),
            p=params["p"]
        )

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> A.BasicTransform:
        return A.ToGray(p=params["p"])

    @staticmethod
    def ColorJitter(params: dict[str, Any]) -> A.BasicTransform:
        return A.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=params["p"]
        )

    @staticmethod
    def PlankianJitter(params: dict[str, Any]) -> A.BasicTransform:
        return A.PlanckianJitter(
            mode=params["mode"],
            p=params["p"]
        )

    @staticmethod
    def RandomPerspective(params: dict[str, Any]) -> A.BasicTransform:
        return A.Perspective(
            scale=params["scale"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=params["p"]
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> A.BasicTransform:
        return A.GaussianBlur(
            blur_limit=params["kernel_size"][0],  # assuming square kernel
            sigma_limit=(params["sigma"], params["sigma"]),
            p=params["p"]
        )

    @staticmethod
    def MedianBlur(params: dict[str, Any]) -> A.BasicTransform:
        return A.MedianBlur(
            blur_limit=(params["blur_limit"], params["blur_limit"]),
            p=params["p"]
        )

    @staticmethod
    def MotionBlur(params: dict[str, Any]) -> A.BasicTransform:
        return A.MotionBlur(
            blur_limit=params["kernel_size"],
            p=params["p"]
        )

    @staticmethod
    def Posterize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Posterize(
            num_bits=params["bits"],
            p=params["p"]
        )

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> A.BasicTransform:
        return A.ImageCompression(
            quality_lower=params["quality"],
            quality_upper=params["quality"],
            p=params["p"]
        )

    @staticmethod
    def GaussianNoise(params: dict[str, Any]) -> A.BasicTransform:
        return A.GaussNoise(
            var_limit=params["var"] * 255,  # convert to 0-255 range
            mean=params["mean"],
            per_channel=params["per_channel"],
            p=params["p"]
        )

    @staticmethod
    def Elastic(params: dict[str, Any]) -> A.BasicTransform:
        return A.ElasticTransform(
            alpha=params["alpha"],
            sigma=params["sigma"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            approximate=params["approximate"],
            p=params["p"]
        )

    @staticmethod
    def Normalize(params: dict[str, Any]) -> A.BasicTransform:
        return A.Normalize(
            mean=params["mean"],
            std=params["std"],
            p=params["p"]
        )

    @staticmethod
    def __call__(transform: A.BasicTransform, image: np.ndarray) -> np.ndarray:
        """Apply the transform to the image"""
        result = transform(image=image)
        return result["image"]