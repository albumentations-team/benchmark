# benchmark/transforms/imgaug_impl.py
from typing import Any

import cv2
import numpy as np
from imgaug import augmenters as iaa

# Ensure single thread
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class ImgaugImpl:
    """Imgaug implementations of transforms"""

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.Fliplr(p=params["p"])

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.Flipud(p=params["p"])

    @staticmethod
    def Rotate(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.Affine(
            rotate=(params["angle"], params["angle"]),
            order=1 if params["interpolation"] == "bilinear" else 0,
            mode=params["mode"],
        )

    @staticmethod
    def Affine(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.Affine(
            scale=(params["scale"], params["scale"]),
            rotate=(params["angle"], params["angle"]),
            translate_px=params["shift"],
            order=1 if params["interpolation"] == "bilinear" else 0,
            mode=params["mode"],
        )

    @staticmethod
    def Equalize(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.AllChannelsHistogramEqualization()

    @staticmethod
    def RandomCrop80(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.CropToFixedSize(width=params["width"], height=params["height"])

    @staticmethod
    def ShiftRGB(params: dict[str, Any]) -> iaa.Augmenter:
        shift = params["pixel_shift"]
        return iaa.Add(value=(-shift, shift), per_channel=params["per_channel"])

    @staticmethod
    def Resize(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.Resize(
            size=params["target_size"],
            interpolation="linear" if params["interpolation"] == "bilinear" else "nearest",
        )

    @staticmethod
    def RandomGamma(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.GammaContrast(
            gamma=params["gamma"] / 100,  # Convert to imgaug scale
        )

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.Grayscale(alpha=1.0)

    @staticmethod
    def RandomPerspective(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.PerspectiveTransform(
            scale=params["scale"],
            mode=params.get("mode", "replicate"),
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.GaussianBlur(sigma=params["sigma"])

    @staticmethod
    def MedianBlur(params: dict[str, Any]) -> iaa.Augmenter:
        blur_limit = params["blur_limit"]
        return iaa.MedianBlur(k=(blur_limit, blur_limit))

    @staticmethod
    def MotionBlur(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.MotionBlur(k=params["kernel_size"], angle=[params["angle"]], direction=params["direction"])

    @staticmethod
    def Posterize(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.Posterize(nb_bits=params["bits"])

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.JpegCompression(compression=params["quality"])

    @staticmethod
    def GaussianNoise(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.AdditiveGaussianNoise(
            loc=params["mean"],
            scale=(0, params["var"]),
            per_channel=params["per_channel"],
        )

    @staticmethod
    def Elastic(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.ElasticTransformation(
            alpha=params["alpha"],
            sigma=params["sigma"],
            order=1 if params["interpolation"] == "bilinear" else 0,
        )

    @staticmethod
    def Clahe(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.AllChannelsCLAHE(clip_limit=params["clip_limit"], tile_grid_size_px=params["tile_grid_size"])

    @staticmethod
    def CoarseDropout(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.CoarseDropout()

    @staticmethod
    def Blur(params: dict[str, Any]) -> iaa.Augmenter:
        return iaa.AverageBlur(k=params["radius"])

    @staticmethod
    def __call__(transform: iaa.Augmenter, image: np.ndarray) -> np.ndarray:
        """Apply the transform to the image"""
        return np.ascontiguousarray(transform.augment_image(image))
