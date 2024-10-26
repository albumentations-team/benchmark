from typing import Any

import augly.image as imaugs
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Disable image size check
if hasattr(Image.core, 'set_threads'):
    Image.core.set_threads(1)


class AuglyImpl:
    """Augly implementations of transforms"""

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.HFlip(p=params["p"])

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.VFlip(p=params["p"])

    @staticmethod
    def Rotate(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.RandomRotation(
            min_degrees=params["angle"],
            max_degrees=params["angle"],
            p=params["p"]
        )

    @staticmethod
    def RandomCrop64(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        # Augly uses relative coordinates (0-1)
        # Using fixed values as augly's crop works differently
        return imaugs.Crop(
            x1=0.25,
            y1=0.25,
            x2=0.75,
            y2=0.75,
            p=params["p"]
        )

    @staticmethod
    def Resize(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.Resize(
            width=params["target_size"],
            height=params["target_size"],
            resample=Image.BILINEAR if params["interpolation"] == "bilinear" else Image.NEAREST,
            p=params["p"]
        )

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.Grayscale(p=params["p"])

    @staticmethod
    def ColorJitter(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.ColorJitter(
            brightness_factor=params["brightness"],
            contrast_factor=params["contrast"],
            saturation_factor=params["saturation"],
            p=params["p"]
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.Blur(
            radius=params["sigma"],
            p=params["p"]
        )

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.EncodingQuality(
            quality=params["quality"],
            p=params["p"]
        )

    @staticmethod
    def GaussianNoise(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.RandomNoise(
            mean=params["mean"],
            var=params["var"],
            p=params["p"]
        )

    @staticmethod
    def Blur(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.Blur(
            radius=params["radius"],
            p=params["p"]
        )

    @staticmethod
    def Brightness(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.RandomBrightness(
            min_factor=params["brightness_limit"][0],
            max_factor=params["brightness_limit"][1],
            p=params["p"]
        )

    @staticmethod
    def Contrast(params: dict[str, Any]) -> imaugs.transforms.BaseTransform:
        return imaugs.Contrast(
            factor=params["contrast_limit"][0],
            p=params["p"]
        )

    @staticmethod
    def __call__(transform: imaugs.transforms.BaseTransform, image: Image.Image) -> Image.Image:
        """Apply the transform to the image"""
        return transform(image)
