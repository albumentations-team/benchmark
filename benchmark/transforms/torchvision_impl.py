from typing import Any

import torch
from torchvision.transforms import InterpolationMode, v2

torch.set_num_threads(1)


class TorchvisionImpl:
    """Torchvision implementations of transforms"""

    @staticmethod
    def Resize(params: dict[str, Any]) -> v2.Transform:
        return v2.Resize(
            size=params["target_size"],
            interpolation=InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else InterpolationMode.NEAREST,
            antialias=True,
        )

    @staticmethod
    def RandomCrop128(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True)

    @staticmethod
    def RandomResizedCrop(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomResizedCrop(
            size=(params["height"], params["width"]),
            scale=params["scale"],
            ratio=params["ratio"],
            interpolation=InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else InterpolationMode.NEAREST,
        )

    @staticmethod
    def CenterCrop128(params: dict[str, Any]) -> v2.Transform:
        return v2.CenterCrop(size=(params["height"], params["width"]), pad_if_needed=True)

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomHorizontalFlip(**params)

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomVerticalFlip(**params)

    @staticmethod
    def Pad(params: dict[str, Any]) -> v2.Transform:
        return v2.Pad(padding=params["padding"], fill=params["fill"], padding_mode=params["border_mode"])

    @staticmethod
    def Rotate(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomRotation(
            degrees=params["angle"],
            interpolation=InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else InterpolationMode.NEAREST,
            fill=params["fill"],
        )

    @staticmethod
    def Affine(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomAffine(
            degrees=params["angle"],
            translate=[x / 100 for x in params["shift"]],  # Convert to relative coordinates
            scale=(params["scale"], params["scale"]),
            shear=params["shear"],
            interpolation=InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else InterpolationMode.NEAREST,
        )

    @staticmethod
    def Perspective(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomPerspective(
            distortion_scale=params["scale"][1],  # Using max scale
            interpolation=InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else InterpolationMode.NEAREST,
            fill=params["fill"],
            p=1,
        )

    @staticmethod
    def Elastic(params: dict[str, Any]) -> v2.Transform:
        return v2.ElasticTransform(
            alpha=params["alpha"],
            sigma=params["sigma"],
            interpolation=InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else InterpolationMode.NEAREST,
        )

    @staticmethod
    def ColorJitter(params: dict[str, Any]) -> v2.Transform:
        return v2.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
        )

    @staticmethod
    def ChannelShuffle(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomChannelPermutation()

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> v2.Transform:
        return v2.Grayscale(
            num_output_channels=params["num_output_channels"],
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> v2.Transform:
        return v2.GaussianBlur(kernel_size=params["kernel_size"], sigma=(params["sigma"], params["sigma"]))

    @staticmethod
    def Invert(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomInvert(p=1)

    @staticmethod
    def Posterize(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomPosterize(bits=params["bits"], p=1)

    @staticmethod
    def Solarize(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomSolarize(threshold=params["threshold"], p=1)

    @staticmethod
    def Sharpen(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomAdjustSharpness(sharpness_factor=params["lightness"][0], p=1)

    @staticmethod
    def AutoContrast(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomAutocontrast(p=1)

    @staticmethod
    def Equalize(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomEqualize(p=1)

    @staticmethod
    def Normalize(params: dict[str, Any]) -> v2.Transform:
        return v2.Compose(
            [
                v2.ConvertImageDtype(torch.float32),  # Convert to float32 first
                v2.Normalize(mean=params["mean"], std=params["std"]),
            ],
        )

    @staticmethod
    def Erasing(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomErasing(
            scale=params["scale"],
            ratio=params["ratio"],
            value=params["fill"],
            p=1,
        )

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> v2.Transform:
        return v2.JPEG(quality=params["quality"])

    @staticmethod
    def Brightness(params: dict[str, Any]) -> v2.Transform:
        return v2.ColorJitter(brightness=params["brightness_limit"], contrast=0.0, saturation=0.0, hue=0.0)

    @staticmethod
    def Contrast(params: dict[str, Any]) -> v2.Transform:
        return v2.ColorJitter(brightness=0.0, contrast=params["contrast_limit"], saturation=0.0, hue=0.0)

    @staticmethod
    def __call__(transform: v2.Transform, image: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the image"""
        return transform(image).contiguous()
