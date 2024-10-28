from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
torch.set_num_threads(1)

class TorchvisionImpl:
    """Torchvision implementations of transforms"""

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomHorizontalFlip(**params)

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomVerticalFlip(**params)

    @staticmethod
    def Rotate(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomRotation(
            degrees=params["angle"],
            interpolation=InterpolationMode.BILINEAR if params["interpolation"] == "bilinear" else InterpolationMode.NEAREST
        )

    @staticmethod
    def Affine(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomAffine(
            degrees=params["angle"],
            translate=[x/100 for x in params["shift"]],  # Convert to relative coordinates
            scale=(params["scale"], params["scale"]),
            shear=params["shear"],
            interpolation=InterpolationMode.BILINEAR if params["interpolation"] == "bilinear" else InterpolationMode.NEAREST
        )

    @staticmethod
    def Equalize(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomEqualize(**params)

    @staticmethod
    def RandomCrop80(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomCrop(
            size=(params["height"], params["width"])
        )

    @staticmethod
    def RandomResizedCrop(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomResizedCrop(
            size=(params["height"], params["width"]),
            scale=params["scale"],
            ratio=params["ratio"],
            interpolation=InterpolationMode.BILINEAR if params["interpolation"] == "bilinear" else InterpolationMode.NEAREST
        )

    @staticmethod
    def Resize(params: dict[str, Any]) -> v2.Transform:
        return v2.Resize(
            size=params["target_size"],
            interpolation=InterpolationMode.BILINEAR if params["interpolation"] == "bilinear" else InterpolationMode.NEAREST,
            antialias=True
        )

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomGrayscale(
            p=params["p"],
        )

    @staticmethod
    def ColorJitter(params: dict[str, Any]) -> v2.Transform:
        return v2.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"]
        )

    @staticmethod
    def RandomPerspective(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomPerspective(
            distortion_scale=params["scale"][1],  # Using max scale
            p=params["p"],
            interpolation=InterpolationMode.BILINEAR if params["interpolation"] == "bilinear" else InterpolationMode.NEAREST
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> v2.Transform:
        return v2.GaussianBlur(
            kernel_size=params["kernel_size"],
            sigma=(params["sigma"], params["sigma"])
        )

    @staticmethod
    def Posterize(params: dict[str, Any]) -> v2.Transform:
        return v2.RandomPosterize(
            bits=params["bits"],
            p=params["p"]
        )

    @staticmethod
    def Elastic(params: dict[str, Any]) -> v2.Transform:
        return v2.ElasticTransform(
            alpha=params["alpha"],
            sigma=params["sigma"],
            interpolation=InterpolationMode.BILINEAR if params["interpolation"] == "bilinear" else InterpolationMode.NEAREST
        )

    @staticmethod
    def Normalize(params: dict[str, Any]) -> v2.Transform:
        return v2.Compose([
        v2.ConvertImageDtype(torch.float32),  # Convert to float32 first
        v2.Normalize(
            mean=params["mean"],
            std=params["std"]
        )
    ])

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> v2.Transform:
        return v2.JPEG(quality=params["quality"])

    @staticmethod
    def __call__(transform: v2.Transform, image: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the image"""
        return transform(image).contiguous()
