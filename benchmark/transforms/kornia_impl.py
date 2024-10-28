from typing import Any

import kornia.augmentation as Kaug
import torch

torch.set_num_threads(1)


class KorniaImpl:
    """Kornia implementations of transforms"""

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomHorizontalFlip(**params)

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomVerticalFlip(**params)

    @staticmethod
    def Rotate(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        # Convert degrees to radians for rotation
        angle = torch.tensor(params["angle"]) * (torch.pi / 180.0)
        return Kaug.RandomRotation(
            degrees=angle,
            resample="bilinear",
            p=params["p"],
        )

    @staticmethod
    def Affine(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomAffine(
            degrees=params["angle"],
            translate=0.1,
            scale=[params["scale"], params["scale"]],
            shear=params["shear"],
            p=params["p"],
        )

    @staticmethod
    def Clahe(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomClahe(
            clip_limit=params["clip_limit"],
            grid_size=params["tile_grid_size"],
            p=params["p"],
        )

    @staticmethod
    def Equalize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomEqualize(**params)

    @staticmethod
    def RandomCrop80(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomCrop(
            size=(params["height"], params["width"]),
            p=params["p"],
        )

    @staticmethod
    def RandomResizedCrop(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomResizedCrop(
            size=(params["height"], params["width"]),
            scale=params["scale"],
            ratio=params["ratio"],
            p=params["p"],
        )

    @staticmethod
    def Resize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.Resize(
            size=(params["target_size"], params["target_size"]),
            p=params["p"],
        )

    @staticmethod
    def RandomGamma(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        gamma = params["gamma"] / 100  # Convert to kornia scale
        return Kaug.RandomGamma(
            gamma=(gamma, gamma),
            p=params["p"],
        )

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGrayscale(p=params["p"])

    @staticmethod
    def ColorJitter(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=params["p"],
        )

    @staticmethod
    def PlankianJitter(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlanckianJitter(
            mode=params["mode"],
            p=params["p"],
        )

    @staticmethod
    def RandomPerspective(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPerspective(
            distortion_scale=params["scale"][1],  # Using max scale
            p=params["p"],
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGaussianBlur(
            kernel_size=params["kernel_size"],
            sigma=(params["sigma"], params["sigma"]),
            p=params["p"],
        )

    @staticmethod
    def MedianBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        kernel_size = params["blur_limit"]
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        return Kaug.RandomMedianBlur(
            kernel_size=(kernel_size, kernel_size),
            p=params["p"],
        )

    @staticmethod
    def MotionBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomMotionBlur(
            kernel_size=params["kernel_size"],
            angle=params["angle"],
            direction=params["direction"],
            p=params["p"],
        )

    @staticmethod
    def Posterize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPosterize(
            bits=params["bits"],
            p=params["p"],
        )

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomJPEG(
            jpeg_quality=params["quality"],
            p=params["p"],
        )

    @staticmethod
    def Elastic(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        sigma = torch.tensor([params["sigma"], params["sigma"]]).float()
        alpha = torch.tensor([params["alpha"], params["alpha"]]).float()
        return Kaug.RandomElasticTransform(
            alpha=alpha,
            sigma=sigma,
            p=params["p"],
        )

    @staticmethod
    def Normalize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.Normalize(
            mean=params["mean"],
            std=params["std"],
            p=params["p"],
        )

    @staticmethod
    def Brightness(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomBrightness(
            brightness=params["brightness_limit"],
            p=params["p"],
        )

    @staticmethod
    def Contrast(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomContrast(
            contrast=params["contrast_limit"],
            p=params["p"],
        )

    @staticmethod
    def GaussianNoise(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGaussianNoise(
            mean=params["mean"],
            std=params["var"],
            p=params["p"],
        )

    @staticmethod
    def RGBShift(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomRGBShift(
            r_shift_limit=params["r_shift_limit"] / 255.0,
            g_shift_limit=params["g_shift_limit"] / 255.0,
            b_shift_limit=params["b_shift_limit"] / 255.0,
            p=params["p"],
        )

    @staticmethod
    def Solarize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSolarize(
            threshold=params["threshold"] / 255.0,
            p=params["p"],
        )

    @staticmethod
    def CoarseDropout(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomErasing(
            p=params["p"],
        )

    @staticmethod
    def __call__(transform: Kaug.AugmentationBase2D, image: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the image"""
        return transform(image).contiguous()
