from typing import Any

import kornia
import kornia.augmentation as Kaug
import torch

torch.set_num_threads(1)


class KorniaImpl:
    """Kornia implementations of transforms"""

    @staticmethod
    def ColorJitter(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=1,
        )

    @staticmethod
    def AutoContrast(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomAutoContrast(p=1)

    @staticmethod
    def Blur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomBoxBlur(
            p=1,
            kernel_size=(params["radius"], params["radius"]),
            border_type=params["border_mode"],
        )

    @staticmethod
    def Brightness(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomBrightness(
            brightness=params["brightness_limit"],
            p=1,
        )

    @staticmethod
    def ChannelDropout(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomChannelDropout(p=1)

    @staticmethod
    def ChannelShuffle(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomChannelShuffle(p=1)

    @staticmethod
    def CLAHE(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomClahe(p=1, clip_limit=params["clip_limit"], grid_size=params["tile_grid_size"])

    @staticmethod
    def Contrast(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomContrast(
            contrast=params["contrast_limit"],
            p=1,
        )

    @staticmethod
    def Equalize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomEqualize(p=1)

    @staticmethod
    def RandomGamma(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        gamma = params["gamma"] / 100
        return Kaug.RandomGamma(
            gamma=(gamma, gamma),
            p=1,
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGaussianBlur(
            kernel_size=params["kernel_size"],
            sigma=(params["sigma"], params["sigma"]),
            p=1,
        )

    @staticmethod
    def LinearIllumination(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomLinearIllumination(
            gain=params["gain"],
            p=1,
        )

    @staticmethod
    def CornerIllumination(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomLinearCornerIllumination(
            gain=params["gain"],
            p=1,
        )

    @staticmethod
    def GaussianIllumination(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGaussianIllumination(
            gain=params["gain"],
            p=1,
        )

    @staticmethod
    def GaussianNoise(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGaussianNoise(
            mean=params["mean"],
            std=params["std"],
            p=1,
        )

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGrayscale(p=1)

    @staticmethod
    def Hue(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomHue(
            hue=params["hue"],
            p=1,
        )

    @staticmethod
    def Invert(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomInvert(p=1)

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomJPEG(
            jpeg_quality=params["quality"],
            p=1,
        )

    @staticmethod
    def MedianBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        kernel_size = params["blur_limit"]
        return Kaug.RandomMedianBlur(
            kernel_size=(kernel_size, kernel_size),
            p=1,
        )

    @staticmethod
    def MotionBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomMotionBlur(
            kernel_size=params["kernel_size"],
            angle=params["angle_range"],
            direction=params["direction_range"],
            p=1,
        )

    @staticmethod
    def PlankianJitter(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlanckianJitter(
            mode=params["mode"],
            p=1,
        )

    @staticmethod
    def PlasmaBrightness(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlasmaBrightness(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        )

    @staticmethod
    def PlasmaContrast(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlasmaContrast(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        )

    @staticmethod
    def PlasmaShadow(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlasmaShadow(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        )

    @staticmethod
    def Rain(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomRain(
            drop_width=(params["drop_width"], params["drop_width"]),
            drop_height=(params["drop_height"], params["drop_height"]),
            p=1,
        )

    @staticmethod
    def RGBShift(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomRGBShift(
            r_shift_limit=params["pixel_shift"] / 255.0,
            g_shift_limit=params["pixel_shift"] / 255.0,
            b_shift_limit=params["pixel_shift"] / 255.0,
            p=1,
        )

    @staticmethod
    def SaltAndPepper(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSaltAndPepperNoise(
            amount=params["amount"],
            salt_vs_pepper=params["salt_vs_pepper"],
            p=1,
        )

    @staticmethod
    def Saturation(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSaturation(
            saturation=params["saturation_factor"],
            p=1,
        )

    @staticmethod
    def Sharpen(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSharpness(
            p=1,
        )

    @staticmethod
    def Snow(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSnow(
            snow_coefficient=params["snow_point_range"],
            p=1,
        )

    @staticmethod
    def Solarize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSolarize(
            thresholds=params["threshold"],
            p=1,
        )

    @staticmethod
    def CenterCrop128(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.CenterCrop(
            size=(params["height"], params["width"]),
            pad_if_needed=True,
            p=1,
        )

    @staticmethod
    def Affine(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomAffine(
            degrees=params["angle"],
            translate=params["shift"][0] / 255.0,
            scale=(params["scale"], params["scale"]),
            shear=params["shear"],
            p=1,
        )

    @staticmethod
    def RandomCrop128(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomCrop(
            size=(params["height"], params["width"]),
            pad_if_needed=True,
            p=1,
        )

    @staticmethod
    def Elastic(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomElasticTransform(
            alpha=(params["alpha"], params["alpha"]),
            sigma=(params["sigma"], params["sigma"]),
            p=1,
        )

    @staticmethod
    def Erasing(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomErasing(
            p=1,
            scale=params["scale"],
            ratio=params["ratio"],
            value=params["fill"],
        )

    @staticmethod
    def OpticalDistortion(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return kornia.augmentation.RandomFisheye(
            center_x=torch.tensor([-0.3, 0.3]),
            center_y=torch.tensor([-0.3, 0.3]),
            gamma=torch.tensor([0.9, 1.1]),
            p=1,
        )

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomHorizontalFlip(p=1)

    @staticmethod
    def Perspective(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPerspective(
            distortion_scale=params["scale"][1],
            resample=params["interpolation"],
            p=1,
        )

    @staticmethod
    def RandomResizedCrop(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomResizedCrop(
            size=(params["height"], params["width"]),
            scale=params["scale"],
            ratio=params["ratio"],
            p=1,
        )

    @staticmethod
    def RandomRotate90(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomRotation90(
            times=(0, 3),
            p=1,
        )

    @staticmethod
    def Rotate(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        # Convert degrees to radians for rotation
        angle = torch.tensor(params["angle"]) * (torch.pi / 180.0)
        return kornia.geometry.transform.Rotate(
            angle=angle,
            mode="bilinear",
        )

    @staticmethod
    def Shear(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomShear(
            shear=params["shear"],
            p=1,
        )

    @staticmethod
    def ThinPlateSpline(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomThinPlateSpline(
            p=1,
        )

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomVerticalFlip(p=1)

    @staticmethod
    def Resize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.Resize(
            size=(params["target_size"], params["target_size"]),
            p=1,
        )

    @staticmethod
    def Normalize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.Normalize(
            mean=params["mean"],
            std=params["std"],
            p=1,
        )

    @staticmethod
    def Posterize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPosterize(
            bits=params["bits"],
            p=1,
        )

    @staticmethod
    def __call__(transform: Kaug.AugmentationBase2D, image: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the image"""
        return transform(image).contiguous()
