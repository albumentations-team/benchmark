from typing import Any

import kornia
import kornia.augmentation as Kaug
import torch

torch.set_num_threads(1)


class KorniaVideoImpl:
    """Kornia implementations of video transforms"""

    @staticmethod
    def ColorJitter(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def AutoContrast(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomAutoContrast(p=1, same_on_batch=True)

    @staticmethod
    def Blur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomBoxBlur(
            p=1,
            kernel_size=(params["radius"], params["radius"]),
            border_type=params["border_mode"],
            same_on_batch=True,
        )

    @staticmethod
    def Brightness(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomBrightness(
            brightness=params["brightness_limit"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def ChannelDropout(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomChannelDropout(p=1, same_on_batch=True)

    @staticmethod
    def ChannelShuffle(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomChannelShuffle(p=1, same_on_batch=True)

    @staticmethod
    def CLAHE(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomClahe(
            p=1,
            clip_limit=params["clip_limit"],
            grid_size=params["tile_grid_size"],
            same_on_batch=True,
        )

    @staticmethod
    def Contrast(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomContrast(
            contrast=params["contrast_limit"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Equalize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomEqualize(p=1, same_on_batch=True)

    @staticmethod
    def RandomGamma(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        gamma = params["gamma"] / 100
        return Kaug.RandomGamma(
            gamma=(gamma, gamma),
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def GaussianBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGaussianBlur(
            kernel_size=params["kernel_size"],
            sigma=(params["sigma"], params["sigma"]),
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def LinearIllumination(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomLinearIllumination(
            gain=params["gain"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def CornerIllumination(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomLinearCornerIllumination(
            gain=params["gain"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def GaussianIllumination(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGaussianIllumination(
            gain=params["gain"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def GaussianNoise(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGaussianNoise(
            mean=params["mean"],
            std=params["std"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Grayscale(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomGrayscale(p=1, same_on_batch=True)

    @staticmethod
    def Hue(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomHue(
            hue=params["hue"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Invert(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomInvert(p=1, same_on_batch=True)

    @staticmethod
    def JpegCompression(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomJPEG(
            jpeg_quality=params["quality"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def MedianBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        kernel_size = params["blur_limit"]
        return Kaug.RandomMedianBlur(
            kernel_size=(kernel_size, kernel_size),
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def MotionBlur(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomMotionBlur(
            kernel_size=params["kernel_size"],
            angle=params["angle_range"],
            direction=params["direction_range"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def PlankianJitter(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlanckianJitter(
            mode=params["mode"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def PlasmaBrightness(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlasmaBrightness(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def PlasmaContrast(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlasmaContrast(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def PlasmaShadow(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPlasmaShadow(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Rain(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomRain(
            drop_width=(params["drop_width"], params["drop_width"]),
            drop_height=(params["drop_height"], params["drop_height"]),
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def RGBShift(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomRGBShift(
            r_shift_limit=params["pixel_shift"] / 255.0,
            g_shift_limit=params["pixel_shift"] / 255.0,
            b_shift_limit=params["pixel_shift"] / 255.0,
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def SaltAndPepper(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSaltAndPepperNoise(
            amount=params["amount"],
            salt_vs_pepper=params["salt_vs_pepper"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Saturation(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSaturation(
            saturation=params["saturation_factor"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Sharpen(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSharpness(
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Snow(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSnow(
            snow_coefficient=params["snow_point_range"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Solarize(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomSolarize(
            thresholds=params["threshold"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def CenterCrop128(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.CenterCrop(
            size=(params["height"], params["width"]),
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
            same_on_batch=True,
        )

    @staticmethod
    def RandomCrop128(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomCrop(
            size=(params["height"], params["width"]),
            pad_if_needed=True,
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def Elastic(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomElasticTransform(
            p=1,
            sigma=(params["sigma"], params["sigma"]),
            alpha=(params["alpha"], params["alpha"]),
            same_on_batch=True,
        )

    @staticmethod
    def Erasing(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomErasing(
            p=1,
            scale=params["scale"],
            ratio=params["ratio"],
            value=params["fill"],
            same_on_batch=True,
        )

    @staticmethod
    def OpticalDistortion(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return kornia.augmentation.RandomFisheye(
            center_x=torch.tensor([-0.3, 0.3]),
            center_y=torch.tensor([-0.3, 0.3]),
            gamma=torch.tensor([0.9, 1.1]),
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomHorizontalFlip(p=1, same_on_batch=True)

    @staticmethod
    def Perspective(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomPerspective(
            distortion_scale=params["scale"][1],
            resample=params["interpolation"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def RandomResizedCrop(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomResizedCrop(
            size=params["size"],
            scale=params["scale"],
            ratio=params["ratio"],
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def RandomRotate90(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomRotation(
            times=(0, 3),
            p=1,
            same_on_batch=True,
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
            same_on_batch=True,
        )

    @staticmethod
    def ThinPlateSpline(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomThinPlateSpline(
            p=1,
            same_on_batch=True,
        )

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> Kaug.AugmentationBase2D:
        return Kaug.RandomVerticalFlip(p=1, same_on_batch=True)

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
            same_on_batch=True,
        )

    @staticmethod
    def __call__(transform: Kaug.AugmentationBase2D, video: torch.Tensor) -> torch.Tensor:
        """Apply transform to video tensor

        Args:
            transform: Kornia transform to apply
            video: Video tensor of shape (T, C, H, W)

        Returns:
            Transformed video tensor of shape (T, C, H, W)
        """
        # Treat time dimension (T) as batch dimension
        # video shape is already (T, C, H, W) which is what Kornia expects for batched images
        # Apply transform directly to the video tensor
        # This will apply the same transform to all frames due to same_on_batch=True
        return transform(video)
