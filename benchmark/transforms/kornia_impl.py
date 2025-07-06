"""Kornia implementations of transforms for images in custom format."""

from typing import Any

import kornia
import kornia.augmentation as Kaug
import torch

from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec

# Force CPU only for fair benchmarking
device = torch.device("cpu")

# Required: Library name for dependency installation
LIBRARY = "kornia"


# Required: Define how to apply transforms to images
def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    """Apply kornia transform to a single image

    Args:
        transform: Kornia augmentation instance
        image: torch.Tensor of shape (C, H, W)

    Returns:
        Transformed image as torch.Tensor
    """
    # Apply transform directly - let PyTorch handle memory layout
    return transform(image)


# Helper function to create transforms from specs
def create_transform(spec: TransformSpec) -> Any | None:
    """Create a Kornia transform from a TransformSpec."""
    params = spec.params

    if spec.name == "ColorJitter":
        return Kaug.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=1,
            same_on_batch=False,
        ).to(device)
    if spec.name == "AutoContrast":
        return Kaug.RandomAutoContrast(p=1).to(device)
    if spec.name == "Blur":
        return Kaug.RandomBoxBlur(
            p=1,
            kernel_size=(params["radius"], params["radius"]),
            border_type=params["border_mode"],
        ).to(device)
    if spec.name == "Brightness":
        return Kaug.RandomBrightness(
            brightness=params["brightness_limit"],
            p=1,
        ).to(device)
    if spec.name == "ChannelDropout":
        return Kaug.RandomChannelDropout(p=1).to(device)
    if spec.name == "ChannelShuffle":
        return Kaug.RandomChannelShuffle(p=1).to(device)
    if spec.name == "CLAHE":
        return Kaug.RandomClahe(
            p=1,
            clip_limit=params["clip_limit"],
            grid_size=params["tile_grid_size"],
        ).to(device)
    if spec.name == "Contrast":
        return Kaug.RandomContrast(
            contrast=params["contrast_limit"],
            p=1,
        ).to(device)
    if spec.name == "Equalize":
        return Kaug.RandomEqualize(p=1).to(device)
    if spec.name == "RandomGamma":
        gamma = params["gamma"] / 100
        return Kaug.RandomGamma(
            gamma=(gamma, gamma),
            p=1,
        ).to(device)
    if spec.name == "GaussianBlur":
        return Kaug.RandomGaussianBlur(
            kernel_size=params["kernel_size"],
            sigma=(params["sigma"], params["sigma"]),
            p=1,
        ).to(device)
    if spec.name == "LinearIllumination":
        return Kaug.RandomLinearIllumination(
            gain=params["gain"],
            p=1,
        ).to(device)
    if spec.name == "CornerIllumination":
        return Kaug.RandomLinearCornerIllumination(
            gain=params["gain"],
            p=1,
        ).to(device)
    if spec.name == "GaussianIllumination":
        return Kaug.RandomGaussianIllumination(
            gain=params["gain"],
            p=1,
        ).to(device)
    if spec.name == "GaussianNoise":
        return Kaug.RandomGaussianNoise(
            mean=params["mean"],
            std=params["std"],
            p=1,
        ).to(device)
    if spec.name == "Grayscale":
        return Kaug.RandomGrayscale(p=1).to(device)
    if spec.name == "Hue":
        return Kaug.RandomHue(
            hue=params["hue"],
            p=1,
        ).to(device)
    if spec.name == "Invert":
        return Kaug.RandomInvert(p=1).to(device)
    if spec.name == "JpegCompression":
        return Kaug.RandomJPEG(
            jpeg_quality=params["quality"],
            p=1,
        ).to(device)
    if spec.name == "MedianBlur":
        kernel_size = params["blur_limit"]
        return Kaug.RandomMedianBlur(
            kernel_size=(kernel_size, kernel_size),
            p=1,
        ).to(device)
    if spec.name == "MotionBlur":
        return Kaug.RandomMotionBlur(
            kernel_size=params["kernel_size"],
            angle=params["angle_range"],
            direction=params["direction_range"],
            p=1,
        ).to(device)
    if spec.name == "PlankianJitter":
        return Kaug.RandomPlanckianJitter(
            mode=params["mode"],
            p=1,
        ).to(device)
    if spec.name == "PlasmaBrightness":
        return Kaug.RandomPlasmaBrightness(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        ).to(device)
    if spec.name == "PlasmaContrast":
        return Kaug.RandomPlasmaContrast(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        ).to(device)
    if spec.name == "PlasmaShadow":
        return Kaug.RandomPlasmaShadow(
            roughness=(params["roughness"], params["roughness"]),
            p=1,
        ).to(device)
    if spec.name == "Rain":
        return Kaug.RandomRain(
            drop_width=(params["drop_width"], params["drop_width"]),
            drop_height=(params["drop_height"], params["drop_height"]),
            p=1,
        ).to(device)
    if spec.name == "RGBShift":
        return Kaug.RandomRGBShift(
            r_shift_limit=params["pixel_shift"] / 255.0,
            g_shift_limit=params["pixel_shift"] / 255.0,
            b_shift_limit=params["pixel_shift"] / 255.0,
            p=1,
        ).to(device)
    if spec.name == "SaltAndPepper":
        return Kaug.RandomSaltAndPepperNoise(
            amount=params["amount"],
            salt_vs_pepper=params["salt_vs_pepper"],
            p=1,
        ).to(device)
    if spec.name == "Saturation":
        return Kaug.RandomSaturation(
            saturation=params["saturation_factor"],
            p=1,
        ).to(device)
    if spec.name == "Sharpen":
        return Kaug.RandomSharpness(
            p=1,
        ).to(device)
    if spec.name == "Snow":
        return Kaug.RandomSnow(
            snow_coefficient=params["snow_point_range"],
            p=1,
        ).to(device)
    if spec.name == "Solarize":
        return Kaug.RandomSolarize(
            thresholds=params["threshold"],
            p=1,
        ).to(device)
    if spec.name == "CenterCrop128":
        return Kaug.CenterCrop(
            size=(params["height"], params["width"]),
            p=1,
        ).to(device)
    if spec.name == "Affine":
        # Create a simple affine transform with fixed parameters
        angle_degrees = float(params["angle"])
        translate = float(params["shift"][0]) / 255.0
        scale_factor = float(params["scale"])
        shear_value = float(params["shear"]) if isinstance(params["shear"], int | float) else 0.0

        # Create a fixed affine transform
        return kornia.geometry.transform.Affine(
            angle=torch.tensor([angle_degrees]),
            translation=torch.tensor([[translate, translate]]),
            scale_factor=torch.tensor([[scale_factor, scale_factor]]),
            shear=torch.tensor([[shear_value, shear_value]]),
            center=torch.tensor([[160.0, 120.0]]),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(device)
    if spec.name == "RandomCrop128":
        return Kaug.RandomCrop(
            size=(params["height"], params["width"]),
            pad_if_needed=True,
            p=1,
        ).to(device)
    if spec.name == "Elastic":
        return Kaug.RandomElasticTransform(
            p=1,
            sigma=(params["sigma"], params["sigma"]),
            alpha=(params["alpha"], params["alpha"]),
        ).to(device)
    if spec.name == "Erasing":
        return Kaug.RandomErasing(
            p=1,
            scale=params["scale"],
            ratio=params["ratio"],
            value=params["fill"],
        ).to(device)
    if spec.name == "OpticalDistortion":
        return kornia.augmentation.RandomFisheye(
            center_x=torch.tensor([-0.3, 0.3]),
            center_y=torch.tensor([-0.3, 0.3]),
            gamma=torch.tensor([0.9, 1.1]),
            p=1,
        ).to(device)
    if spec.name == "HorizontalFlip":
        return Kaug.RandomHorizontalFlip(p=1).to(device)
    if spec.name == "Perspective":
        return Kaug.RandomPerspective(
            distortion_scale=params["scale"][1],
            resample=params["interpolation"],
            p=1,
        ).to(device)
    if spec.name == "RandomResizedCrop":
        return Kaug.RandomResizedCrop(
            size=params["size"],
            scale=params["scale"],
            ratio=params["ratio"],
            p=1,
        ).to(device)
    if spec.name == "RandomRotate90":
        return Kaug.RandomRotation(
            times=(0, 3),
            p=1,
        ).to(device)
    if spec.name == "Rotate":
        # Use RandomRotation with fixed degrees to avoid tensor shape issues
        return Kaug.RandomRotation(
            degrees=(params["angle"], params["angle"]),  # Fixed rotation
            p=1,
        ).to(device)
    if spec.name == "Shear":
        return Kaug.RandomShear(
            shear=params["shear"],
            p=1,
        ).to(device)
    if spec.name == "ThinPlateSpline":
        return Kaug.RandomThinPlateSpline(
            p=1,
        ).to(device)
    if spec.name == "VerticalFlip":
        return Kaug.RandomVerticalFlip(p=1).to(device)
    if spec.name == "Resize":
        return Kaug.Resize(
            size=(params["target_size"], params["target_size"]),
            p=1,
        ).to(device)
    if spec.name == "Normalize":
        return Kaug.Normalize(
            mean=params["mean"],
            std=params["std"],
            p=1,
        ).to(device)
    if spec.name == "Posterize":
        return Kaug.RandomPosterize(
            bits=float(params["bits"]),
            p=1,
        ).to(device)
    # Skip transforms not supported by kornia
    return None


# Required: Transform definitions from specs
TRANSFORMS = [
    {
        "name": spec.name,
        "transform": create_transform(spec),
    }
    for spec in TRANSFORM_SPECS
    if create_transform(spec) is not None
]
