"""Torchvision implementations of transforms for videos in custom format."""

from typing import Any

import torch
import torchvision.transforms.v2 as tv_transforms

from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec

# Force CPU only for fair benchmarking
torch.set_num_threads(1)

# Required: Library name for dependency installation
LIBRARY = "torchvision"


# Required: Define how to apply transforms to videos
def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    """Apply torchvision transform to video tensor

    Args:
        transform: Torchvision transform instance
        video: torch.Tensor of shape (T, C, H, W)

    Returns:
        Transformed video as torch.Tensor
    """
    # Apply transform to each frame
    return transform(video)


# Helper function to create transforms from specs
def create_transform(spec: TransformSpec) -> Any | None:
    """Create a Torchvision transform from a TransformSpec."""
    params = spec.params

    if spec.name == "Resize":
        return tv_transforms.Resize(
            size=params["target_size"],
            interpolation=tv_transforms.InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else tv_transforms.InterpolationMode.NEAREST,
            antialias=True,
        )
    if spec.name == "RandomCrop128":
        return tv_transforms.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True)
    if spec.name == "RandomResizedCrop":
        return tv_transforms.RandomResizedCrop(
            size=params["size"],
            scale=params["scale"],
            ratio=params["ratio"],
            interpolation=tv_transforms.InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else tv_transforms.InterpolationMode.NEAREST,
        )
    if spec.name == "CenterCrop128":
        return tv_transforms.CenterCrop(size=(params["height"], params["width"]))
    if spec.name == "HorizontalFlip":
        return tv_transforms.RandomHorizontalFlip(**params)
    if spec.name == "VerticalFlip":
        return tv_transforms.RandomVerticalFlip(**params)
    if spec.name == "Pad":
        return tv_transforms.Pad(padding=params["padding"], fill=params["fill"], padding_mode=params["border_mode"])
    if spec.name == "Rotate":
        return tv_transforms.RandomRotation(
            degrees=params["angle"],
            interpolation=tv_transforms.InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else tv_transforms.InterpolationMode.NEAREST,
            fill=params["fill"],
        )
    if spec.name == "Affine":
        return tv_transforms.RandomAffine(
            degrees=params["angle"],
            translate=[x / 100 for x in params["shift"]],  # Convert to relative coordinates
            scale=(params["scale"], params["scale"]),
            shear=params["shear"],
            interpolation=tv_transforms.InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else tv_transforms.InterpolationMode.NEAREST,
        )
    if spec.name == "Perspective":
        return tv_transforms.RandomPerspective(
            distortion_scale=params["scale"][1],  # Using max scale
            interpolation=tv_transforms.InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else tv_transforms.InterpolationMode.NEAREST,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "Elastic":
        return tv_transforms.ElasticTransform(
            alpha=params["alpha"],
            sigma=params["sigma"],
            interpolation=tv_transforms.InterpolationMode.BILINEAR
            if params["interpolation"] == "bilinear"
            else tv_transforms.InterpolationMode.NEAREST,
        )
    if spec.name == "ColorJitter":
        return tv_transforms.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
        )
    if spec.name == "ChannelShuffle":
        return tv_transforms.RandomChannelPermutation()
    if spec.name == "Grayscale":
        return tv_transforms.Grayscale(
            num_output_channels=params["num_output_channels"],
        )
    if spec.name == "GaussianBlur":
        return tv_transforms.GaussianBlur(kernel_size=params["kernel_size"], sigma=(params["sigma"], params["sigma"]))
    if spec.name == "Invert":
        return tv_transforms.RandomInvert(p=1)
    if spec.name == "Posterize":
        return tv_transforms.RandomPosterize(bits=params["bits"], p=1)
    if spec.name == "Solarize":
        return tv_transforms.RandomSolarize(threshold=params["threshold"], p=1)
    if spec.name == "Sharpen":
        return tv_transforms.RandomAdjustSharpness(sharpness_factor=params["lightness"][0], p=1)
    if spec.name == "AutoContrast":
        return tv_transforms.RandomAutocontrast(p=1)
    if spec.name == "Equalize":
        return tv_transforms.RandomEqualize(p=1)
    if spec.name == "Normalize":
        return tv_transforms.Compose(
            [
                tv_transforms.ConvertImageDtype(torch.float32),  # Convert to float32 first
                tv_transforms.Normalize(mean=params["mean"], std=params["std"]),
            ],
        )
    if spec.name == "Erasing":
        return tv_transforms.RandomErasing(
            scale=params["scale"],
            ratio=params["ratio"],
            value=params["fill"],
            p=1,
        )
    if spec.name == "JpegCompression":
        return tv_transforms.JPEG(quality=params["quality"])
    if spec.name == "Brightness":
        return tv_transforms.ColorJitter(brightness=params["brightness_limit"], contrast=0.0, saturation=0.0, hue=0.0)
    if spec.name == "Contrast":
        return tv_transforms.ColorJitter(brightness=0.0, contrast=params["contrast_limit"], saturation=0.0, hue=0.0)
    # Skip transforms not supported by torchvision
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
