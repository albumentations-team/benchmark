"""TorchVision transform factories for the active RGB recipe catalog."""

import random
from typing import Any

import torch
import torchvision.transforms.v2 as tv_transforms

from augbench.implementations.geometry import aspect_size, centered_padding
from augbench.implementations.specs import TransformSpec

LIBRARY = "torchvision"


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


def create_transform(spec: TransformSpec) -> Any | None:
    builder = _TRANSFORM_BUILDERS.get(spec.name)
    return builder(spec.params) if builder is not None else None


def _build_resize(params: dict[str, Any]) -> Any | None:
    return tv_transforms.Resize(
        size=(params["target_size"], params["target_size"]),
        interpolation=tv_transforms.InterpolationMode.BILINEAR
        if params["interpolation"] == "bilinear"
        else tv_transforms.InterpolationMode.NEAREST,
        antialias=params.get("antialias", True),
    )


def _build_random_crop224(params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True)


def _build_random_resized_crop(params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomResizedCrop(
        size=params["size"],
        scale=params["scale"],
        ratio=params["ratio"],
        interpolation=tv_transforms.InterpolationMode.BILINEAR
        if params["interpolation"] == "bilinear"
        else tv_transforms.InterpolationMode.NEAREST,
        antialias=params.get("antialias", True),
    )


def _build_horizontal_flip(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomHorizontalFlip(p=1)


def _build_vertical_flip(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomVerticalFlip(p=1)


def _build_random_rotate90(params: dict[str, Any]) -> Any:
    return tv_transforms.Lambda(
        lambda image: torch.rot90(image, random.randint(*params["times"]), (-2, -1))  # noqa: S311 - augmentation RNG.
    )


def _build_center_crop(params: dict[str, Any]) -> Any:
    return tv_transforms.CenterCrop((params["height"], params["width"]))


def _build_crop(params: dict[str, Any]) -> Any:
    return tv_transforms.Lambda(lambda image: tv_transforms.functional.crop(image, **params))


def _build_crop_and_pad(params: dict[str, Any]) -> Any:
    top, right, bottom, left = params["sides"]
    return tv_transforms.Pad((left, top, right, bottom), fill=0)


def _build_resize_by_side(params: dict[str, Any], *, cover: bool) -> Any:
    def resize(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        height, width = aspect_size(height, width, params["max_size"], cover=cover)
        return tv_transforms.functional.resize(image, [height, width], antialias=False)

    return tv_transforms.Lambda(resize)


def _build_letter_box(params: dict[str, Any]) -> Any:
    def letter_box(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        height, width = aspect_size(height, width, params["height"])
        resized = tv_transforms.functional.resize(image, [height, width], antialias=False)
        return tv_transforms.functional.pad(
            resized, list(centered_padding(height, width, params["height"], params["width"])), fill=0
        )

    return tv_transforms.Lambda(letter_box)


def _build_pad_if_needed(params: dict[str, Any]) -> Any:
    def pad(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        right, bottom = max(params["min_width"] - width, 0), max(params["min_height"] - height, 0)
        return tv_transforms.functional.pad(image, [0, 0, right, bottom], fill=0) if right or bottom else image

    return tv_transforms.Lambda(pad)


def _build_random_scale(params: dict[str, Any]) -> Any:
    def resize(image: torch.Tensor) -> torch.Tensor:
        factor = random.uniform(*params["scale"])  # noqa: S311 - augmentation RNG.
        height, width = image.shape[-2:]
        return tv_transforms.functional.resize(image, [int(height * factor), int(width * factor)], antialias=False)

    return tv_transforms.Lambda(resize)


def _build_channel_swap(_params: dict[str, Any]) -> Any:
    return tv_transforms.Lambda(lambda image: tv_transforms.functional.permute_channels(image, [2, 1, 0]))


def _build_gaussian_noise(params: dict[str, Any]) -> Any:
    return tv_transforms.GaussianNoise(mean=params["mean"], sigma=params["std"], clip=True)


def _build_random_gamma(params: dict[str, Any]) -> Any:
    return tv_transforms.Lambda(lambda image: tv_transforms.functional.adjust_gamma(image, params["gamma"] / 100))


def _build_hue(params: dict[str, Any]) -> Any:
    limit = params["hue"] / 180
    return tv_transforms.ColorJitter(hue=(-limit, limit))


def _build_transpose(_params: dict[str, Any]) -> Any:
    return tv_transforms.Lambda(lambda image: image.transpose(-2, -1))


def _build_random_sized_crop(params: dict[str, Any]) -> Any:
    area = (params["crop_size"] / params["input_size"]) ** 2
    return tv_transforms.RandomResizedCrop(params["size"], scale=(area, area), ratio=(1.0, 1.0), antialias=False)


def _build_coarse_dropout(params: dict[str, Any]) -> Any:
    area = (params["hole_size"] / params["input_size"]) ** 2
    return tv_transforms.RandomErasing(scale=(area, area), ratio=(1.0, 1.0), value=0, p=1)


def _build_shift_scale_rotate(params: dict[str, Any]) -> Any:
    def warp(image: torch.Tensor) -> torch.Tensor:
        dtype = image.dtype
        if image.is_floating_point():
            image = image.float()
        return tv_transforms.functional.affine(
            image,
            angle=-params["angle"],
            translate=params["shift"],
            scale=params["scale"],
            shear=[0.0, 0.0],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            fill=None,
        ).to(dtype)

    return tv_transforms.Lambda(warp)


def _build_pad(params: dict[str, Any]) -> Any | None:
    return tv_transforms.Pad(padding=params["padding"], fill=params["fill"], padding_mode=params["border_mode"])


def _build_rotate(params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomRotation(
        degrees=params["angle_range"],
        interpolation=tv_transforms.InterpolationMode.BILINEAR
        if params["interpolation"] == "bilinear"
        else tv_transforms.InterpolationMode.NEAREST,
        fill=params["fill"],
    )


def _build_affine(params: dict[str, Any]) -> Any | None:
    # TV translate is a fraction of image size; spec shift is in pixels for reference_size
    ref = float(params.get("reference_size", 512))
    translate_fraction = [x / ref for x in params["shift"]]
    return tv_transforms.RandomAffine(
        degrees=params["angle"],
        translate=translate_fraction,
        scale=(params["scale"], params["scale"]),
        shear=params["shear"],
        interpolation=tv_transforms.InterpolationMode.BILINEAR
        if params["interpolation"] == "bilinear"
        else tv_transforms.InterpolationMode.NEAREST,
    )


class _FixedXShear(torch.nn.Module):
    def __init__(self, degrees: float) -> None:
        super().__init__()
        self._degrees = degrees

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        dtype = image.dtype
        if image.is_floating_point():
            image = image.float()
        return tv_transforms.functional.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[-self._degrees, 0.0],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            fill=None,
        ).to(dtype)


def _build_shear(params: dict[str, Any]) -> Any | None:
    return _FixedXShear(float(params["shear"]))


def _build_perspective(params: dict[str, Any]) -> Any | None:
    maximum_distortion = params["scale"][1]
    return tv_transforms.RandomPerspective(
        distortion_scale=maximum_distortion,
        interpolation=tv_transforms.InterpolationMode.BILINEAR
        if params["interpolation"] == "bilinear"
        else tv_transforms.InterpolationMode.NEAREST,
        fill=params["fill"],
        p=1,
    )


def _build_elastic(params: dict[str, Any]) -> Any | None:
    return tv_transforms.ElasticTransform(
        alpha=params["torchvision_alpha"],
        sigma=params["sigma"],
        interpolation=tv_transforms.InterpolationMode.BILINEAR
        if params["interpolation"] == "bilinear"
        else tv_transforms.InterpolationMode.NEAREST,
    )


def _build_color_jitter(params: dict[str, Any]) -> Any | None:
    return tv_transforms.ColorJitter(
        brightness=params["brightness"],
        contrast=params["contrast"],
        saturation=params["saturation"],
        hue=params["hue"],
    )


def _build_color_jiggle(params: dict[str, Any]) -> Any | None:
    return tv_transforms.ColorJitter(
        brightness=params["brightness"],
        contrast=params["contrast"],
        saturation=params["saturation"],
        hue=params["hue"],
    )


def _build_channel_shuffle(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomChannelPermutation()


def _build_grayscale(params: dict[str, Any]) -> Any | None:
    return tv_transforms.Grayscale(
        num_output_channels=params["num_output_channels"],
    )


def _build_gaussian_blur(params: dict[str, Any]) -> Any | None:
    return tv_transforms.GaussianBlur(kernel_size=params["kernel_size"], sigma=(params["sigma"], params["sigma"]))


def _build_invert(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomInvert(p=1)


def _build_posterize(params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomPosterize(bits=params["bits"], p=1)


def _build_solarize(params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomSolarize(threshold=params["threshold"], p=1)


def _build_sharpen(_params: dict[str, Any]) -> Any | None:
    # sharpness_factor=2.0 corresponds to maximum sharpening (1.0=original, 0.0=blurred)
    return tv_transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1)


def _build_auto_contrast(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomAutocontrast(p=1)


def _build_equalize(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomEqualize(p=1)


def _build_normalize(_params: dict[str, Any]) -> Any | None:
    # Normalization is a CUDA batch stage, shared by all RGB paths.
    return None


def _build_erasing(params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomErasing(
        scale=params["scale"],
        ratio=params["ratio"],
        value=params["fill"],
        p=1,
    )


def _build_jpeg_compression(params: dict[str, Any]) -> Any | None:
    return tv_transforms.JPEG(quality=params["quality"])


def _build_brightness(params: dict[str, Any]) -> Any | None:
    # Use scalar so TV picks factor in [1-limit, 1+limit], matching albumentations additive semantics
    limit = params["brightness_limit"]
    brightness_scalar = limit[0] if isinstance(limit, (list, tuple)) else limit
    return tv_transforms.ColorJitter(brightness=brightness_scalar, contrast=0.0, saturation=0.0, hue=0.0)


def _build_contrast(params: dict[str, Any]) -> Any | None:
    limit = params["contrast_limit"]
    contrast_scalar = limit[0] if isinstance(limit, (list, tuple)) else limit
    return tv_transforms.ColorJitter(brightness=0.0, contrast=contrast_scalar, saturation=0.0, hue=0.0)


def _build_photo_metric_distort(params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomPhotometricDistort(
        brightness=params["brightness_range"],
        contrast=params["contrast_range"],
        saturation=params["saturation_range"],
        hue=params["hue_range"],
        p=1,
    )


_TRANSFORM_BUILDERS = {
    "Resize": _build_resize,
    "RandomCrop224": _build_random_crop224,
    "CenterCrop": _build_center_crop,
    "Crop": _build_crop,
    "CropAndPad": _build_crop_and_pad,
    "LetterBox": _build_letter_box,
    "PadIfNeeded": _build_pad_if_needed,
    "LongestMaxSize": lambda params: _build_resize_by_side(params, cover=False),
    "SmallestMaxSize": lambda params: _build_resize_by_side(params, cover=True),
    "RandomScale": _build_random_scale,
    "RandomSizedCrop": _build_random_sized_crop,
    "CoarseDropout": _build_coarse_dropout,
    "ShiftScaleRotate": _build_shift_scale_rotate,
    "ChannelSwap": _build_channel_swap,
    "GaussianNoise": _build_gaussian_noise,
    "RandomGamma": _build_random_gamma,
    "Hue": _build_hue,
    "Transpose": _build_transpose,
    "RandomResizedCrop": _build_random_resized_crop,
    "HorizontalFlip": _build_horizontal_flip,
    "VerticalFlip": _build_vertical_flip,
    "RandomRotate90": _build_random_rotate90,
    "Pad": _build_pad,
    "Rotate": _build_rotate,
    "Affine": _build_affine,
    "Shear": _build_shear,
    "Perspective": _build_perspective,
    "Elastic": _build_elastic,
    "ColorJitter": _build_color_jitter,
    "ColorJiggle": _build_color_jiggle,
    "ChannelShuffle": _build_channel_shuffle,
    "Grayscale": _build_grayscale,
    "GaussianBlur": _build_gaussian_blur,
    "Invert": _build_invert,
    "Posterize": _build_posterize,
    "Solarize": _build_solarize,
    "Sharpen": _build_sharpen,
    "AutoContrast": _build_auto_contrast,
    "Equalize": _build_equalize,
    "Normalize": _build_normalize,
    "Erasing": _build_erasing,
    "JpegCompression": _build_jpeg_compression,
    "Brightness": _build_brightness,
    "Contrast": _build_contrast,
    "PhotoMetricDistort": _build_photo_metric_distort,
}
