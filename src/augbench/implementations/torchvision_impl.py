"""TorchVision transform factories for the active RGB recipe catalog."""

import random
from math import sqrt
from typing import Any

import torch
import torchvision.transforms.v2 as tv_transforms

from augbench.implementations.geometry import (
    affine_axis_ranges,
    aspect_size,
    centered_padding,
    perspective_edges_squared,
    sample_perspective_source,
)
from augbench.implementations.specs import TransformSpec, parameter_range

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
    height, width = params["height"], params["width"]
    crop = tv_transforms.RandomCrop(size=(height, width))

    def apply(image: torch.Tensor) -> torch.Tensor:
        image_height, image_width = image.shape[-2:]
        padding = centered_padding(image_height, image_width, max(height, image_height), max(width, image_width))
        if any(padding):
            image = tv_transforms.functional.pad(image, list(padding))
        return crop(image)

    return tv_transforms.Lambda(apply)


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
    def rotate(image: torch.Tensor) -> torch.Tensor:
        turns = random.randint(*params["times"])  # noqa: S311 - augmentation RNG.
        return tv_transforms.functional.rotate(image, 90 * turns)

    return tv_transforms.Lambda(rotate)


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


def _build_channel_swap(params: dict[str, Any]) -> Any:
    order = list(params["channel_order"])
    return tv_transforms.Lambda(lambda image: tv_transforms.functional.permute_channels(image, order))


def _build_gaussian_noise(params: dict[str, Any]) -> Any:
    return tv_transforms.GaussianNoise(mean=params["mean"], sigma=params["std"], clip=True)


def _build_random_gamma(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["gamma"])

    def gamma(image: torch.Tensor) -> torch.Tensor:
        exponent = random.uniform(*bounds) / 100  # noqa: S311 - augmentation RNG.
        return tv_transforms.functional.adjust_gamma(image, exponent)

    return tv_transforms.Lambda(gamma)


def _build_hue(params: dict[str, Any]) -> Any:
    return tv_transforms.ColorJitter(hue=tuple(angle / 360 for angle in params["hue_degrees"]))


def _build_random_sized_crop(params: dict[str, Any]) -> Any:
    area = (params["crop_size"] / params["input_size"]) ** 2
    return tv_transforms.RandomResizedCrop(params["size"], scale=(area, area), ratio=(1.0, 1.0), antialias=False)


def _build_coarse_dropout(params: dict[str, Any]) -> Any:
    area = (params["hole_size"] / params["input_size"]) ** 2
    return tv_transforms.RandomErasing(scale=(area, area), ratio=(1.0, 1.0), value=0, p=1)


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


def _build_affine(params: dict[str, Any]) -> Any:
    shift_x, shift_y = affine_axis_ranges(params["shift"])
    shear = affine_axis_ranges(params["shear"]) if "shear" in params else None
    input_size = params["input_size"]
    return tv_transforms.RandomAffine(
        degrees=parameter_range(params["angle"]),
        translate=(max(map(abs, shift_x)) / input_size, max(map(abs, shift_y)) / input_size),
        scale=parameter_range(params["scale"]),
        shear=(*shear[0], *shear[1]) if shear is not None else None,
        interpolation=tv_transforms.InterpolationMode.BILINEAR,
        fill=None,
    )


def _build_shear(params: dict[str, Any]) -> Any:
    return tv_transforms.RandomAffine(
        degrees=0,
        shear=parameter_range(params["shear"]),
        interpolation=tv_transforms.InterpolationMode.BILINEAR,
        fill=None,
    )


def _build_perspective(params: dict[str, Any]) -> Any:
    scale = tuple(params["scale"])

    def warp(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        corners = sample_perspective_source(height, width, scale)
        widths, heights = perspective_edges_squared(corners)
        out_width = round(width**2 / int(sqrt(max(widths))))
        out_height = round(height**2 / int(sqrt(max(heights))))
        source = [[round(x), round(y)] for x, y in corners]
        destination = [[0, 0], [out_width, 0], [out_width, out_height], [0, out_height]]
        return tv_transforms.functional.perspective(image, source, destination)

    return tv_transforms.Lambda(warp)


class _ElasticTransform(tv_transforms.ElasticTransform):
    def __init__(
        self, amplitude: tuple[float, float], sigma: float, interpolation: tv_transforms.InterpolationMode
    ) -> None:
        super().__init__(alpha=amplitude[1], sigma=sigma, interpolation=interpolation)
        self._strength = (amplitude[0] / amplitude[1], 1.0) if amplitude[1] else (1.0, 1.0)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        params = super().make_params(flat_inputs)
        multiplier = random.uniform(*self._strength)  # noqa: S311 - augmentation RNG.
        params["displacement"].mul_(multiplier)
        return params


def _build_elastic(params: dict[str, Any]) -> Any:
    interpolation = (
        tv_transforms.InterpolationMode.BILINEAR
        if params["interpolation"] == "bilinear"
        else tv_transforms.InterpolationMode.NEAREST
    )
    return _ElasticTransform(parameter_range(params["torchvision_alpha"]), params["sigma"], interpolation)


def _build_color_jitter(params: dict[str, Any]) -> Any | None:
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
    return tv_transforms.GaussianBlur(kernel_size=params["kernel_size"], sigma=parameter_range(params["sigma"]))


def _build_invert(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomInvert(p=1)


def _build_posterize(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["bits"])

    def posterize(image: torch.Tensor) -> torch.Tensor:
        bits = random.randint(*bounds)  # noqa: S311 - augmentation RNG.
        return tv_transforms.functional.posterize(image, bits)

    return tv_transforms.Lambda(posterize)


def _build_solarize(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["threshold"])

    def solarize(image: torch.Tensor) -> torch.Tensor:
        threshold = random.uniform(*bounds)  # noqa: S311 - augmentation RNG.
        return tv_transforms.functional.solarize(image, threshold if image.is_floating_point() else 255 * threshold)

    return tv_transforms.Lambda(solarize)


def _build_sharpen(params: dict[str, Any]) -> Any:
    def sharpen(image: torch.Tensor) -> torch.Tensor:
        alpha = random.uniform(*params["alpha"])  # noqa: S311 - augmentation RNG.
        return tv_transforms.functional.adjust_sharpness(image, 1 + 13 * alpha)

    return tv_transforms.Lambda(sharpen)


def _build_auto_contrast(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomAutocontrast(p=1)


def _build_equalize(_params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomEqualize(p=1)


def _build_erasing(params: dict[str, Any]) -> Any | None:
    return tv_transforms.RandomErasing(
        scale=params["scale"],
        ratio=params["ratio"],
        value=params["fill"],
        p=1,
    )


def _build_jpeg_compression(params: dict[str, Any]) -> Any | None:
    return tv_transforms.JPEG(quality=tuple(params["quality_range"]))


def _build_brightness(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["brightness_limit"])

    def adjust(image: torch.Tensor) -> torch.Tensor:
        factor = 1 + random.uniform(*bounds)  # noqa: S311 - augmentation RNG.
        return tv_transforms.functional.adjust_brightness(image, factor)

    return tv_transforms.Lambda(adjust)


def _build_contrast(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["contrast_limit"])

    def adjust(image: torch.Tensor) -> torch.Tensor:
        factor = 1 + random.uniform(*bounds)  # noqa: S311 - augmentation RNG.
        return tv_transforms.functional.adjust_contrast(image, factor)

    return tv_transforms.Lambda(adjust)


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
    "PadIfNeeded": _build_pad_if_needed,
    "LongestMaxSize": lambda params: _build_resize_by_side(params, cover=False),
    "SmallestMaxSize": lambda params: _build_resize_by_side(params, cover=True),
    "RandomScale": _build_random_scale,
    "RandomSizedCrop": _build_random_sized_crop,
    "CoarseDropout": _build_coarse_dropout,
    "ShiftScaleRotate": _build_affine,
    "ChannelSwap": _build_channel_swap,
    "GaussianNoise": _build_gaussian_noise,
    "RandomGamma": _build_random_gamma,
    "Hue": _build_hue,
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
    "ColorJiggle": _build_color_jitter,
    "ChannelShuffle": _build_channel_shuffle,
    "Grayscale": _build_grayscale,
    "GaussianBlur": _build_gaussian_blur,
    "Invert": _build_invert,
    "Posterize": _build_posterize,
    "Solarize": _build_solarize,
    "Sharpen": _build_sharpen,
    "AutoContrast": _build_auto_contrast,
    "Equalize": _build_equalize,
    "Erasing": _build_erasing,
    "JpegCompression": _build_jpeg_compression,
    "Brightness": _build_brightness,
    "Contrast": _build_contrast,
    "PhotoMetricDistort": _build_photo_metric_distort,
}
