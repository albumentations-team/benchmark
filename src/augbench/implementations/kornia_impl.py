"""Kornia transform factories for the active RGB recipe catalog."""

import random
from collections.abc import Callable
from functools import partial
from math import sqrt
from typing import Any

import kornia
import kornia.augmentation as kornia_augmentation
import torch

from augbench.implementations.geometry import (
    affine_axis_ranges,
    centered_padding,
    perspective_edges_squared,
)
from augbench.implementations.specs import TransformSpec, odd_kernel_sizes, parameter_range

LIBRARY = "kornia"

_TPS_POINTS_PER_AXIS = 2


class _TensorOperation(torch.nn.Module):
    def __init__(self, operation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self._operation = operation

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._operation(image)


class _PerImageJitter(torch.nn.Module):
    def __init__(self, jitter: torch.nn.Module) -> None:
        super().__init__()
        self.jitter = jitter

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (
            torch.cat([self.jitter(sample) for sample in image.split(1)]) if image.shape[0] > 1 else self.jitter(image)
        )


class _Morphological(torch.nn.Module):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self._operation = kornia.morphology.dilation if operation == "dilation" else kornia.morphology.erosion
        self.register_buffer("kernel", torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._operation(image, self.kernel.to(image), engine="convolution")


def _build_center_crop(params: dict[str, Any]) -> Any:
    height, width = params["height"], params["width"]
    crop = kornia.augmentation.CenterCrop(size=(height, width), cropping_mode="slice", p=1)

    def apply(image: torch.Tensor) -> torch.Tensor:
        in_height, in_width = image.shape[-2:]
        if in_height < height or in_width < width:
            return kornia.geometry.transform.center_crop(image, (height, width), mode="nearest")
        return crop(image)

    return _TensorOperation(apply)


def _build_crop(params: dict[str, Any]) -> Any:
    top, left = params["top"], params["left"]
    height, width = params["height"], params["width"]
    bottom, right = top + height - 1, left + width - 1
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))

    def crop(image: torch.Tensor) -> torch.Tensor:
        boxes = image.new_tensor([corners]).expand(image.shape[0], -1, -1)
        return kornia.geometry.transform.crop_by_indices(image, boxes, size=(height, width))

    return _TensorOperation(crop)


def _build_pad_if_needed(params: dict[str, Any]) -> Any:
    return kornia_augmentation.PadTo((params["min_height"], params["min_width"]), crop_if_exceeds=False)


def _build_random_scale(params: dict[str, Any]) -> Any:
    def resize(image: torch.Tensor) -> torch.Tensor:
        factor = random.uniform(*params["scale"])  # noqa: S311 - augmentation RNG.
        height, width = image.shape[-2:]
        size = (int(height * factor), int(width * factor))
        return kornia.geometry.transform.resize(image, size, align_corners=False, antialias=False)

    return _TensorOperation(resize)


def _build_random_sized_crop(params: dict[str, Any]) -> Any:
    area = (params["crop_size"] / params["input_size"]) ** 2
    return kornia_augmentation.RandomResizedCrop(
        tuple(params["size"]), scale=(area, area), ratio=(1.0, 1.0), cropping_mode="slice", align_corners=False, p=1
    )


def _build_coarse_dropout(params: dict[str, Any]) -> Any:
    area = (params["hole_size"] / params["input_size"]) ** 2
    return kornia_augmentation.RandomErasing(scale=(area, area), ratio=(1.0, 1.0), value=0, p=1)


def _build_colorize(params: dict[str, Any]) -> Any:
    anchors = list(zip(params["black"], params["white"], strict=True))
    palette = [tuple((black + (white - black) * value / 255) / 255 for black, white in anchors) for value in range(256)]
    colormap = kornia.color.ColorMap(palette, num_colors=256)
    return kornia.color.ApplyColorMap(colormap)


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image.unsqueeze(0)).squeeze(0)


def create_transform(spec: TransformSpec) -> Any | None:
    builder = _TRANSFORM_BUILDERS.get(spec.name)
    return builder(spec.params) if builder is not None else None


def _build_color_jitter(params: dict[str, Any]) -> Any | None:
    return _PerImageJitter(
        kornia_augmentation.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
            p=1,
            same_on_batch=False,
        )
    )


def _build_auto_contrast(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomAutoContrast(p=1)


def _sample_uniform(image: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    return torch.empty(image.shape[0], device=image.device, dtype=torch.float32).uniform_(*bounds)


def _build_sized_filter(
    sizes: range, apply: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor]
) -> torch.nn.Module:
    def blur(image: torch.Tensor) -> torch.Tensor:
        if image.shape[0] == 1:
            size = random.choice(sizes)  # noqa: S311 - augmentation RNG.
            return apply(image, (size, size))
        choices = torch.randint(len(sizes), (image.shape[0],), device="cpu")
        result = torch.empty_like(image)
        for index, size in enumerate(sizes):
            indices = (choices == index).nonzero().flatten()
            if indices.numel():
                indices = indices.to(image.device)
                group = image.index_select(0, indices)
                result.index_copy_(0, indices, apply(group, (size, size)))
        return result

    return _TensorOperation(blur)


def _build_blur(params: dict[str, Any]) -> Any:
    return _build_sized_filter(
        odd_kernel_sizes(params["kernel_size"]), partial(kornia.filters.box_blur, border_type=params["border_mode"])
    )


def _build_brightness(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["brightness_limit"])
    return _TensorOperation(
        lambda image: kornia.enhance.adjust_brightness_accumulative(image, 1 + _sample_uniform(image, bounds))
    )


def _build_channel_dropout(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomChannelDropout(p=1)


def _build_channel_shuffle(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomChannelShuffle(p=1)


def _build_channel_swap(params: dict[str, Any]) -> Any:
    if tuple(params["channel_order"]) != (2, 1, 0):
        raise ValueError("Kornia ChannelSwap supports only RGB <-> BGR")
    return kornia.color.RgbToBgr()


def _build_clahe(params: dict[str, Any]) -> Any:
    return kornia_augmentation.RandomClahe(
        clip_limit=tuple(params["kornia_clip_range"]), grid_size=tuple(params["tile_grid_size"]), p=1
    )


def _build_contrast(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["contrast_limit"])
    return _TensorOperation(
        lambda image: kornia.enhance.adjust_contrast_with_mean_subtraction(image, 1 + _sample_uniform(image, bounds))
    )


def _build_equalize(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomEqualize(p=1)


def _build_random_gamma(params: dict[str, Any]) -> Any | None:
    low, high = parameter_range(params["gamma"])
    return kornia_augmentation.RandomGamma(
        gamma=(low / 100, high / 100),
        p=1,
    )


def _build_gaussian_blur(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomGaussianBlur(
        kernel_size=tuple(params["kernel_size"]),
        sigma=parameter_range(params["sigma"]),
        p=1,
    )


def _build_linear_illumination(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomLinearIllumination(
        gain=tuple(params["gain"]),
        p=1,
    )


def _build_corner_illumination(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomLinearCornerIllumination(
        gain=tuple(params["kornia_gain"]),
        p=1,
    )


def _build_gaussian_illumination(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomGaussianIllumination(
        gain=tuple(params["kornia_gain"]),
        p=1,
    )


def _build_gaussian_noise(params: dict[str, Any]) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        kornia_augmentation.RandomGaussianNoise(mean=params["mean"], std=params["std"], p=1),
        _TensorOperation(partial(torch.clamp_, min=0, max=1)),
    )


def _build_grayscale(params: dict[str, Any]) -> Any | None:
    if params["num_output_channels"] == 1:
        return kornia.color.RgbToGrayscale()
    return kornia_augmentation.RandomGrayscale(p=1)


def _build_hue(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomHue(
        hue=tuple(angle / 360 for angle in params["hue_degrees"]),
        p=1,
    )


def _build_invert(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomInvert(p=1)


def _build_jpeg_compression(params: dict[str, Any]) -> Any | None:
    low, high = params["quality_range"]
    (left, start), (middle, center), (right, end) = params["kornia_quality_anchors"]
    codec = kornia.enhance.JPEGCodecDifferentiable()

    def compress(image: torch.Tensor) -> torch.Tensor:
        quality = torch.randint(low, high + 1, (image.shape[0],), device=image.device)
        calibrated = torch.where(
            quality < middle,
            start + (quality - left) * ((center - start) / (middle - left)),
            center + (quality - middle) * ((end - center) / (right - middle)),
        )
        return codec(image, calibrated.to(image.dtype))

    return _TensorOperation(compress)


def _build_median_blur(params: dict[str, Any]) -> Any:
    return _build_sized_filter(
        odd_kernel_sizes(params["blur_limit"]), partial(kornia.filters.median_blur, border_type="replicate")
    )


def _build_motion_blur(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomMotionBlur(
        kernel_size=params["kernel_size"],
        angle=tuple(params["angle_range"]),
        direction=tuple(params["direction_range"]),
        p=1,
    )


def _build_plankian_jitter(params: dict[str, Any]) -> Any | None:
    # AX splits its blackbody draws evenly around 6000 K. Kornia's table has six warmer rows and nineteen cooler rows.
    indices = list(range(6)) * 3 + list(range(6, 25))
    return kornia_augmentation.RandomPlanckianJitter(
        mode=params["mode"],
        select_from=indices,
        p=1,
    )


def _build_plasma_brightness(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlasmaBrightness(
        roughness=(params["roughness"], params["roughness"]),
        intensity=tuple(params["kornia_intensity"]),
        p=1,
    )


def _build_plasma_contrast(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlasmaContrast(
        roughness=(params["roughness"], params["roughness"]),
        p=1,
    )


def _build_plasma_shadow(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlasmaShadow(
        roughness=(params["roughness"], params["roughness"]),
        shade_intensity=tuple(params["kornia_intensity"]),
        shade_quantity=tuple(params["kornia_quantity"]),
        p=1,
    )


def _build_rain(params: dict[str, Any]) -> Any:
    count = params["kornia_drop_count"]
    rows = params["input_height"] // 8 + 1
    return kornia_augmentation.RandomRain(
        number_of_drops=(count, count), drop_height=(rows, rows), drop_width=(-11, 11), p=1
    )


def _build_rgbshift(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomRGBShift(
        r_shift_limit=params["pixel_shift"] / 255.0,
        g_shift_limit=params["pixel_shift"] / 255.0,
        b_shift_limit=params["pixel_shift"] / 255.0,
        p=1,
    )


def _build_salt_and_pepper(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomSaltAndPepperNoise(
        amount=tuple(params["amount"]),
        salt_vs_pepper=tuple(params["salt_vs_pepper"]),
        p=1,
    )


def _build_saturation(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["saturation_factor"])
    return _TensorOperation(
        lambda image: kornia.enhance.adjust_saturation_with_gray_subtraction(image, 1 + _sample_uniform(image, bounds))
    )


def _build_sharpen(params: dict[str, Any]) -> Any:
    return kornia_augmentation.RandomSharpness(sharpness=tuple(1 + 13 * alpha for alpha in params["alpha"]), p=1)


def _build_snow(params: dict[str, Any]) -> Any | None:
    low, high = params["snow_point_range"]
    return kornia_augmentation.RandomSnow(
        snow_coefficient=(low / 2 + 1 / 3, high / 2 + 1 / 3),
        brightness=(float(params["brightness"]),) * 2,
        p=1,
    )


def _build_solarize(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["threshold"])
    return _TensorOperation(lambda image: kornia.enhance.solarize(image, thresholds=_sample_uniform(image, bounds)))


def _build_affine(params: dict[str, Any]) -> Any:
    shift_x, shift_y = affine_axis_ranges(params["shift"])
    shear = affine_axis_ranges(params["shear"]) if "shear" in params else None
    input_size = params["input_size"]
    return kornia_augmentation.RandomAffine(
        degrees=parameter_range(params["angle"]),
        translate=(max(map(abs, shift_x)) / input_size, max(map(abs, shift_y)) / input_size),
        scale=parameter_range(params["scale"]),
        shear=(*shear[0], *shear[1]) if shear is not None else None,
        p=1,
    )


def _build_random_crop224(params: dict[str, Any]) -> Any | None:
    return _CenteredRandomCrop((params["height"], params["width"]), p=1)


class _CenteredRandomCrop(kornia_augmentation.RandomCrop):
    def compute_padding(self, shape: tuple[int, ...], flags: dict[str, Any] | None = None) -> list[int]:
        height, width = shape[-2:]
        crop_height, crop_width = (self.flags if flags is None else flags)["size"]
        left, top, right, bottom = centered_padding(height, width, max(crop_height, height), max(crop_width, width))
        return [left, right, top, bottom]


class _ElasticTransform(kornia_augmentation.RandomElasticTransform):
    def __init__(self, amplitude: tuple[float, float], sigma: float) -> None:
        super().__init__(p=1, sigma=(sigma, sigma), alpha=(amplitude[1], amplitude[1]))
        self._strength = (amplitude[0] / amplitude[1], 1.0) if amplitude[1] else (1.0, 1.0)

    def generate_parameters(self, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        params = super().generate_parameters(shape)
        noise = params["noise"]
        multiplier = _sample_uniform(noise, self._strength).reshape(-1, 1, 1, 1)
        noise.mul_(multiplier)
        return params


def _build_elastic(params: dict[str, Any]) -> Any:
    return _ElasticTransform(parameter_range(params["kornia_alpha"]), params["sigma"])


def _build_erasing(params: dict[str, Any]) -> Any | None:
    # Pass float value to avoid Kornia's internal Long-dtype sampling bug
    return kornia_augmentation.RandomErasing(
        p=1,
        scale=tuple(params["scale"]),
        ratio=tuple(params["ratio"]),
        value=float(params["fill"]),
    )


def _build_optical_distortion(params: dict[str, Any]) -> Any:
    center = torch.zeros(2)
    return kornia_augmentation.RandomFisheye(
        center_x=center, center_y=center, gamma=torch.tensor(params["kornia_gamma"]), p=1
    )


def _build_horizontal_flip(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomHorizontalFlip(p=1)


def _build_perspective(params: dict[str, Any]) -> Any:
    return _InwardPerspective(distortion_scale=params["uniform_corner_scale"], p=1)


class _InwardPerspective(kornia_augmentation.RandomPerspective):
    def generate_parameters(self, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        params = super().generate_parameters(shape)
        height, width = shape[-2:]
        # Use the sampled inner quadrilateral as the source, as AX does.
        source = params["end_points"]
        corners = tuple((point[:, 0], point[:, 1]) for point in source.unbind(1))
        widths, heights = perspective_edges_squared(corners)
        out_width = width**2 / torch.maximum(*widths).sqrt().floor()
        out_height = height**2 / torch.maximum(*heights).sqrt().floor()
        scale = torch.stack((out_width / (width - 1), out_height / (height - 1)), dim=-1)
        params["end_points"] = params["start_points"] * scale[:, None, :]
        params["start_points"] = source
        return params


def _build_random_resized_crop(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomResizedCrop(
        size=tuple(params["size"]),
        scale=tuple(params["scale"]),
        ratio=tuple(params["ratio"]),
        resample=params["interpolation"],
        align_corners=False,
        cropping_mode="slice",
        p=1,
    )


def _build_random_rotate90(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomRotation90(times=tuple(params["times"]), p=1)


def _build_random_jigsaw(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomJigsaw(grid=tuple(params["grid"]), ensure_perm=False, p=1)


def _build_rotate(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomRotation(
        degrees=tuple(params["angle_range"]),
        resample=params["interpolation"],
        p=1,
    )


def _build_shear(params: dict[str, Any]) -> Any:
    return kornia_augmentation.RandomAffine(degrees=0, shear=parameter_range(params["shear"]), p=1)


def _build_thin_plate_spline(params: dict[str, Any]) -> Any:
    if params["num_control_points"] != _TPS_POINTS_PER_AXIS:
        raise ValueError("Kornia ThinPlateSpline requires four corners and a center point")
    # Match AX's anchor-offset standard deviation with Kornia's uniform law.
    return kornia_augmentation.RandomThinPlateSpline(scale=sqrt(3) * params["distortion_scale"] / 5, p=1)


def _build_vertical_flip(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomVerticalFlip(p=1)


def _build_resize(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.Resize(
        size=(params["target_size"], params["target_size"]),
        resample=params["interpolation"],
        align_corners=False,
        antialias=params.get("antialias", False),
        p=1,
    )


def _build_posterize(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPosterize(
        bits=parameter_range(params["bits"]),
        p=1,
    )


def _build_longest_max_size(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.LongestMaxSize(
        max_size=params["max_size"],
        resample=params["interpolation"],
        p=1,
    )


def _build_smallest_max_size(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.SmallestMaxSize(
        max_size=params["max_size"],
        resample=params["interpolation"],
        p=1,
    )


_TRANSFORM_BUILDERS = {
    "ColorJitter": _build_color_jitter,
    "ColorJiggle": _build_color_jitter,
    "AutoContrast": _build_auto_contrast,
    "Blur": _build_blur,
    "Brightness": _build_brightness,
    "ChannelDropout": _build_channel_dropout,
    "ChannelShuffle": _build_channel_shuffle,
    "CLAHE": _build_clahe,
    "Contrast": _build_contrast,
    "Equalize": _build_equalize,
    "RandomGamma": _build_random_gamma,
    "GaussianBlur": _build_gaussian_blur,
    "LinearIllumination": _build_linear_illumination,
    "CornerIllumination": _build_corner_illumination,
    "GaussianIllumination": _build_gaussian_illumination,
    "GaussianNoise": _build_gaussian_noise,
    "Grayscale": _build_grayscale,
    "Hue": _build_hue,
    "Invert": _build_invert,
    "JpegCompression": _build_jpeg_compression,
    "MedianBlur": _build_median_blur,
    "MotionBlur": _build_motion_blur,
    "PlankianJitter": _build_plankian_jitter,
    "PlasmaBrightness": _build_plasma_brightness,
    "PlasmaContrast": _build_plasma_contrast,
    "PlasmaShadow": _build_plasma_shadow,
    "Rain": _build_rain,
    "RGBShift": _build_rgbshift,
    "SaltAndPepper": _build_salt_and_pepper,
    "Saturation": _build_saturation,
    "Sharpen": _build_sharpen,
    "Snow": _build_snow,
    "Solarize": _build_solarize,
    "Affine": _build_affine,
    "RandomCrop224": _build_random_crop224,
    "CenterCrop": _build_center_crop,
    "Crop": _build_crop,
    "PadIfNeeded": _build_pad_if_needed,
    "RandomScale": _build_random_scale,
    "RandomSizedCrop": _build_random_sized_crop,
    "CoarseDropout": _build_coarse_dropout,
    "Colorize": _build_colorize,
    "ShiftScaleRotate": _build_affine,
    "ChannelSwap": _build_channel_swap,
    "Morphological": lambda params: _Morphological(params["operation"]),
    "Elastic": _build_elastic,
    "Erasing": _build_erasing,
    "OpticalDistortion": _build_optical_distortion,
    "HorizontalFlip": _build_horizontal_flip,
    "Perspective": _build_perspective,
    "RandomResizedCrop": _build_random_resized_crop,
    "RandomRotate90": _build_random_rotate90,
    "RandomJigsaw": _build_random_jigsaw,
    "Rotate": _build_rotate,
    "Shear": _build_shear,
    "ThinPlateSpline": _build_thin_plate_spline,
    "VerticalFlip": _build_vertical_flip,
    "Resize": _build_resize,
    "Posterize": _build_posterize,
    "LongestMaxSize": _build_longest_max_size,
    "SmallestMaxSize": _build_smallest_max_size,
}
