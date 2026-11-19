"""Kornia transform factories for the active RGB recipe catalog."""

import random
from collections.abc import Callable
from functools import partial
from math import isqrt, pi
from typing import Any

import kornia
import kornia.augmentation as kornia_augmentation
import torch
import torch.nn.functional as torch_functional

from augbench.implementations.geometry import (
    affine_axis_ranges,
    aspect_size,
    centered_inverse_affine,
    centered_inverse_perspective,
    centered_padding,
    emboss_kernel,
)
from augbench.implementations.specs import TransformSpec, channel_orders, odd_kernel_sizes, parameter_range

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
        left, top, right, bottom = centered_padding(in_height, in_width, max(in_height, height), max(in_width, width))
        if left or top or right or bottom:
            image = torch_functional.pad(image, (left, right, top, bottom))
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


def _build_crop_and_pad(params: dict[str, Any]) -> Any:
    top, right, bottom, left = params["sides"]
    return _TensorOperation(lambda image: torch_functional.pad(image, (left, right, top, bottom)))


def _build_pad(params: dict[str, Any]) -> Any:
    return _TensorOperation(lambda image: torch_functional.pad(image, (params["padding"],) * 4))


def _build_pad_if_needed(params: dict[str, Any]) -> Any:
    def pad(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        right, bottom = max(params["min_width"] - width, 0), max(params["min_height"] - height, 0)
        return torch_functional.pad(image, (0, right, 0, bottom)) if right or bottom else image

    return _TensorOperation(pad)


def _build_letter_box(params: dict[str, Any]) -> Any:
    def letter_box(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        height, width = aspect_size(height, width, params["height"])
        resized = kornia.geometry.transform.resize(image, (height, width), align_corners=False, antialias=False)
        left, top, right, bottom = centered_padding(height, width, params["height"], params["width"])
        return torch_functional.pad(resized, (left, right, top, bottom))

    return _TensorOperation(letter_box)


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
    return torch.nn.Sequential(kornia.color.RgbToGrayscale(), kornia.color.ApplyColorMap(colormap))


class _Emboss(torch.nn.Module):
    def __init__(self, alpha: tuple[float, float], strength: tuple[float, float]) -> None:
        super().__init__()
        self._alpha = alpha
        self._strength = strength

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        alpha = _sample_uniform(image, self._alpha)
        strength = _sample_uniform(image, self._strength)
        kernel = torch.stack(emboss_kernel(alpha, strength), dim=-1).reshape(-1, 3, 3)
        return kornia.filters.filter2d(image, kernel.to(dtype=image.dtype)).clamp_(0, 1)


class _RandomJigsawWithPad(torch.nn.Module):
    def __init__(self, grid: tuple[int, int]) -> None:
        super().__init__()
        self.grid = grid
        self.jigsaw = kornia_augmentation.RandomJigsaw(grid=grid, p=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        grid_h, grid_w = self.grid
        pad_height = (-height) % grid_h
        pad_width = (-width) % grid_w
        if pad_height or pad_width:
            image = torch_functional.pad(image, (0, pad_width, 0, pad_height))
        return self.jigsaw(image)[..., :height, :width]


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
    orders = channel_orders(params)
    if len(orders) == 1:
        return _TensorOperation(lambda image: image[:, orders[0], ...])
    table = torch.tensor(orders, dtype=torch.long)

    def swap(image: torch.Tensor) -> torch.Tensor:
        choices = torch.randint(len(orders), (image.shape[0],), device=image.device)
        indices = table.to(image.device)[choices, :, None, None]
        return image.gather(dim=1, index=indices.expand_as(image))

    return _TensorOperation(swap)


def _build_clahe(params: dict[str, Any]) -> Any | None:
    levels = params["clip_limits"]
    return _DiscreteClahe(
        clip_limit=(float(levels[0]), float(levels[-1] + 1)), grid_size=tuple(params["tile_grid_size"]), p=1
    )


class _DiscreteClahe(kornia_augmentation.RandomClahe):
    def forward_parameters(self, batch_shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        parameters = super().forward_parameters(batch_shape)
        parameters["clip_limit_factor"].floor_()
        return parameters


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
        gain=tuple(params["gain"]),
        p=1,
    )


def _build_gaussian_illumination(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomGaussianIllumination(
        gain=tuple(params["gain"]),
        p=1,
    )


def _build_gaussian_noise(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomGaussianNoise(
        mean=params["mean"],
        std=params["std"],
        p=1,
    )


def _build_grayscale(_params: dict[str, Any]) -> Any | None:
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
    codec = kornia.enhance.JPEGCodecDifferentiable()

    def compress(image: torch.Tensor) -> torch.Tensor:
        quality = torch.randint(low, high + 1, (image.shape[0],), device=image.device).to(image.dtype)
        return codec(image, quality)

    return _TensorOperation(compress)


def _build_median_blur(params: dict[str, Any]) -> Any:
    return _build_sized_filter(odd_kernel_sizes(params["blur_limit"]), kornia.filters.median_blur)


def _build_motion_blur(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomMotionBlur(
        kernel_size=params["kernel_size"],
        angle=tuple(params["angle_range"]),
        direction=tuple(params["direction_range"]),
        p=1,
    )


def _build_plankian_jitter(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlanckianJitter(
        mode=params["mode"],
        p=1,
    )


def _build_plasma_brightness(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlasmaBrightness(
        roughness=(params["roughness"], params["roughness"]),
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
        p=1,
    )


class _Rain(kornia_augmentation.RandomRain):
    def __init__(self, height: int, thickness: int, brightness: float) -> None:
        count = height // 3
        rows = height // 8 + 1
        super().__init__(number_of_drops=(count, count), drop_height=(rows, rows), drop_width=(0, 19), p=1)
        self._brightness = brightness
        # OpenCV's thick lines use a disk around the centerline. Keep only maximal rectangles in the disk.
        radius = 0 if thickness == 1 else (thickness + 1) // 2
        half_widths = {isqrt(radius**2 - y**2): y for y in range(radius + 1)}
        self._footprint = tuple((2 * y + 1, 2 * x + 1) for x, y in half_widths.items())

    def generate_parameters(self, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        params = super().generate_parameters(shape)
        # AX's continuous slant in [-10, 10] lands on integer offsets -10 through +9.
        slant = params["drop_width_factor"] - 10
        params["drop_width_factor"] = slant + slant.sign()
        return params

    def apply_transform(
        self,
        image: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        centerlines = super().apply_transform(torch.zeros_like(image[:, :1]), params, flags, transform)
        overlay = torch.zeros_like(centerlines)
        for h, w in self._footprint:
            thick = torch_functional.max_pool2d(centerlines, (h, w), stride=1, padding=(h // 2, w // 2))
            torch.maximum(overlay, thick, out=overlay)
        softened = kornia.filters.box_blur(overlay, 7)
        result = (image + softened).clamp_(0, 1)
        if self._brightness != 1:
            result.mul_(self._brightness)
        return result


def _build_rain(params: dict[str, Any]) -> Any:
    if "input_height" in params:
        return _Rain(params["input_height"], params["drop_width"], params["brightness_coefficient"])
    return kornia_augmentation.RandomRain(
        drop_width=(params["drop_width"], params["drop_width"]),
        drop_height=(params["drop_height"], params["drop_height"]),
        p=1,
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
    return _ReflectSharpness(tuple(1 + 13 * alpha for alpha in params["alpha"]))


class _ReflectSharpness(torch.nn.Module):
    def __init__(self, factor: tuple[float, float]) -> None:
        super().__init__()
        self.sharpen = kornia_augmentation.RandomSharpness(sharpness=factor, p=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        padded = torch_functional.pad(image, (1, 1, 1, 1), mode="reflect")
        return self.sharpen(padded)[..., 1:-1, 1:-1]


def _build_snow(params: dict[str, Any]) -> Any | None:
    # Pass explicit float brightness to avoid Kornia's internal Long-dtype sampling bug
    return kornia_augmentation.RandomSnow(
        snow_coefficient=tuple(params["snow_point_range"]),
        brightness=(params["brightness"], params["brightness"]),
        p=1,
    )


def _build_solarize(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["threshold"])
    return _TensorOperation(lambda image: kornia.enhance.solarize(image, thresholds=_sample_uniform(image, bounds)))


def _build_affine(params: dict[str, Any]) -> Any:
    angles = parameter_range(params["angle"])
    scales = parameter_range(params["scale"])
    (min_x, max_x), (min_y, max_y) = affine_axis_ranges(params["shift"])
    shear = affine_axis_ranges(params["shear"]) if "shear" in params else None

    def warp(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        center = ((width - 1) / 2, (height - 1) / 2)
        angle = _sample_uniform(image, angles) * (pi / 180)
        scale = _sample_uniform(image, scales)
        batch_shape = (image.shape[0],)
        x = torch.randint(min_x, max_x + 1, batch_shape, device=image.device)
        y = torch.randint(min_y, max_y + 1, batch_shape, device=image.device)
        slopes: tuple[float | torch.Tensor, float | torch.Tensor] = (0.0, 0.0)
        if shear is not None:
            shear_x, shear_y = shear
            slopes = (
                (_sample_uniform(image, shear_x) * (pi / 180)).tan(),
                (_sample_uniform(image, shear_y) * (pi / 180)).tan(),
            )
        inverse = centered_inverse_affine(center, (angle.sin(), angle.cos()), scale, (x, y), slopes)
        matrix = torch.stack(inverse, dim=-1).reshape(-1, 2, 3)
        forward = kornia.geometry.transform.invert_affine_transform(matrix)
        return kornia.geometry.transform.warp_affine(image.float(), forward, (height, width), align_corners=True).to(
            image.dtype
        )

    return _TensorOperation(warp)


def _build_random_crop224(params: dict[str, Any]) -> Any | None:
    return _CenteredRandomCrop((params["height"], params["width"]))


class _CenteredRandomCrop(torch.nn.Module):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size
        self.crop = kornia_augmentation.RandomCrop(size=size, p=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        left, top, right, bottom = centered_padding(height, width, max(self.size[0], height), max(self.size[1], width))
        if left or right or top or bottom:
            image = torch_functional.pad(image, (left, right, top, bottom))
        return self.crop(image)


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
    if params["mode"] != "fisheye":
        raise ValueError(f"Kornia optical distortion requires fisheye mode, got {params['mode']!r}")
    limit = params["distort_limit"]
    bounds = parameter_range(limit) if isinstance(limit, (list, tuple)) else (-limit, limit)

    def warp(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        cx, cy = width / 2, height / 2
        x = torch.arange(width, device=image.device, dtype=torch.float32) - cx
        y = torch.arange(height, device=image.device, dtype=torch.float32)[:, None] - cy
        k = _sample_uniform(image, bounds).reshape(-1, 1, 1)
        radius_squared = (x.square() + y.square()) / (cx**2 + cy**2)
        scale = 1 + k * radius_squared
        return kornia.geometry.transform.remap(image.float(), cx + x * scale, cy + y * scale, align_corners=True).to(
            image.dtype
        )

    return _TensorOperation(warp)


def _build_horizontal_flip(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomHorizontalFlip(p=1)


def _build_perspective(params: dict[str, Any]) -> Any:
    size = params["input_size"]
    low, high = params["scale"]

    def warp(image: torch.Tensor) -> torch.Tensor:
        shape = (2, image.shape[0])
        magnitude = torch.empty(shape, device=image.device, dtype=torch.float32).uniform_(low, high)
        sign = (torch.randint(0, 2, shape, device=image.device) * 2) - 1
        x, y = magnitude * sign / size
        coefficients = centered_inverse_perspective((size - 1) / 2, -x, -y)
        forward = torch.stack((*coefficients, torch.ones_like(x)), dim=-1).reshape(-1, 3, 3)
        return kornia.geometry.transform.warp_perspective(image.float(), forward, (size, size), align_corners=True).to(
            image.dtype
        )

    return _TensorOperation(warp)


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
    return _RandomJigsawWithPad(grid=tuple(params["grid"]))


def _build_rotate(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomRotation(
        degrees=tuple(params["angle_range"]),
        resample=params["interpolation"],
        p=1,
    )


def _build_shear(params: dict[str, Any]) -> Any:
    angles = parameter_range(params["shear"])

    def shear(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        slope = (_sample_uniform(image, angles) * (pi / 180)).tan()
        zero = torch.zeros_like(slope)
        one = torch.ones_like(slope)
        matrix = torch.stack((one, slope, -slope * ((height - 1) / 2), zero, one, zero), dim=-1).reshape(-1, 2, 3)
        return kornia.geometry.transform.warp_affine(image.float(), matrix, (height, width), align_corners=True).to(
            image.dtype
        )

    return _TensorOperation(shear)


class _ThinPlateSpline(kornia_augmentation.RandomThinPlateSpline):
    def __init__(self, std_fraction: float) -> None:
        super().__init__(scale=0, align_corners=False, p=1)
        self._std = 2 * std_fraction

    def generate_parameters(self, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        batch, _, height, width = shape
        anchors = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]], device=self.device, dtype=torch.float32)
        # AX anchors are at 0, W/2 and W in continuous pixel coordinates.
        offset = anchors.new_tensor((1 / width, 1 / height))
        dst = (anchors + offset).expand(batch, -1, -1)
        src = dst + torch.randn(dst.shape, device=dst.device, dtype=dst.dtype) * self._std
        return {"src": src, "dst": dst}


def _build_thin_plate_spline(params: dict[str, Any]) -> Any:
    if params["num_control_points"] != _TPS_POINTS_PER_AXIS:
        raise ValueError("Kornia ThinPlateSpline requires four corners and a center point")
    return _ThinPlateSpline(params["distortion_scale"] / 10)


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


def _build_normalize(_params: dict[str, Any]) -> Any | None:
    return None


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
    "CropAndPad": _build_crop_and_pad,
    "Pad": _build_pad,
    "PadIfNeeded": _build_pad_if_needed,
    "LetterBox": _build_letter_box,
    "RandomScale": _build_random_scale,
    "RandomSizedCrop": _build_random_sized_crop,
    "CoarseDropout": _build_coarse_dropout,
    "Colorize": _build_colorize,
    "Emboss": lambda params: _Emboss(parameter_range(params["alpha"]), parameter_range(params["strength"])),
    "ShiftScaleRotate": _build_affine,
    "ChannelSwap": _build_channel_swap,
    "Transpose": lambda _: _TensorOperation(lambda image: image.transpose(-2, -1)),
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
    "Normalize": _build_normalize,
    "Posterize": _build_posterize,
    "LongestMaxSize": _build_longest_max_size,
    "SmallestMaxSize": _build_smallest_max_size,
}
