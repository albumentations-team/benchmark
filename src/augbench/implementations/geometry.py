"""Size calculations shared by the RGB adapters."""

from __future__ import annotations

import random
from math import cos, radians, sin, tan
from typing import Any

from augbench.implementations.specs import parameter_range


def aspect_size(height: int, width: int, size: int, *, cover: bool = False) -> tuple[int, int]:
    scale = size / (min(height, width) if cover else max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def centered_padding(height: int, width: int, target_height: int, target_width: int) -> tuple[int, int, int, int]:
    extra_height, extra_width = target_height - height, target_width - width
    top, left = extra_height // 2, extra_width // 2
    return left, top, extra_width - left, extra_height - top


def emboss_kernel(alpha: Any, strength: Any) -> tuple[Any, ...]:
    edge = alpha * strength
    diagonal = alpha + edge
    zero = alpha * 0
    return (-diagonal, -edge, zero, -edge, zero + 1, edge, zero, edge, diagonal)


def sample_inverse_affine(center: tuple[float, float], params: dict[str, Any]) -> tuple[float, ...]:
    angle = radians(random.uniform(*parameter_range(params["angle"])))  # noqa: S311 - augmentation RNG.
    scale = random.uniform(*parameter_range(params["scale"]))  # noqa: S311 - augmentation RNG.
    x_range, y_range = affine_axis_ranges(params["shift"])
    x = random.randint(*x_range)  # noqa: S311 - augmentation RNG.
    y = random.randint(*y_range)  # noqa: S311 - augmentation RNG.
    sx = sy = 0.0
    if "shear" in params:
        sx_range, sy_range = affine_axis_ranges(params["shear"])
        sx = random.uniform(*sx_range)  # noqa: S311 - augmentation RNG.
        sy = random.uniform(*sy_range)  # noqa: S311 - augmentation RNG.
    rotation = sin(angle), cos(angle)
    return centered_inverse_affine(center, rotation, scale, (x, y), (tan(radians(sx)), tan(radians(sy))))


def affine_axis_ranges(value: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    x, y = (value["x"], value["y"]) if isinstance(value, dict) else value
    return parameter_range(x), parameter_range(y)


def centered_inverse_affine(
    center: tuple[float, float],
    rotation: tuple[Any, Any],
    scale: Any,
    shift: tuple[Any, Any],
    shear_slopes: tuple[Any, Any],
) -> tuple[Any, Any, Any, Any, Any, Any]:
    cx, cy = center
    x, y = cx + shift[0], cy + shift[1]
    sine, cosine = rotation
    sx, sy = shear_slopes
    divisor = scale * (1 - sx * sy)
    a, b = (cosine + sine * sy) / divisor, -(cosine * sx + sine) / divisor
    d, e = (sine - cosine * sy) / divisor, (cosine - sine * sx) / divisor
    return a, b, cx - a * x - b * y, d, e, cy - d * x - e * y


def x_shear_matrix(degrees: float, center_y: float) -> tuple[float, float, float, float, float, float]:
    """Map source pixels to destination pixels, fixing the horizontal center line."""
    slope = tan(radians(degrees))
    return (1.0, slope, -slope * center_y, 0.0, 1.0, 0.0)


def sample_perspective_axis(low: float, high: float, size: int) -> float:
    magnitude = random.uniform(low, high)  # noqa: S311 - augmentation RNG.
    return random.choice((-magnitude, magnitude)) / size  # noqa: S311 - augmentation RNG.


def centered_inverse_perspective(center: float, x: Any, y: Any) -> tuple[Any, ...]:
    offset = -center * (x + y)
    scale = 1 + offset
    return (
        (1 + center * x) / scale,
        center * y / scale,
        center * offset / scale,
        center * x / scale,
        (1 + center * y) / scale,
        center * offset / scale,
        x / scale,
        y / scale,
    )


def sample_inverse_perspective(size: int, scale: list[float], *, pixel_center: bool) -> tuple[Any, ...]:
    low, high = scale
    return centered_inverse_perspective(
        (size - int(pixel_center)) / 2,
        sample_perspective_axis(low, high, size),
        sample_perspective_axis(low, high, size),
    )
