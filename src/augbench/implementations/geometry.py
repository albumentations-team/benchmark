"""Size calculations shared by the RGB adapters."""

from __future__ import annotations

import random
from math import cos, radians, sin, tan
from typing import Any


def aspect_size(height: int, width: int, size: int, *, cover: bool = False) -> tuple[int, int]:
    scale = size / (min(height, width) if cover else max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def centered_padding(height: int, width: int, target_height: int, target_width: int) -> tuple[int, int, int, int]:
    extra_height, extra_width = target_height - height, target_width - width
    top, left = extra_height // 2, extra_width // 2
    return left, top, extra_width - left, extra_height - top


def emboss_kernel(alpha: float, strength: float) -> tuple[float, ...]:
    edge = alpha * strength
    diagonal = alpha + edge
    return (-diagonal, -edge, 0, -edge, 1, edge, 0, edge, diagonal)


def centered_inverse_affine(
    center: tuple[float, float],
    angle: float,
    scale: float,
    shift: tuple[float, float],
    shear: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float, float, float]:
    cx, cy = center
    x, y = cx + shift[0], cy + shift[1]
    sine, cosine = sin(radians(angle)), cos(radians(angle))
    sx, sy = (tan(radians(value)) for value in shear)
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
