"""Size calculations shared by the RGB adapters."""

from __future__ import annotations

import random
from math import cos, radians, sin, sqrt, tan
from typing import TYPE_CHECKING, Any

from augbench.implementations.specs import parameter_range

if TYPE_CHECKING:
    from collections.abc import Sequence

PERSPECTIVE_JITTER_LIMIT = 0.32


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


def perspective_corners(offsets: Sequence[tuple[Any, Any]], height: int, width: int) -> tuple[tuple[Any, Any], ...]:
    tl, tr, br, bl = offsets
    return (
        (tl[0] * width, tl[1] * height),
        ((1 - tr[0]) * width, tr[1] * height),
        ((1 - br[0]) * width, (1 - br[1]) * height),
        (bl[0] * width, (1 - bl[1]) * height),
    )


def perspective_edges_squared(corners: Sequence[tuple[Any, Any]]) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    def distance_squared(p: tuple[Any, Any], q: tuple[Any, Any]) -> Any:
        x = p[0] - q[0]
        y = p[1] - q[1]
        return x * x + y * y

    tl, tr, br, bl = corners
    return (distance_squared(tl, tr), distance_squared(bl, br)), (distance_squared(tl, bl), distance_squared(tr, br))


def inverse_perspective_coefficients(corners: Sequence[tuple[Any, Any]], sx: Any, sy: Any) -> tuple[Any, ...]:
    """Map the output rectangle to the source quadrilateral. sx and sy are reciprocal rectangle extents."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
    dx1, dx2 = x1 - x2, x3 - x2
    dy1, dy2 = y1 - y2, y3 - y2
    dx3 = x0 - x1 + x2 - x3
    dy3 = y0 - y1 + y2 - y3
    determinant = dx1 * dy2 - dx2 * dy1
    g = (dx3 * dy2 - dx2 * dy3) / determinant
    h = (dx1 * dy3 - dx3 * dy1) / determinant
    return (
        (x1 * (1 + g) - x0) * sx,
        (x3 * (1 + h) - x0) * sy,
        x0,
        (y1 * (1 + g) - y0) * sx,
        (y3 * (1 + h) - y0) * sy,
        y0,
        g * sx,
        h * sy,
    )


def sample_perspective_source(height: int, width: int, scale: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    sigma = random.uniform(*scale)  # noqa: S311 - augmentation RNG.
    offsets = [
        (abs(random.gauss(0, sigma)) % PERSPECTIVE_JITTER_LIMIT, abs(random.gauss(0, sigma)) % PERSPECTIVE_JITTER_LIMIT)
        for _ in range(4)
    ]
    return perspective_corners(offsets, height, width)


def sample_perspective_coefficients(height: int, width: int, scale: tuple[float, float]) -> tuple[float, ...]:
    corners = sample_perspective_source(height, width, scale)
    widths, heights = perspective_edges_squared(corners)
    a, b, c, d, e, f, g, h = inverse_perspective_coefficients(
        corners, int(sqrt(max(widths))) / width**2, int(sqrt(max(heights))) / height**2
    )
    # Pillow and TorchVision use the pixel corner as the origin.
    denominator = 1 - (g + h) / 2
    return tuple(
        value / denominator
        for value in (
            a + g / 2,
            b + h / 2,
            c + (denominator - a - b) / 2,
            d + g / 2,
            e + h / 2,
            f + (denominator - d - e) / 2,
            g,
            h,
        )
    )
