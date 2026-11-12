"""Size calculations shared by the RGB adapters."""

from __future__ import annotations

from math import cos, radians, sin


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
    center: tuple[float, float], angle: float, scale: float, shift: tuple[float, float]
) -> tuple[float, float, float, float, float, float]:
    cx, cy = center
    x, y = cx + shift[0], cy + shift[1]
    a, b = cos(radians(angle)) / scale, sin(radians(angle)) / scale
    return a, -b, cx - a * x + b * y, b, a, cy - b * x - a * y
