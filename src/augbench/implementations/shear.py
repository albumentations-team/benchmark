"""Pixel-space X shear shared by the RGB adapters."""

from math import radians, tan


def x_shear_matrix(degrees: float, center_y: float) -> tuple[float, float, float, float, float, float]:
    """Map source pixels to destination pixels, fixing the horizontal center line."""
    slope = tan(radians(degrees))
    return (1.0, slope, -slope * center_y, 0.0, 1.0, 0.0)
