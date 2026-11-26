"""RandAugment parameters shared by the paths that assemble the policy themselves.

TorchVision, Kornia, and DALI ship their own policy. AlbumentationsX and Pillow have none, so those two
build it from stock operations. Their magnitudes follow TorchVision v2 at the benchmark's bin, so the
assembled paths and the native TorchVision path apply the same strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OPERATIONS = (
    "Identity",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "Brightness",
    "Color",
    "Contrast",
    "Sharpness",
    "Posterize",
    "Solarize",
    "AutoContrast",
    "Equalize",
)
SIGNED = frozenset(
    {"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate", "Brightness", "Color", "Contrast", "Sharpness"}
)

# TorchVision's translation spans 150 of 331 pixels of the side at the top bin.
_TRANSLATE_REACH = 150 / 331
_MAX_SHEAR = 0.3
_MAX_ROTATE_DEGREES = 30.0
_MAX_ENHANCE = 0.9
_POSTERIZE_BIT_SPAN = 4
_FULL_BITS = 8


@dataclass(frozen=True)
class RandAugmentSpec:
    num_ops: int
    magnitude: int
    bins: int
    fill: float

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RandAugmentSpec:
        if tuple(params["operations"]) != OPERATIONS:
            raise ValueError("RandAugment operations must be TorchVision's pool in its order")
        if params["interpolation"] != "bilinear":
            raise ValueError("assembled RandAugment paths warp bilinearly")
        spec = cls(
            num_ops=int(params["num_ops"]),
            magnitude=int(params["magnitude"]),
            bins=int(params["num_magnitude_bins"]),
            fill=float(params["fill"]),
        )
        if spec.num_ops < 1 or not 0 <= spec.magnitude < spec.bins:
            raise ValueError("RandAugment needs num_ops >= 1 and 0 <= magnitude < num_magnitude_bins")
        return spec

    @property
    def level(self) -> float:
        return self.magnitude / (self.bins - 1)

    @property
    def shear(self) -> float:
        return _MAX_SHEAR * self.level

    @property
    def translate_fraction(self) -> float:
        return _TRANSLATE_REACH * self.level

    @property
    def rotate_degrees(self) -> float:
        return _MAX_ROTATE_DEGREES * self.level

    @property
    def enhance(self) -> float:
        return _MAX_ENHANCE * self.level

    @property
    def posterize_bits(self) -> int:
        return _FULL_BITS - round(self.magnitude / ((self.bins - 1) / _POSTERIZE_BIT_SPAN))

    @property
    def solarize_fraction(self) -> float:
        return 1.0 - self.level
