"""Kornia transforms excluded from video DataLoader/pipeline recipes.

These hit CUDA device asserts, integer-tensor sampling bugs, or CPU/GPU tensor mixups in our
video pipeline runs; keep them available for Kornia micro and image pipeline benchmarks.
"""

from __future__ import annotations

KORNIA_BENCHMARK_EXCLUDED_NAMES: frozenset[str] = frozenset(
    {
        "CLAHE",
        "ColorJiggle",
        "CornerIllumination",
        "Erasing",
        "LinearIllumination",
        "MotionBlur",
        "Perspective",
        "Posterize",
        "RandomJigsaw",
        "RandomRotate90",
        "Shear",
    },
)
