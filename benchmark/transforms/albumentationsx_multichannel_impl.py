r"""AlbumentationsX multi-channel image benchmark spec.

Tests transforms on 9-channel images (3x stacked RGB, simulating e.g. hyperspectral data).
The runner synthesizes these in-memory via make_multichannel_loader; no special dataset needed.

Run with:
    python -m benchmark.cli run \\
        --spec benchmark/transforms/albumentationsx_multichannel_impl.py \\
        --data-dir /path/to/images \\
        --output output/multichannel \\
        --num-channels 9
"""

import os
from typing import Any

import albumentations as A
import cv2
import numpy as np

from benchmark.transforms.albumentationsx_impl import create_transform as _create_rgb_transform
from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

LIBRARY = "albumentationsx"
NUM_CHANNELS = 9  # 3 RGB repetitions stacked; must match --num-channels passed to runner

# These RGB transforms reject (H, W, 9) directly. Colorize only warns and returns
# the input unchanged, so benchmarking it would measure a no-op.
_EXCLUDED_TRANSFORMS = {
    "ColorJitter",
    "ColorJiggle",
    "RGBShift",
    "Equalize",
    "CLAHE",
    "Hue",
    "PlankianJitter",
    "Rain",
    "Saturation",
    "Snow",
    "PhotoMetricDistort",
    "Colorize",
}


def _transform_filter_names() -> set[str] | None:
    filter_env = os.environ.get("BENCHMARK_TRANSFORMS_FILTER", "").strip()
    if not filter_env:
        return None
    return {name.strip() for name in filter_env.split(",") if name.strip()}


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return np.ascontiguousarray(transform(image=image)["image"])


def create_transform(spec: TransformSpec) -> Any | None:
    if spec.name in _EXCLUDED_TRANSFORMS:
        return None
    if spec.name == "Normalize":
        return A.Normalize(
            mean=(0.485, 0.456, 0.406) * (NUM_CHANNELS // 3),
            std=(0.229, 0.224, 0.225) * (NUM_CHANNELS // 3),
            p=1,
        )
    if spec.name == "Grayscale":
        return A.ToGray(num_output_channels=NUM_CHANNELS, method="average", p=1)
    return _create_rgb_transform(spec)


def _build_transforms() -> list[dict[str, Any]]:
    allowed = _transform_filter_names()
    transforms: list[dict[str, Any]] = []
    for spec in TRANSFORM_SPECS:
        if allowed is not None and spec.name not in allowed:
            continue
        transform = create_transform(spec)
        if transform is not None:
            transforms.append({"name": spec.name, "transform": transform})
    return transforms


TRANSFORMS = _build_transforms()
