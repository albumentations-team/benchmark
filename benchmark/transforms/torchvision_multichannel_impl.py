"""TorchVision multi-channel image benchmark spec."""

import os
from typing import Any

import torch
from torch import nn

from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec
from benchmark.transforms.torchvision_impl import create_transform as _create_rgb_transform

LIBRARY = "torchvision"
NUM_CHANNELS = 9

_RGB_CHUNK_TRANSFORMS = {
    "AutoContrast",
    "Brightness",
    "ColorJiggle",
    "ColorJitter",
    "Contrast",
    "Grayscale",
    "JpegCompression",
    "PhotoMetricDistort",
    "Sharpen",
}


def _transform_filter_names() -> set[str] | None:
    filter_env = os.environ.get("BENCHMARK_TRANSFORMS_FILTER", "").strip()
    if not filter_env:
        return None
    return {name.strip() for name in filter_env.split(",") if name.strip()}


class _ApplyToRgbChunks(nn.Module):
    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(image, chunks=NUM_CHANNELS // 3, dim=0)
        return torch.cat([self.transform(chunk) for chunk in chunks], dim=0)


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


def create_transform(spec: TransformSpec) -> Any | None:
    if spec.name == "Normalize":
        normalize_spec = TransformSpec(
            spec.name,
            {
                **spec.params,
                "mean": (0.485, 0.456, 0.406) * (NUM_CHANNELS // 3),
                "std": (0.229, 0.224, 0.225) * (NUM_CHANNELS // 3),
            },
        )
        return _create_rgb_transform(normalize_spec)

    transform = _create_rgb_transform(spec)
    if transform is not None and spec.name in _RGB_CHUNK_TRANSFORMS:
        return _ApplyToRgbChunks(transform)
    return transform


_ALLOWED_TRANSFORMS = _transform_filter_names()

TRANSFORMS = [
    {"name": spec.name, "transform": transform}
    for spec in TRANSFORM_SPECS
    if (_ALLOWED_TRANSFORMS is None or spec.name in _ALLOWED_TRANSFORMS)
    if (transform := create_transform(spec)) is not None
]
