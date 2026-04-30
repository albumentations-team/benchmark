from __future__ import annotations

from typing import Any

import torch
from torch import nn

from benchmark.transforms.image_recipe_specs import (
    is_crop_recipe_spec,
    recipe_augmentation_specs,
    recipe_name,
    repeated_stats,
    spec_by_name,
)
from benchmark.transforms.torchvision_multichannel_impl import create_transform

LIBRARY = "torchvision"
NUM_CHANNELS = 9


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


class _Recipe(nn.Module):
    def __init__(self, transforms: list[nn.Module]) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        mean, std = repeated_stats(NUM_CHANNELS)
        self.register_buffer("mean", torch.tensor(mean).view(NUM_CHANNELS, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(NUM_CHANNELS, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        image = image.float() / 255.0 if not image.is_floating_point() else image.float()
        return (image - self.mean) / self.std


def _random_crop() -> nn.Module:
    import torchvision.transforms.v2 as tv_transforms

    params = spec_by_name("RandomCrop224").params
    return tv_transforms.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True)


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs(NUM_CHANNELS):
    _transform = create_transform(_spec)
    if _transform is not None:
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        TRANSFORMS.append(
            {"name": recipe_name(_spec), "transform": _Recipe(_transforms)},
        )
