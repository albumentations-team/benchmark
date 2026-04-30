from __future__ import annotations

from typing import Any

import kornia.augmentation as Kaug
from torch import nn

from benchmark.transforms.image_recipe_specs import (
    is_crop_recipe_spec,
    recipe_augmentation_specs,
    recipe_name,
    repeated_stats,
    spec_by_name,
)
from benchmark.transforms.kornia_impl import create_transform

LIBRARY = "kornia"
NUM_CHANNELS = 3


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image.unsqueeze(0)).squeeze(0)


def _normalize() -> Kaug.Normalize:
    mean, std = repeated_stats(NUM_CHANNELS)
    return Kaug.Normalize(mean=mean, std=std, p=1)


def _random_crop() -> Kaug.RandomCrop:
    params = spec_by_name("RandomCrop224").params
    return Kaug.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True, p=1)


def _recipe(name: str, transforms: list[nn.Module]) -> dict[str, Any]:
    return {"name": name, "transform": nn.Sequential(*transforms, _normalize())}


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs(NUM_CHANNELS):
    _transform = create_transform(_spec)
    if _transform is not None:
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        TRANSFORMS.append(_recipe(recipe_name(_spec), _transforms))
