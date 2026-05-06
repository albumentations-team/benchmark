from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms.v2 as tv_transforms

from benchmark.transforms.torchvision_video_impl import create_transform
from benchmark.transforms.video_recipe_specs import (
    is_crop_recipe_spec,
    recipe_augmentation_specs,
    recipe_name,
    repeated_stats,
    spec_by_name,
)

LIBRARY = "torchvision"


def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    return transform(video)


def _normalize() -> tv_transforms.Normalize:
    mean, std = repeated_stats()
    return tv_transforms.Normalize(mean=mean, std=std)


def _to_float_tensor() -> tv_transforms.ToDtype:
    return tv_transforms.ToDtype(torch.float32, scale=True)


def _random_crop() -> tv_transforms.RandomCrop:
    params = spec_by_name("RandomCrop224").params
    return tv_transforms.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True)


def _recipe(name: str, transforms: list[Any]) -> dict[str, Any]:
    return {"name": name, "transform": tv_transforms.Compose([*transforms, _to_float_tensor(), _normalize()])}


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs():
    _transform = create_transform(_spec)
    if _transform is not None:
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        TRANSFORMS.append(_recipe(recipe_name(_spec), _transforms))
