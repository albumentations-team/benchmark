from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np

from benchmark.transforms.albumentationsx_multichannel_impl import create_transform
from benchmark.transforms.image_recipe_specs import (
    is_crop_recipe_spec,
    recipe_augmentation_specs,
    recipe_name,
    repeated_stats,
    spec_by_name,
)

LIBRARY = "albumentationsx"
NUM_CHANNELS = 9


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return np.ascontiguousarray(transform(image=image)["image"])


def _normalize() -> A.Normalize:
    mean, std = repeated_stats(NUM_CHANNELS)
    return A.Normalize(mean=mean, std=std, p=1)


def _random_crop() -> A.RandomCrop:
    params = spec_by_name("RandomCrop224").params
    return A.RandomCrop(height=params["height"], width=params["width"], pad_if_needed=True, p=1)


def _recipe(name: str, transforms: list[Any]) -> dict[str, Any]:
    return {"name": name, "transform": A.Compose([*transforms, _normalize()])}


def _is_compose_transform(transform: Any) -> bool:
    return hasattr(transform, "available_keys")


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs(NUM_CHANNELS):
    _transform = create_transform(_spec)
    if _transform is not None and _is_compose_transform(_transform):
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        TRANSFORMS.append(_recipe(recipe_name(_spec), _transforms))
