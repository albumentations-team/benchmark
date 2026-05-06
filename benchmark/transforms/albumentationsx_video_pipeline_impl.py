from __future__ import annotations

from typing import Any

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

from benchmark.transforms.albumentationsx_video_impl import create_transform
from benchmark.transforms.video_recipe_specs import (
    is_crop_recipe_spec,
    recipe_augmentation_specs,
    recipe_name,
    repeated_stats,
    spec_by_name,
)

LIBRARY = "albumentationsx"


def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    result = transform(images=video)["images"]
    if isinstance(result, torch.Tensor):
        return result.contiguous()
    if isinstance(result, list) and result and all(isinstance(frame, torch.Tensor) for frame in result):
        return torch.stack(result).contiguous()
    return result


def _normalize() -> A.Normalize:
    mean, std = repeated_stats()
    return A.Normalize(mean=mean, std=std, p=1)


def _random_crop() -> A.RandomCrop:
    params = spec_by_name("RandomCrop224").params
    return A.RandomCrop(height=params["height"], width=params["width"], pad_if_needed=True, p=1)


def _recipe(name: str, transforms: list[Any]) -> dict[str, Any]:
    return {"name": name, "transform": A.Compose([*transforms, _normalize(), ToTensorV2()])}


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs():
    _transform = create_transform(_spec)
    if _transform is not None:
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        TRANSFORMS.append(_recipe(recipe_name(_spec), _transforms))
