from __future__ import annotations

from typing import Any, cast

import kornia.augmentation as Kaug
from torch import nn

from benchmark.transforms.image_recipe_specs import (
    is_crop_recipe_spec,
    recipe_augmentation_specs,
    recipe_name,
    repeated_stats,
    spec_by_name,
)
from benchmark.transforms.kornia_multichannel_impl import create_transform

LIBRARY = "kornia"
NUM_CHANNELS = 9


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image.unsqueeze(0)).squeeze(0)


def _force_per_image_randomness(transform: nn.Module) -> nn.Module:
    for module in transform.modules():
        if hasattr(module, "same_on_batch"):
            cast("Any", module).same_on_batch = False
    return transform


class _SplitRecipe(nn.Module):
    def __init__(self, cpu_transform: nn.Module, gpu_transform: nn.Module) -> None:
        super().__init__()
        self.cpu_transform = cpu_transform
        self.gpu_transform = gpu_transform

    def forward(self, image: Any) -> Any:
        return self.gpu_transform(self.cpu_transform(image))


def _normalize() -> Kaug.Normalize:
    mean, std = repeated_stats(NUM_CHANNELS)
    return Kaug.Normalize(mean=mean, std=std, p=1)


def _random_crop() -> Kaug.RandomCrop:
    params = spec_by_name("RandomCrop224").params
    return Kaug.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True, p=1)


def _recipe(name: str, transforms: list[nn.Module]) -> dict[str, Any]:
    cpu_transform = transforms[0]
    gpu_transform = _force_per_image_randomness(nn.Sequential(*transforms[1:], _normalize()))
    return {"name": name, "transform": _SplitRecipe(cpu_transform, gpu_transform)}


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs(NUM_CHANNELS):
    _transform = create_transform(_spec)
    if _transform is not None:
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        TRANSFORMS.append(_recipe(recipe_name(_spec), _transforms))
