from __future__ import annotations

from typing import Any

import kornia.augmentation as Kaug
from torch import nn

from benchmark.transforms.kornia_common import set_same_on_batch
from benchmark.transforms.kornia_video_impl import KorniaVideoSequential, create_transform, device
from benchmark.transforms.video_recipe_specs import (
    is_crop_recipe_spec,
    is_supported_by_library,
    recipe_augmentation_specs,
    recipe_name,
    repeated_stats,
    spec_by_name,
)

LIBRARY = "kornia"


def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    return transform(video.to(device))


def _normalize() -> Kaug.Normalize:
    mean, std = repeated_stats()
    return Kaug.Normalize(mean=mean, std=std, p=1)


def _random_crop() -> Kaug.RandomCrop:
    params = spec_by_name("RandomCrop224").params
    return Kaug.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True, p=1, same_on_batch=False)


class _KorniaVideoRecipe(nn.Module):
    def __init__(self, transforms: list[nn.Module]) -> None:
        super().__init__()
        self.video = KorniaVideoSequential(
            *transforms,
            _normalize(),
        )

    def forward(self, video: Any) -> Any:
        return self.video(video)


def _recipe(name: str, transforms: list[nn.Module]) -> dict[str, Any]:
    for transform in transforms:
        set_same_on_batch(transform, False)
    return {"name": name, "transform": _KorniaVideoRecipe(transforms).to(device)}


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs():
    if not is_supported_by_library(_spec, LIBRARY):
        continue
    _transform = create_transform(_spec)
    if _transform is not None:
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        TRANSFORMS.append(_recipe(recipe_name(_spec), _transforms))
