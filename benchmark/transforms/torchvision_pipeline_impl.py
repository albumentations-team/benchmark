from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms.v2 as tv_transforms
from torch import nn

from benchmark.transforms.image_recipe_specs import (
    is_crop_recipe_spec,
    recipe_augmentation_specs,
    recipe_name,
    repeated_stats,
    spec_by_name,
)
from benchmark.transforms.torchvision_impl import create_transform

LIBRARY = "torchvision"
NUM_CHANNELS = 3


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


class _BatchNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mean, std = repeated_stats(NUM_CHANNELS)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, NUM_CHANNELS, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, NUM_CHANNELS, 1, 1))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean")
        std = self.get_buffer("std")
        batch = batch.float() / 255.0 if not batch.is_floating_point() else batch.float()
        return (batch - mean) / std


class _GpuBatchRecipe(nn.Module):
    def __init__(self, transform: nn.Module | None) -> None:
        super().__init__()
        self.per_sample_transform = transform
        self.batch_normalize = _BatchNormalize()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.per_sample_transform is not None:
            batch = torch.stack([self.per_sample_transform(sample) for sample in batch], dim=0)
        return self.batch_normalize(batch)


class _SplitRecipe(nn.Module):
    def __init__(self, full_transform: nn.Module, cpu_transform: nn.Module, gpu_transform: nn.Module) -> None:
        super().__init__()
        self.full_transform = full_transform
        self.cpu_transform = cpu_transform
        self.gpu_transform = gpu_transform

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.full_transform(image)


def _normalize() -> tv_transforms.Normalize:
    mean, std = repeated_stats(NUM_CHANNELS)
    return tv_transforms.Normalize(mean=mean, std=std)


def _to_float_tensor() -> tv_transforms.ToDtype:
    return tv_transforms.ToDtype(torch.float32, scale=True)


def _random_crop() -> tv_transforms.RandomCrop:
    params = spec_by_name("RandomCrop224").params
    return tv_transforms.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True)


def _recipe(name: str, transforms: list[Any]) -> dict[str, Any]:
    cpu_transform = transforms[0]
    gpu_transform = _GpuBatchRecipe(transforms[1] if len(transforms) > 1 else None)
    full_transform = tv_transforms.Compose([*transforms, _to_float_tensor(), _normalize()])
    if isinstance(cpu_transform, nn.Module):
        return {"name": name, "transform": _SplitRecipe(full_transform, cpu_transform, gpu_transform)}
    return {"name": name, "transform": full_transform}


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs(NUM_CHANNELS):
    _transform = create_transform(_spec)
    if _transform is not None:
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        TRANSFORMS.append(_recipe(recipe_name(_spec), _transforms))
