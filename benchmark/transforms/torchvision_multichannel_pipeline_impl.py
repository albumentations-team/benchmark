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
        return (image - self.get_buffer("mean")) / self.get_buffer("std")


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


def _random_crop() -> nn.Module:
    import torchvision.transforms.v2 as tv_transforms

    params = spec_by_name("RandomCrop224").params
    return tv_transforms.RandomCrop(size=(params["height"], params["width"]), pad_if_needed=True)


TRANSFORMS: list[dict[str, Any]] = []
for _spec in recipe_augmentation_specs(NUM_CHANNELS):
    _transform = create_transform(_spec)
    if _transform is not None:
        _transforms = [_transform] if is_crop_recipe_spec(_spec) else [_random_crop(), _transform]
        _gpu_transform = _GpuBatchRecipe(_transforms[1] if len(_transforms) > 1 else None)
        _split_transform = _SplitRecipe(_Recipe(_transforms), _transforms[0], _gpu_transform)
        TRANSFORMS.append(
            {"name": recipe_name(_spec), "transform": _split_transform},
        )
