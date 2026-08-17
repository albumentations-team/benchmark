from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from augbench.implementations.gpu_normalize import GpuBatchNormalize
from augbench.implementations.kornia_common import force_per_sample_randomness
from augbench.implementations.kornia_impl import create_transform
from augbench.implementations.recipe_stages import collatable_prefix, normalization_stats, transform_specs
from augbench.recipes.runtime import UnsupportedRecipeError

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeSpec


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image.unsqueeze(0)).squeeze(0)


class _HostBatchFloat16(nn.Module):
    """Shrink a completed CPU Kornia batch before its pinned H2D copy."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dtype != torch.float32:
            raise TypeError(f"Kornia CPU recipe must produce float32, got {batch.dtype}")
        return batch.to(dtype=torch.float16)


class _DeferredCpuRecipe(nn.Module):
    defer_batch_to_gpu = True

    def __init__(self, *, transforms: tuple[nn.Module, ...], recipe: RecipeSpec) -> None:
        super().__init__()
        mean, std = normalization_stats(recipe)
        self.cpu_transform = force_per_sample_randomness(nn.Sequential(*transforms))
        self.host_batch_transform = _HostBatchFloat16()
        self.gpu_transform = GpuBatchNormalize(mean=mean, std=std, input_layout="BCHW")


class _GpuBatchRecipe(nn.Module):
    def __init__(self, *, tail: nn.Module | None, recipe: RecipeSpec) -> None:
        super().__init__()
        mean, std = normalization_stats(recipe)
        self._tail = tail
        self._batch_normalize = GpuBatchNormalize(mean=mean, std=std, input_layout="BCHW")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if not batch.is_floating_point():
            raise TypeError(f"Kornia GPU transforms require floating input, got {batch.dtype}")
        batch = batch.to(dtype=torch.float16)
        if self._tail is not None:
            batch = self._tail(batch)
        return self._batch_normalize(batch)


class _GpuRecipe(nn.Module):
    def __init__(self, *, cpu_transform: nn.Module, gpu_transform: nn.Module) -> None:
        super().__init__()
        self.cpu_transform = cpu_transform
        self.host_batch_transform = _HostBatchFloat16()
        self.gpu_transform = gpu_transform

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.cpu_transform(image)


def build_recipe(recipe: RecipeSpec, implementation_id: str) -> nn.Module:
    cpu_specs, gpu_specs = collatable_prefix(recipe)
    cpu_transforms = tuple(_transform(spec, recipe.recipe_id) for spec in cpu_specs)
    gpu_transforms = tuple(_transform(spec, recipe.recipe_id) for spec in gpu_specs)
    if not cpu_transforms:
        raise UnsupportedRecipeError(f"Kornia recipe {recipe.recipe_id!r} has no augmentation stages")
    if implementation_id == "kornia_cpu":
        transforms = tuple(_transform(spec, recipe.recipe_id) for spec in transform_specs(recipe))
        return _DeferredCpuRecipe(transforms=transforms, recipe=recipe)
    if implementation_id == "kornia_gpu":
        tail = force_per_sample_randomness(nn.Sequential(*gpu_transforms)) if gpu_transforms else None
        return _GpuRecipe(
            cpu_transform=force_per_sample_randomness(nn.Sequential(*cpu_transforms)),
            gpu_transform=_GpuBatchRecipe(tail=tail, recipe=recipe),
        )
    raise UnsupportedRecipeError(f"{implementation_id!r} is not a Kornia implementation")


def _transform(spec: Any, recipe_id: str) -> nn.Module:
    transform = create_transform(spec)
    if transform is None:
        raise UnsupportedRecipeError(f"Kornia cannot build {spec.name!r} for {recipe_id!r}")
    return transform
