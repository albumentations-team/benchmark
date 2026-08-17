from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torchvision.transforms.v2 as tv_transforms
from torch import nn

from augbench.implementations.gpu_normalize import GpuBatchNormalize
from augbench.implementations.recipe_stages import collatable_prefix, normalization_stats, transform_specs
from augbench.implementations.torchvision_impl import create_transform
from augbench.recipes.runtime import UnsupportedRecipeError

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeSpec


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


class _GpuBatchRecipe(nn.Module):
    def __init__(self, *, tail: nn.Module | None, recipe: RecipeSpec) -> None:
        super().__init__()
        mean, std = normalization_stats(recipe)
        self._tail = tail
        self._batch_normalize = GpuBatchNormalize(mean=mean, std=std, input_layout="BCHW")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dtype == torch.uint8:
            batch = batch.to(dtype=torch.float16).div_(255.0)
        elif batch.is_floating_point():
            batch = batch.to(dtype=torch.float16)
        else:
            raise TypeError(f"TorchVision GPU transforms require uint8 or floating input, got {batch.dtype}")
        if self._tail is not None:
            batch = torch.stack([self._tail(sample) for sample in batch], dim=0)
        return self._batch_normalize(batch)


class _SplitRecipe(nn.Module):
    def __init__(self, *, cpu_transform: nn.Module, gpu_transform: nn.Module) -> None:
        super().__init__()
        self.cpu_transform = cpu_transform
        self.gpu_transform = gpu_transform

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.cpu_transform(image)


class _DeferredCpuRecipe(nn.Module):
    defer_batch_to_gpu = True

    def __init__(self, *, transforms: tuple[nn.Module, ...], recipe: RecipeSpec) -> None:
        super().__init__()
        mean, std = normalization_stats(recipe)
        self.cpu_transform = tv_transforms.Compose(list(transforms))
        self.gpu_transform = GpuBatchNormalize(mean=mean, std=std, input_layout="BCHW")


def build_recipe(recipe: RecipeSpec, implementation_id: str) -> nn.Module:
    cpu_specs, gpu_specs = collatable_prefix(recipe)
    cpu_transforms = tuple(_transform(spec, recipe.recipe_id) for spec in cpu_specs)
    gpu_transforms = tuple(_transform(spec, recipe.recipe_id) for spec in gpu_specs)
    if not cpu_transforms:
        raise UnsupportedRecipeError(f"TorchVision recipe {recipe.recipe_id!r} has no augmentation stages")
    if implementation_id == "torchvision_cpu":
        transforms = tuple(_transform(spec, recipe.recipe_id) for spec in transform_specs(recipe))
        return _DeferredCpuRecipe(transforms=transforms, recipe=recipe)
    if implementation_id == "torchvision_gpu":
        return _SplitRecipe(
            cpu_transform=tv_transforms.Compose(list(cpu_transforms)),
            gpu_transform=_GpuBatchRecipe(
                tail=tv_transforms.Compose(list(gpu_transforms)) if gpu_transforms else None,
                recipe=recipe,
            ),
        )
    raise UnsupportedRecipeError(f"{implementation_id!r} is not a TorchVision implementation")


def _transform(spec: Any, recipe_id: str) -> nn.Module:
    transform = create_transform(spec)
    if transform is None:
        raise UnsupportedRecipeError(f"TorchVision cannot build {spec.name!r} for {recipe_id!r}")
    return transform
