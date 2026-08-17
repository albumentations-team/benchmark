from __future__ import annotations

from typing import TYPE_CHECKING, Any

import albumentations as A

from augbench.implementations.albumentationsx_impl import create_transform
from augbench.implementations.gpu_normalize import GpuBatchNormalize
from augbench.implementations.recipe_stages import normalization_stats, transform_specs
from augbench.recipes.runtime import UnsupportedRecipeError

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeSpec


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image=image)["image"]


class _DeferredRecipe:
    defer_batch_to_gpu = True

    def __init__(self, *, cpu_transform: Any, recipe: RecipeSpec) -> None:
        mean, std = normalization_stats(recipe)
        self.cpu_transform = cpu_transform
        self.gpu_transform = GpuBatchNormalize(mean=mean, std=std, input_layout="BHWC")


def build_recipe(recipe: RecipeSpec, implementation_id: str) -> _DeferredRecipe:
    if implementation_id != "albumentationsx_cpu":
        raise UnsupportedRecipeError(f"{implementation_id!r} is not an AlbumentationsX CPU implementation")
    transforms = [_transform(spec, recipe.recipe_id) for spec in transform_specs(recipe)]
    return _DeferredRecipe(cpu_transform=A.Compose(transforms), recipe=recipe)


def _transform(spec: Any, recipe_id: str) -> Any:
    transform = create_transform(spec)
    if transform is None:
        raise UnsupportedRecipeError(f"AlbumentationsX cannot build {spec.name!r} for {recipe_id!r}")
    return transform
