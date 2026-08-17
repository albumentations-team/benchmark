"""Read active transform and normalization parameters from a recipe."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augbench.implementations.specs import TransformSpec

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeSpec


_TERMINAL_STAGES = frozenset({"Normalize", "ToTensor"})
_COLLATABLE_STAGES = frozenset({"Resize", "RandomCrop224", "RandomResizedCrop"})


def transform_specs(recipe: RecipeSpec) -> tuple[TransformSpec, ...]:
    """Return every recipe stage executed before GPU batch normalization."""
    return tuple(
        TransformSpec(name=stage.operation_id, params=dict(stage.parameters))
        for stage in recipe.stages
        if stage.operation_id not in _TERMINAL_STAGES
    )


def collatable_prefix(recipe: RecipeSpec) -> tuple[tuple[TransformSpec, ...], tuple[TransformSpec, ...]]:
    """Split a recipe at the first stage that gives DataLoader a common shape."""
    specs = transform_specs(recipe)
    for index, spec in enumerate(specs, start=1):
        if spec.name in _COLLATABLE_STAGES:
            return specs[:index], specs[index:]
    raise ValueError(f"recipe {recipe.recipe_id!r} never forms a collatable image shape")


def normalization_stats(recipe: RecipeSpec) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the recipe's explicit RGB normalization parameters."""
    stages = [stage for stage in recipe.stages if stage.operation_id == "Normalize"]
    if len(stages) != 1:
        raise ValueError(f"recipe {recipe.recipe_id!r} must contain exactly one Normalize stage")
    mean = _numbers(stages[0].parameters, "mean", recipe.recipe_id)
    std = _numbers(stages[0].parameters, "std", recipe.recipe_id)
    if len(mean) != recipe.channels or len(std) != recipe.channels:
        raise ValueError(f"recipe {recipe.recipe_id!r} normalization channels do not match the recipe")
    return mean, std


def _numbers(parameters: dict[str, Any], name: str, recipe_id: str) -> tuple[float, ...]:
    values = parameters.get(name)
    if not isinstance(values, list) or not all(isinstance(value, (float, int)) for value in values):
        raise TypeError(f"recipe {recipe_id!r} Normalize.{name} must be a numeric list")
    return tuple(float(value) for value in values)
