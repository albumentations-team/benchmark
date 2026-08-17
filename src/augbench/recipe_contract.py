"""Validate that the recipe catalog agrees with the family-level output config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeCatalog, RecipeSpec
    from augbench.run_config import FamilyRunConfig


def validate_rgb_recipes(config: FamilyRunConfig, recipes: RecipeCatalog) -> None:
    if config.family != "rgb" or recipes.family != "rgb":
        raise ValueError("RGB recipe validation requires RGB config and catalog")
    for recipe in recipes.recipes:
        _validate_recipe(config, recipe)


def _validate_recipe(config: FamilyRunConfig, recipe: RecipeSpec) -> None:
    if recipe.channels != config.output.channels:
        raise ValueError(f"recipe {recipe.recipe_id} has the wrong channel count")
    stages = {stage.operation_id: stage.parameters for stage in recipe.stages}
    tensor = stages.get("ToTensor")
    if not isinstance(tensor, dict) or tensor.get("dtype") != config.output.dtype:
        raise ValueError(f"recipe {recipe.recipe_id} does not match output dtype")
    shape_stages = [
        stage for stage in recipe.stages if stage.operation_id in {"Resize", "RandomCrop224", "RandomResizedCrop"}
    ]
    if not shape_stages:
        raise ValueError(f"recipe {recipe.recipe_id} has no explicit shape-forming stage")
    shape_stage = shape_stages[-1]
    if shape_stage.operation_id == "Resize":
        target = _integer(shape_stage.parameters, "target_size", recipe.recipe_id)
        if (target, target) != (config.output.height, config.output.width):
            raise ValueError(f"recipe {recipe.recipe_id} does not match the configured Resize shape")
        return
    if shape_stage.operation_id == "RandomCrop224":
        _require_shape(shape_stage.parameters, config, recipe.recipe_id)
        return
    if shape_stage.operation_id == "RandomResizedCrop":
        size = shape_stage.parameters.get("size")
        if size != [config.output.height, config.output.width]:
            raise ValueError(f"recipe {recipe.recipe_id} does not match the configured RandomResizedCrop shape")
        return
    raise ValueError(f"recipe {recipe.recipe_id} has an unsupported shape-forming stage")


def _require_shape(parameters: dict[str, Any], config: FamilyRunConfig, recipe_id: str) -> None:
    height = _integer(parameters, "height", recipe_id)
    width = _integer(parameters, "width", recipe_id)
    if (height, width) != (config.output.height, config.output.width):
        raise ValueError(f"recipe {recipe_id} does not match the configured RandomCrop shape")


def _integer(parameters: dict[str, Any], name: str, recipe_id: str) -> int:
    value = parameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"recipe {recipe_id} has no integer {name}")
    return value
