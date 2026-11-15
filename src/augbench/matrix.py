"""Build the complete benchmark matrix directly from the run configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from augbench.run_records import CellKey

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeCatalog
    from augbench.run_config import FamilyRunConfig


def build_matrix(*, run_id: str, config: FamilyRunConfig, recipes: RecipeCatalog) -> tuple[CellKey, ...]:
    if recipes.family != config.family:
        raise ValueError("recipe catalog family does not match family configuration")
    _validate_selection(config, recipes)
    cells = tuple(
        CellKey(
            run_id=run_id,
            family="rgb",
            implementation=implementation,
            recipe_id=recipe.recipe_id,
            seed=seed,
        )
        for implementation in config.implementations
        for recipe in recipes.recipes
        if implementation in recipe.supported_implementations
        and (config.measurements is None or implementation in config.measurements.get(recipe.recipe_id, ()))
        for seed in config.execution.seeds
    )
    if not cells:
        raise ValueError("the configuration and recipe catalog have no shared implementations")
    return cells


def _validate_selection(config: FamilyRunConfig, recipes: RecipeCatalog) -> None:
    if config.measurements is None:
        return
    supported = {recipe.recipe_id: set(recipe.supported_implementations) for recipe in recipes.recipes}
    active = set(config.implementations)
    for recipe, implementations in config.measurements.items():
        if not implementations or not set(implementations) <= active & supported.get(recipe, set()):
            raise ValueError(f"invalid measurement selection for {recipe!r}")
