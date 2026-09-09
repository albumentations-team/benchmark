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
        for seed in config.execution.seeds
    )
    if not cells:
        raise ValueError("the configuration and recipe catalog have no shared implementations")
    return cells
