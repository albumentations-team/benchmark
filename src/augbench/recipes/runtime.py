from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeSpec, RecipeStage


class UnsupportedRecipeError(RuntimeError):
    """Raised when a declarative recipe has no implementation in an adapter."""


class RecipeBackend[Compiled](Protocol):
    def compile(self, stages: tuple[RecipeStage, ...]) -> Compiled: ...


def compile_recipe[Compiled](
    recipe: RecipeSpec,
    *,
    implementation_id: str,
    backend: RecipeBackend[Compiled],
) -> Compiled:
    if implementation_id not in recipe.supported_implementations:
        raise UnsupportedRecipeError(
            f"recipe {recipe.recipe_id!r} is unsupported by implementation {implementation_id!r}",
        )
    return backend.compile(recipe.stages)
