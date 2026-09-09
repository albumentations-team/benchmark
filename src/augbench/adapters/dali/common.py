from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from augbench.recipes.runtime import UnsupportedRecipeError

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeSpec


class DaliAdapter:
    implementation_id: str
    placement: Literal["cuda"] = "cuda"

    def __init__(self, implementation_id: str) -> None:
        self.implementation_id = implementation_id

    def load_source(self, source: Any) -> Any:
        return source

    def build_recipe(self, recipe: RecipeSpec) -> Any:
        from augbench.adapters.dali.native import supports_recipe

        if not supports_recipe(recipe):
            raise UnsupportedRecipeError(f"DALI does not implement every stage in {recipe.recipe_id!r}")
        return recipe

    def native_source(self, *, sources: Any, recipe: RecipeSpec, execution: Any) -> Any:
        from augbench.adapters.dali.native import DaliBatchSource

        return DaliBatchSource(
            sources=sources,
            recipe=recipe,
            execution=execution,
        )

    def metadata(self) -> dict[str, str]:
        return {"adapter": type(self).__name__, "placement": "cuda", "reader": "dali-native"}
