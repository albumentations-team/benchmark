from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class _RecipeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecipeStage(_RecipeModel):
    operation_id: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    stochastic: bool
    randomness_scope: Literal["deterministic", "per-image", "shared-across-frames"]

    @model_validator(mode="after")
    def validate_randomness(self) -> RecipeStage:
        if self.stochastic == (self.randomness_scope == "deterministic"):
            raise ValueError("stochastic stages require a non-deterministic randomness scope")
        return self


class RecipeSpec(_RecipeModel):
    recipe_id: str = Field(min_length=1)
    augmentation_id: str = Field(min_length=1)
    channels: int = Field(ge=1)
    stages: tuple[RecipeStage, ...] = Field(min_length=1)
    supported_libraries: tuple[str, ...] = Field(min_length=2)
    supported_implementations: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_stages(self) -> RecipeSpec:
        stage_ids = tuple(stage.operation_id for stage in self.stages)
        if stage_ids[-2:] != ("Normalize", "ToTensor"):
            raise ValueError("every recipe must end with Normalize and ToTensor")
        return self


class RecipeCatalog(_RecipeModel):
    schema_version: Literal[1] = 1
    recipe_set_id: str = Field(min_length=1)
    family: Literal["rgb"]
    recipes: tuple[RecipeSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_recipes(self) -> RecipeCatalog:
        recipe_ids = [recipe.recipe_id for recipe in self.recipes]
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("recipe catalog contains duplicate recipe IDs")
        if any(recipe.channels != 3 for recipe in self.recipes):
            raise ValueError("RGB recipes require three channels")
        return self
