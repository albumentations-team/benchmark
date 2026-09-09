from pathlib import Path

import pytest

from augbench.recipe_contract import validate_rgb_recipes
from augbench.recipes.load import load_recipe_catalog
from augbench.run_config import load_family_config

ROOT = Path(__file__).parents[1]


def test_all_rgb_recipe_shapes_are_bound_to_the_family_config() -> None:
    config = load_family_config(ROOT / "configs" / "families" / "rgb.yaml")
    recipes = load_recipe_catalog(ROOT / config.recipes)

    validate_rgb_recipes(config, recipes)


def test_recipe_contract_rejects_a_mismatched_target_size() -> None:
    config = load_family_config(ROOT / "configs" / "families" / "rgb.yaml")
    recipes = load_recipe_catalog(ROOT / config.recipes)
    mismatched = config.model_copy(update={"output": config.output.model_copy(update={"height": 225})})

    with pytest.raises(ValueError, match="recipe"):
        validate_rgb_recipes(mismatched, recipes)
