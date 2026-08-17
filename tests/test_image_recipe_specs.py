from pathlib import Path

from augbench.implementations.recipe_stages import normalization_stats, transform_specs
from augbench.recipes.load import load_recipe_catalog


def test_recipe_catalog_is_the_single_source_of_runtime_parameters() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    resize = next(recipe for recipe in recipes.recipes if recipe.recipe_id == "Resize224+Normalize+ToTensor")

    assert [spec.name for spec in transform_specs(resize)] == ["Resize"]
    assert transform_specs(resize)[0].params["target_size"] == 224
    assert normalization_stats(resize) == ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def test_rgb_recipes_declare_a_float16_ready_gpu_batch() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")

    assert all(recipe.stages[-1].parameters["dtype"] == "float16" for recipe in recipes.recipes)
