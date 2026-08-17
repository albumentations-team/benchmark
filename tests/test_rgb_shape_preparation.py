from pathlib import Path

from augbench.implementations.recipe_stages import collatable_prefix
from augbench.recipes.load import load_recipe_catalog


def test_shape_preparation_is_explicit_in_the_catalog() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    by_id = {recipe.recipe_id: recipe for recipe in recipes.recipes}

    assert [stage.operation_id for stage in by_id["Resize224+Normalize+ToTensor"].stages] == [
        "Resize",
        "Normalize",
        "ToTensor",
    ]
    assert [stage.operation_id for stage in by_id["Pad+RandomCrop224+Normalize+ToTensor"].stages] == [
        "Pad",
        "RandomCrop224",
        "Normalize",
        "ToTensor",
    ]
    prefix, tail = collatable_prefix(by_id["LongestMaxSize+RandomCrop224+Normalize+ToTensor"])
    assert [stage.name for stage in prefix] == ["LongestMaxSize", "RandomCrop224"]
    assert tail == ()
