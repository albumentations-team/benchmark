from pathlib import Path

from augbench.matrix import build_matrix
from augbench.recipes.load import load_recipe_catalog
from augbench.run_config import load_family_config

ROOT = Path(__file__).parents[1]


def test_matrix_contains_only_supported_library_recipe_seed_cells() -> None:
    config = load_family_config(ROOT / "configs" / "families" / "rgb.yaml")
    recipes = load_recipe_catalog(ROOT / config.recipes)
    cells = build_matrix(run_id="a" * 64, config=config, recipes=recipes)

    assert cells
    assert len({cell.cell_id for cell in cells}) == len(cells)
    assert {cell.seed for cell in cells} == {137, 138, 139}
    assert {cell.implementation for cell in cells} <= set(config.implementations)
    supported = {
        (recipe.recipe_id, implementation)
        for recipe in recipes.recipes
        for implementation in recipe.supported_implementations
    }
    assert {(cell.recipe_id, cell.implementation) for cell in cells} <= supported


def test_matrix_runs_each_implementation_through_all_recipes_and_seeds_before_the_next() -> None:
    config = load_family_config(ROOT / "configs" / "families" / "rgb.yaml")
    recipes = load_recipe_catalog(ROOT / config.recipes)
    cells = build_matrix(run_id="a" * 64, config=config, recipes=recipes)

    implementations = [cell.implementation for cell in cells]
    groups = [
        implementation
        for index, implementation in enumerate(implementations)
        if index == 0 or implementation != implementations[index - 1]
    ]
    assert groups == list(config.implementations)
    for implementation in config.implementations:
        implementation_cells = [cell for cell in cells if cell.implementation == implementation]
        assert implementation_cells
        assert {cell.seed for cell in implementation_cells} == set(config.execution.seeds)
