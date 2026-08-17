from __future__ import annotations

import ast
from pathlib import Path

from augbench.recipes.load import load_recipe_catalog


def test_recipe_factories_contain_only_current_rgb_operations() -> None:
    root = Path(__file__).parents[1]
    recipes = load_recipe_catalog(root / "catalog" / "recipes" / "rgb.yaml")
    catalog_operations = {stage.operation_id for recipe in recipes.recipes for stage in recipe.stages}

    for filename in (
        "albumentationsx_impl.py",
        "kornia_impl.py",
        "pillow_impl.py",
        "torchvision_impl.py",
    ):
        factory_operations = _factory_operations(root / "src" / "augbench" / "implementations" / filename)
        assert factory_operations <= catalog_operations, filename


def test_kornia_illumination_factories_adapt_catalog_lists_to_kornia_ranges() -> None:
    source = (Path(__file__).parents[1] / "src" / "augbench" / "implementations" / "kornia_impl.py").read_text(
        encoding="utf-8"
    )

    assert source.count('gain=tuple(params["gain"])') == 3


def _factory_operations(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not _is_spec_name(node.left):
            continue
        if len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        if isinstance(node.ops[0], ast.Eq) and isinstance(node.comparators[0], ast.Constant):
            value = node.comparators[0].value
            if isinstance(value, str):
                names.add(value)
        if isinstance(node.ops[0], ast.In) and isinstance(node.comparators[0], (ast.Set, ast.Tuple, ast.List)):
            names.update(
                element.value
                for element in node.comparators[0].elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return names


def _is_spec_name(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "spec"
        and node.attr == "name"
    )
