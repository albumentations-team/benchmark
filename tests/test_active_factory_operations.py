from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from augbench.adapters.dali.native import _apply_vertical_flip, supports_recipe
from augbench.recipes.load import load_recipe_catalog


def test_recipe_factories_contain_only_current_image_operations() -> None:
    root = Path(__file__).parents[1]
    catalog_operations = {
        stage.operation_id
        for family in ("rgb", "grayscale", "imagenet_train")
        for recipe in load_recipe_catalog(root / "catalog" / "recipes" / f"{family}.yaml").recipes
        for stage in recipe.stages
    }

    for filename in (
        "albumentationsx_impl.py",
        "kornia_impl.py",
        "pillow_impl.py",
        "torchvision_impl.py",
    ):
        factory_operations = _factory_operations(root / "src" / "augbench" / "implementations" / filename)
        assert factory_operations, filename
        assert factory_operations <= catalog_operations, filename


@pytest.mark.parametrize("family", ["rgb", "imagenet_train"])
def test_dali_stage_handlers_cover_the_supported_catalog_recipes(family: str) -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / f"{family}.yaml")
    for recipe in recipes.recipes:
        if "dali_gpu" in recipe.supported_implementations:
            assert supports_recipe(recipe), recipe.recipe_id


def test_dali_vertical_flip_disables_default_horizontal_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    flip = Mock()
    monkeypatch.setitem(sys.modules, "nvidia.dali", SimpleNamespace(fn=SimpleNamespace(flip=flip)))
    image = object()
    _apply_vertical_flip(image, {}, 224, 224)

    flip.assert_called_once_with(image, horizontal=0, vertical=1)


def test_kornia_gpu_support_policy_matches_the_recipe_catalog() -> None:
    root = Path(__file__).parents[1]
    recipes = load_recipe_catalog(root / "catalog/recipes/rgb.yaml")
    builders = _factory_operations(root / "src/augbench/implementations/kornia_impl.py")

    for recipe in recipes.recipes:
        if "kornia_gpu" in recipe.supported_implementations:
            assert {stage.operation_id for stage in recipe.stages[:-2]} <= builders, recipe.recipe_id


def _factory_operations(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not _is_transform_builder_registry(node):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        return {key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    raise AssertionError(f"{path} must define _TRANSFORM_BUILDERS as a string-keyed dictionary")


def _is_transform_builder_registry(node: ast.Assign) -> bool:
    return any(isinstance(target, ast.Name) and target.id == "_TRANSFORM_BUILDERS" for target in node.targets)
