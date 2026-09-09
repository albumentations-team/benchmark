from __future__ import annotations

import ast
from pathlib import Path

from augbench.adapters.dali.native import supports_recipe
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
        assert factory_operations, filename
        assert factory_operations <= catalog_operations, filename


def test_dali_stage_handlers_cover_the_supported_catalog_recipes() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    for recipe in recipes.recipes:
        if "dali_gpu" in recipe.supported_implementations:
            assert supports_recipe(recipe), recipe.recipe_id


def test_kornia_factories_adapt_catalog_lists_to_kornia_ranges() -> None:
    source = (Path(__file__).parents[1] / "src" / "augbench" / "implementations" / "kornia_impl.py").read_text(
        encoding="utf-8"
    )

    assert source.count('gain=tuple(params["gain"])') == 3
    assert 'amount=tuple(params["amount"])' in source
    assert 'salt_vs_pepper=tuple(params["salt_vs_pepper"])' in source
    assert 'snow_coefficient=tuple(params["snow_point_range"])' in source
    assert 'RandomRotation90(times=tuple(params["times"]), p=1)' in source
    assert 'grid_size=tuple(params["tile_grid_size"])' in source
    assert 'kernel_size=tuple(params["kernel_size"])' in source
    assert 'angle=tuple(params["angle_range"])' in source
    assert 'direction=tuple(params["direction_range"])' in source
    assert 'scale=tuple(params["scale"])' in source
    assert 'ratio=tuple(params["ratio"])' in source
    assert 'size=tuple(params["size"])' in source
    assert '_RandomJigsawWithPad(grid=tuple(params["grid"]))' in source
    assert 'degrees=tuple(params["angle_range"])' in source
    assert "angle = self.angle.to(image)" in source
    assert "translation = self.translation.to(image)" in source
    assert "scale_factor = self.scale_factor.to(image)" in source
    assert "shear = self.shear.to(image)" in source


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
