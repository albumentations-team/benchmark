from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from augbench.adapters.registry import default_adapter_registry


def test_active_recipe_modules_reference_existing_local_modules() -> None:
    """Keep a removed helper from becoming a production-only import failure."""
    recipe_modules = {
        registration.module
        for registration in default_adapter_registry().all()
        if registration.adapter_kind == "python-module"
    }
    recipe_modules.update(
        {
            "augbench.implementations.albumentationsx_pipeline_impl",
            "augbench.implementations.kornia_pipeline_impl",
            "augbench.implementations.pillow_pipeline_impl",
            "augbench.implementations.torchvision_pipeline_impl",
            "augbench.adapters.dali.native",
        },
    )

    checked: set[str] = set()
    for module_name in recipe_modules:
        _assert_local_import_tree(module_name, checked)


def _assert_local_import_tree(module_name: str, checked: set[str]) -> None:
    if module_name in checked:
        return
    checked.add(module_name)
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, module_name
    assert spec.origin is not None, module_name
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    local_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("augbench.")
    }
    for imported_module in local_imports:
        imported_spec = importlib.util.find_spec(imported_module)
        assert imported_spec is not None, f"{module_name} imports missing local module {imported_module}"
        _assert_local_import_tree(imported_module, checked)
