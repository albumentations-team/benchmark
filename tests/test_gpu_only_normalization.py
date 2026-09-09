from __future__ import annotations

import ast
from pathlib import Path

import pytest

from augbench.adapters.registry import default_adapter_registry
from augbench.implementations.specs import TransformSpec


@pytest.mark.parametrize(
    "module_name",
    [
        "augbench.implementations.albumentationsx_impl",
        "augbench.implementations.kornia_impl",
        "augbench.implementations.torchvision_impl",
    ],
)
def test_rgb_cpu_transform_factories_refuse_normalize(module_name: str) -> None:
    module = pytest.importorskip(module_name)

    assert (
        module.create_transform(
            TransformSpec(
                name="Normalize",
                params={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ),
        )
        is None
    )


def test_active_rgb_cpu_pipeline_modules_do_not_construct_a_cpu_normalizer() -> None:
    root = Path(__file__).parents[1] / "src" / "augbench" / "implementations"
    for filename in (
        "albumentationsx_impl.py",
        "albumentationsx_pipeline_impl.py",
        "kornia_impl.py",
        "pillow_pipeline_impl.py",
        "torchvision_impl.py",
        "torchvision_pipeline_impl.py",
        "kornia_pipeline_impl.py",
    ):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Normalize"
        ]
        assert not calls, filename


def test_current_adapter_registry_exposes_only_rgb() -> None:
    assert all(registration.module.endswith(".rgb") for registration in default_adapter_registry().all())
