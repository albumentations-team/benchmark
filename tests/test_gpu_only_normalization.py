from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from augbench.adapters.registry import default_adapter_registry
from augbench.execution.pipeline import ReadyGpuBatchConsumer
from augbench.implementations.recipe_stages import normalization_stats
from augbench.implementations.specs import TransformSpec
from augbench.recipes.load import load_recipe_catalog


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
    ):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Normalize"
        ]
        assert not calls, filename


@pytest.mark.parametrize(
    ("implementation_id", "augmentation_id"),
    [
        ("kornia_cpu", "HorizontalFlip"),
        ("kornia_gpu", "HorizontalFlip"),
        ("torchvision_cpu", "HorizontalFlip"),
        ("torchvision_gpu", "HorizontalFlip"),
        ("torchvision_gpu", "JpegCompression"),
    ],
)
def test_rgb_native_normalize_receives_one_cuda_float16_batch(
    implementation_id: str,
    augmentation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    is_kornia = implementation_id.startswith("kornia")
    native = pytest.importorskip("kornia.enhance" if is_kornia else "torchvision.transforms.v2")
    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == augmentation_id)
    runtime = default_adapter_registry().load(implementation_id).build_recipe(recipe)
    mean, std = normalization_stats(recipe)
    forward = native.Normalize.forward
    expected = []

    def record_normalize(normalizer: Any, batch: Any) -> Any:
        assert batch.shape == (2, 3, 224, 224)
        assert batch.device.type == "cuda"
        assert batch.dtype == torch.float16
        mean_tensor = batch.new_tensor(mean).view(1, 3, 1, 1)
        std_tensor = batch.new_tensor(std).view(1, 3, 1, 1)
        expected.append((batch - mean_tensor) / std_tensor)
        return forward(normalizer, batch)

    monkeypatch.setattr(native.Normalize, "forward", record_normalize)
    samples = [torch.full((3, 224, 224), value, dtype=torch.uint8) for value in (64, 192)]
    if runtime.batch_input_on_host:
        io = pytest.importorskip("torchvision.io")
        assert runtime.sample_transform is None
        assert runtime.host_batch_transform is None
        samples[1] = torch.full((3, 240, 200), 192, dtype=torch.uint8)
        batch = [bytes(encoded.tolist()) for encoded in io.encode_jpeg(samples)]
    else:
        assert runtime.sample_transform is not None
        if is_kornia:
            samples = [sample.float().div_(255.0) for sample in samples]
        batch = torch.stack([runtime.sample_transform(sample) for sample in samples])
        if runtime.host_batch_transform is not None:
            batch = runtime.host_batch_transform(batch)
    output = ReadyGpuBatchConsumer(runtime=runtime, expected_channels=3).prepare(batch)

    assert len(expected) == 1
    torch.testing.assert_close(output, expected[0], rtol=0, atol=0)


def test_current_adapter_registry_exposes_only_rgb() -> None:
    assert all(registration.module.endswith(".rgb") for registration in default_adapter_registry().all())
