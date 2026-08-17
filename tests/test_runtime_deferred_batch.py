import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import pytest

from augbench.adapters.runtime import PythonModuleAdapter
from augbench.execution.pipeline import TorchDataLoaderSource

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeSpec


class _Recipe:
    recipe_id = "Resize224+Normalize+ToTensor"
    supported_implementations = ("cpu_impl", "gpu_impl")
    stages = ()


class _TensorDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> object:
        torch = pytest.importorskip("torch")
        return torch.full((3, 2, 2), index, dtype=torch.float32)


class _CpuStage:
    def __call__(self, sample: object) -> tuple[str, object]:
        return ("cpu", sample)


class _GpuStage:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> "_GpuStage":
        self.device = device
        return self

    def __call__(self, batch: object) -> tuple[str, str | None, object]:
        return ("gpu", self.device, batch)


class _HostBatchStage:
    def __call__(self, batch: object) -> tuple[str, object]:
        return ("host", batch)


class _DeferredRecipe:
    def __init__(self) -> None:
        self.defer_batch_to_gpu = True
        self.cpu_transform = _CpuStage()
        self.gpu_transform = _GpuStage()


class _DeferredRecipeWithHostBatch(_DeferredRecipe):
    def __init__(self) -> None:
        super().__init__()
        self.host_batch_transform = _HostBatchStage()


class _OrdinarySplitRecipe:
    def __init__(self) -> None:
        self.cpu_transform = _CpuStage()
        self.gpu_transform = _GpuStage()
        self.host_batch_transform = _HostBatchStage()

    def __call__(self, sample: object) -> tuple[str, object]:
        return ("full", sample)


def test_cpu_adapter_can_defer_a_recipe_tail_until_after_h2d(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "test_deferred_cpu_recipe"
    module = ModuleType(module_name)
    module.__dict__["build_recipe"] = lambda _recipe, _implementation: _DeferredRecipe()

    def call_transform(transform: Any, value: Any) -> Any:
        return transform(value)

    module.__dict__["__call__"] = call_transform
    monkeypatch.setitem(sys.modules, module_name, module)

    adapter = PythonModuleAdapter(
        implementation_id="cpu_impl",
        recipe_module=module_name,
        load_source=lambda source: source,
        placement="cpu",
        source_id="test",
    )

    runtime = adapter.build_recipe(cast("RecipeSpec", _Recipe()))

    assert runtime.sample_transform is not None
    assert runtime.batch_transform is not None
    assert runtime.sample_transform("image") == ("cpu", "image")
    assert runtime.batch_transform("batch") == ("gpu", "cuda", "batch")


def test_cpu_adapter_can_transform_a_collated_batch_before_h2d(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "test_deferred_cpu_recipe_with_host_batch"
    module = ModuleType(module_name)
    module.__dict__["build_recipe"] = lambda _recipe, _implementation: _DeferredRecipeWithHostBatch()

    def call_transform(transform: Any, value: Any) -> Any:
        return transform(value)

    module.__dict__["__call__"] = call_transform
    monkeypatch.setitem(sys.modules, module_name, module)

    adapter = PythonModuleAdapter(
        implementation_id="cpu_impl",
        recipe_module=module_name,
        load_source=lambda source: source,
        placement="cpu",
        source_id="test",
    )

    runtime = adapter.build_recipe(cast("RecipeSpec", _Recipe()))

    assert runtime.host_batch_transform is not None
    assert runtime.host_batch_transform("batch") == ("host", "batch")


def test_loader_casts_the_collated_host_batch_before_pinning() -> None:
    torch = pytest.importorskip("torch")
    observed_shapes: list[tuple[int, ...]] = []

    def cast_batch(batch: object) -> object:
        assert isinstance(batch, torch.Tensor)
        observed_shapes.append(tuple(batch.shape))
        return batch.to(dtype=torch.float16)

    loader = TorchDataLoaderSource(
        dataset=_TensorDataset(),  # type: ignore[arg-type]
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
        seed=137,
        host_batch_transform=cast_batch,
    )
    loader.open()
    try:
        batch = loader.next_batch()
    finally:
        loader.close()

    assert observed_shapes == [(2, 3, 2, 2)]
    assert batch.dtype == torch.float16


def test_cpu_adapter_keeps_an_ordinary_split_recipe_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "test_ordinary_cpu_recipe"
    module = ModuleType(module_name)
    module.__dict__["build_recipe"] = lambda _recipe, _implementation: _OrdinarySplitRecipe()

    def call_transform(transform: Any, value: Any) -> Any:
        return transform(value)

    module.__dict__["__call__"] = call_transform
    monkeypatch.setitem(sys.modules, module_name, module)

    adapter = PythonModuleAdapter(
        implementation_id="cpu_impl",
        recipe_module=module_name,
        load_source=lambda source: source,
        placement="cpu",
        source_id="test",
    )

    runtime = adapter.build_recipe(cast("RecipeSpec", _Recipe()))

    assert runtime.sample_transform is not None
    assert runtime.sample_transform("image") == ("full", "image")
    assert runtime.batch_transform is None


def test_gpu_adapter_can_cast_a_collated_shape_prefix_before_h2d(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "test_gpu_recipe_with_host_batch"
    module = ModuleType(module_name)
    module.__dict__["build_recipe"] = lambda _recipe, _implementation: _OrdinarySplitRecipe()
    module.__dict__["__call__"] = lambda transform, value: transform(value)
    monkeypatch.setitem(sys.modules, module_name, module)

    adapter = PythonModuleAdapter(
        implementation_id="gpu_impl",
        recipe_module=module_name,
        load_source=lambda source: source,
        placement="cuda",
        source_id="test",
    )

    runtime = adapter.build_recipe(cast("RecipeSpec", _Recipe()))

    assert runtime.sample_transform is not None
    assert runtime.host_batch_transform is not None
    assert runtime.host_batch_transform("batch") == ("host", "batch")
