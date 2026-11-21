import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from augbench.adapters.runtime import PythonModuleAdapter, PythonModuleAdapterConfig
from augbench.execution.pipeline import TorchDataLoaderConfig, TorchDataLoaderSource, _collate_images

if TYPE_CHECKING:
    from augbench.execution.pipeline import RecipeDataset
    from augbench.recipes.models import RecipeSpec


def test_collate_flips_cropped_rgb_views_directly_into_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    image = np.arange(6 * 11 * 3, dtype=np.uint8).reshape(6, 11, 3)
    crop = image[1:5, 2:6]
    samples = [crop, crop[:, ::-1], crop[::-1], crop[::-1, ::-1], crop[..., ::-1], image[1:5, 1:8:2][:, ::-1]]
    before = image.copy()
    actual_flip = cv2.flip
    writes = []

    def track_flip(source: np.ndarray, code: int, *, dst: np.ndarray) -> np.ndarray:
        writes.append((code, dst))
        return actual_flip(source, code, dst=dst)

    monkeypatch.setattr(cv2, "flip", track_flip)
    batch = _collate_images(samples)

    assert torch.equal(batch, torch.from_numpy(np.stack(samples)))
    assert batch.is_contiguous()
    assert [(code, dst.ctypes.data) for code, dst in writes] == [
        (1, batch[1].data_ptr()),
        (0, batch[2].data_ptr()),
        (-1, batch[3].data_ptr()),
    ]
    np.testing.assert_array_equal(image, before)


def test_collate_preserves_float_views_and_stack_dtype_promotion() -> None:
    torch = pytest.importorskip("torch")
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    flipped = image[:, ::-1].astype(np.float32, order="C")[::-1]
    expected = torch.from_numpy(np.stack([image, flipped]))
    batch = _collate_images([image, flipped])

    assert batch.dtype == torch.float32
    assert torch.equal(batch, expected)


class _Recipe:
    recipe_id = "Resize224"
    supported_implementations = ("cpu_impl", "gpu_impl")
    stages = ()


class _TensorDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> object:
        torch = pytest.importorskip("torch")
        return torch.full((3, 2, 2), index, dtype=torch.float32)


class _NumpyBackedTensorDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> object:
        torch = pytest.importorskip("torch")
        array = np.full((3, 2, 2), index, dtype=np.uint8)
        return torch.from_numpy(array)


class _FlippedCropDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> np.ndarray:
        pixels = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        return pixels[index : index + 4, 1:5][:, ::-1]


def test_loader_packs_flipped_views_from_worker() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    dataset = _FlippedCropDataset()
    loader = TorchDataLoaderSource(
        dataset=cast("RecipeDataset", dataset),
        config=TorchDataLoaderConfig(
            batch_size=2,
            num_workers=1,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=False,
            seed=137,
        ),
    )
    loader.open()
    try:
        batch = loader.next_batch()
    finally:
        loader.close()

    assert batch.is_contiguous()
    assert torch.equal(batch, torch.from_numpy(np.stack([dataset[0], dataset[1]])))


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
        config=PythonModuleAdapterConfig(
            implementation_id="cpu_impl",
            recipe_module=module_name,
            placement="cpu",
            source_id="test",
        ),
        load_source=lambda source: source,
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
        config=PythonModuleAdapterConfig(
            implementation_id="cpu_impl",
            recipe_module=module_name,
            placement="cpu",
            source_id="test",
        ),
        load_source=lambda source: source,
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
        config=TorchDataLoaderConfig(
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
            pin_memory=False,
            seed=137,
            host_batch_transform=cast_batch,
        ),
    )
    loader.open()
    try:
        batch = loader.next_batch()
    finally:
        loader.close()

    assert observed_shapes == [(2, 3, 2, 2)]
    assert batch.dtype == torch.float16


def test_loader_stacks_numpy_backed_tensors_in_worker_processes() -> None:
    torch = pytest.importorskip("torch")
    loader = TorchDataLoaderSource(
        dataset=_NumpyBackedTensorDataset(),  # type: ignore[arg-type]
        config=TorchDataLoaderConfig(
            batch_size=2,
            num_workers=1,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=False,
            seed=137,
        ),
    )
    loader.open()
    try:
        batch = loader.next_batch()
    finally:
        loader.close()

    assert torch.equal(batch, torch.tensor([0, 1], dtype=torch.uint8).view(2, 1, 1, 1).expand(2, 3, 2, 2))


def test_cpu_adapter_keeps_an_ordinary_split_recipe_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "test_ordinary_cpu_recipe"
    module = ModuleType(module_name)
    module.__dict__["build_recipe"] = lambda _recipe, _implementation: _OrdinarySplitRecipe()

    def call_transform(transform: Any, value: Any) -> Any:
        return transform(value)

    module.__dict__["__call__"] = call_transform
    monkeypatch.setitem(sys.modules, module_name, module)

    adapter = PythonModuleAdapter(
        config=PythonModuleAdapterConfig(
            implementation_id="cpu_impl",
            recipe_module=module_name,
            placement="cpu",
            source_id="test",
        ),
        load_source=lambda source: source,
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
        config=PythonModuleAdapterConfig(
            implementation_id="gpu_impl",
            recipe_module=module_name,
            placement="cuda",
            source_id="test",
        ),
        load_source=lambda source: source,
    )

    runtime = adapter.build_recipe(cast("RecipeSpec", _Recipe()))

    assert runtime.sample_transform is not None
    assert runtime.host_batch_transform is not None
    assert runtime.host_batch_transform("batch") == ("host", "batch")
