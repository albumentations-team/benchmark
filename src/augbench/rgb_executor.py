"""Execute one RGB production cell from disk to a model-ready CUDA batch."""

from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from augbench.adapters.registry import AdapterRegistry, default_adapter_registry
from augbench.adapters.sources import SourceRef
from augbench.dataset_access import reordered_items
from augbench.execution.pipeline import (
    ReadyGpuBatchConsumer,
    RecipeDataset,
    TorchDataLoaderSource,
    canonicalize_model_batch,
    materialize,
    output_validation,
)
from augbench.measurement_window import measure_window
from augbench.nvml_memory import NvmlProcessMemoryMonitor
from augbench.recipe_contract import validate_rgb_recipes
from augbench.run_records import CellKey, OutputObservation, ResultRecord

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from augbench.adapters.runtime import AugmentationAdapter, RecipeRuntime
    from augbench.recipes.models import RecipeCatalog, RecipeSpec
    from augbench.run_config import FamilyRunConfig


@dataclass(frozen=True)
class _DaliExecution:
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    seed: int
    height: int
    width: int


class _DaliAdapter(Protocol):
    def native_source(self, *, sources: Sequence[Any], recipe: RecipeSpec, execution: _DaliExecution) -> Any: ...


class RGBCellExecutor:
    """Execute cells sequentially inside the one benchmark VM process."""

    def __init__(
        self,
        *,
        config: FamilyRunConfig,
        recipes: RecipeCatalog,
        dataset_files: Sequence[Path],
        adapters: AdapterRegistry | None = None,
    ) -> None:
        if config.family != "rgb" or recipes.family != "rgb":
            raise ValueError("RGB executor requires RGB config, recipes, and dataset")
        if len(dataset_files) != config.dataset.item_count:
            raise ValueError("RGB executor dataset does not match configured item count")
        validate_rgb_recipes(config, recipes)
        self._config = config
        self._recipes = recipes
        self._sources = tuple(SourceRef(path=path) for path in dataset_files)
        self._adapters = adapters or default_adapter_registry()

    def execute(self, cell: CellKey) -> ResultRecord:
        recipe = self._recipe(cell.recipe_id)
        if cell.implementation not in recipe.supported_implementations:
            raise ValueError(f"{cell.implementation} is not declared for {cell.recipe_id}")
        _seed_everything(cell.seed)
        adapter = self._adapters.load(cell.implementation)
        sources = self._sources_for_seed(cell.seed)
        _clear_device_state()
        monitor = NvmlProcessMemoryMonitor()
        monitor.start()
        try:
            # Constructing the recipe graph can allocate CUDA buffers.  It is
            # outside the throughput timer but intentionally inside the
            # process-wide memory window.
            runtime = adapter.build_recipe(recipe)
            if cell.implementation == "dali_gpu":
                throughput, output = self._execute_dali(adapter, recipe, sources, cell.seed)
            else:
                throughput, output = self._execute_dataloader(adapter, runtime, sources)
        finally:
            memory = monitor.stop()
        return ResultRecord(
            run_id=cell.run_id,
            cell=cell,
            status="ok",
            throughput=throughput,
            gpu_memory=memory,
            output=output,
            runtime=_runtime_metadata(adapter),
        )

    def preflight(self, cell: CellKey) -> None:
        """Exercise one real batch without producing a benchmark cell."""
        recipe = self._recipe(cell.recipe_id)
        if cell.implementation not in recipe.supported_implementations:
            raise ValueError(f"{cell.implementation} is not declared for {cell.recipe_id}")
        _seed_everything(cell.seed)
        adapter = self._adapters.load(cell.implementation)
        runtime = adapter.build_recipe(recipe)
        sources = self._sources_for_seed(cell.seed)
        _clear_device_state()
        if cell.implementation == "dali_gpu":
            self._preflight_dali(adapter, recipe, sources, cell.seed)
        else:
            self._preflight_dataloader(adapter, runtime, sources)

    def compile_implementation(self, implementation: str) -> None:
        """Build every catalog recipe declared for one implementation."""
        adapter = self._adapters.load(implementation)
        for recipe in self._recipes.recipes:
            if implementation in recipe.supported_implementations:
                adapter.build_recipe(recipe)

    def _execute_dataloader(
        self,
        adapter: AugmentationAdapter,
        runtime: RecipeRuntime,
        sources: Sequence[Any],
    ) -> tuple[Any, OutputObservation]:
        dataset = RecipeDataset(items=sources, adapter=adapter, runtime=runtime)
        loader = TorchDataLoaderSource(
            dataset=dataset,
            batch_size=self._config.execution.batch_size,
            num_workers=self._config.execution.num_workers,
            prefetch_factor=self._config.execution.prefetch_factor,
            persistent_workers=self._config.execution.persistent_workers,
            pin_memory=True,
            seed=0,
            host_batch_transform=runtime.host_batch_transform,
        )
        consumer = ReadyGpuBatchConsumer(runtime=runtime, expected_channels=self._config.output.channels)
        loader.open()
        try:
            batch_count = self._config.execution.warmup_batches + self._config.execution.measured_batches
            throughput = measure_window(
                batches=(loader.next_batch() for _ in range(batch_count)),
                consume=consumer,
                synchronize=consumer.synchronize,
                warmup_batches=self._config.execution.warmup_batches,
                measured_batches=self._config.execution.measured_batches,
                clock=time.perf_counter,
            )
            return throughput, self._output_observation(consumer.last_validation)
        finally:
            loader.close()

    def _preflight_dataloader(
        self,
        adapter: AugmentationAdapter,
        runtime: RecipeRuntime,
        sources: Sequence[Any],
    ) -> None:
        dataset = RecipeDataset(items=sources, adapter=adapter, runtime=runtime)
        loader = TorchDataLoaderSource(
            dataset=dataset,
            batch_size=self._config.execution.batch_size,
            num_workers=self._config.execution.num_workers,
            prefetch_factor=self._config.execution.prefetch_factor,
            persistent_workers=self._config.execution.persistent_workers,
            pin_memory=True,
            seed=0,
            host_batch_transform=runtime.host_batch_transform,
        )
        consumer = ReadyGpuBatchConsumer(runtime=runtime, expected_channels=self._config.output.channels)
        loader.open()
        try:
            consumer(loader.next_batch())
            consumer.synchronize()
            self._output_observation(consumer.last_validation)
        finally:
            loader.close()

    def _execute_dali(
        self,
        adapter: AugmentationAdapter,
        recipe: RecipeSpec,
        sources: Sequence[Any],
        seed: int,
    ) -> tuple[Any, OutputObservation]:
        native = cast("_DaliAdapter", adapter)
        source = native.native_source(
            sources=sources,
            recipe=recipe,
            execution=_DaliExecution(
                batch_size=self._config.execution.batch_size,
                num_workers=self._config.execution.num_workers,
                prefetch_factor=self._config.execution.prefetch_factor,
                seed=seed,
                height=self._config.output.height,
                width=self._config.output.width,
            ),
        )
        latest: dict[str, object] | None = None

        def consume(batch: tuple[Any, Any]) -> int:
            nonlocal latest
            ready = materialize(
                canonicalize_model_batch(batch[0], expected_channels=self._config.output.channels),
            )
            latest = output_validation(ready)
            return int(ready.shape[0])

        source.open()
        try:
            batch_count = self._config.execution.warmup_batches + self._config.execution.measured_batches
            throughput = measure_window(
                batches=(source.next_batch() for _ in range(batch_count)),
                consume=consume,
                synchronize=_cuda_synchronize,
                warmup_batches=self._config.execution.warmup_batches,
                measured_batches=self._config.execution.measured_batches,
                clock=time.perf_counter,
            )
        finally:
            source.close()
        if latest is None:
            raise RuntimeError("DALI completed without a validated output batch")
        return throughput, self._output_observation(latest)

    def _preflight_dali(
        self,
        adapter: AugmentationAdapter,
        recipe: RecipeSpec,
        sources: Sequence[Any],
        seed: int,
    ) -> None:
        native = cast("_DaliAdapter", adapter)
        source = native.native_source(
            sources=sources,
            recipe=recipe,
            execution=_DaliExecution(
                batch_size=self._config.execution.batch_size,
                num_workers=self._config.execution.num_workers,
                prefetch_factor=self._config.execution.prefetch_factor,
                seed=seed,
                height=self._config.output.height,
                width=self._config.output.width,
            ),
        )
        source.open()
        try:
            batch, _labels = source.next_batch()
            ready = materialize(canonicalize_model_batch(batch, expected_channels=self._config.output.channels))
            self._output_observation(output_validation(ready))
            _cuda_synchronize()
        finally:
            source.close()

    def _sources_for_seed(self, seed: int) -> tuple[Any, ...]:
        return reordered_items(
            self._sources,
            required_items=self._config.execution.required_items,
            seed=seed,
        )

    def _recipe(self, recipe_id: str) -> RecipeSpec:
        matches = [recipe for recipe in self._recipes.recipes if recipe.recipe_id == recipe_id]
        if len(matches) != 1:
            raise ValueError(f"recipe catalog has {len(matches)} entries for {recipe_id!r}")
        return matches[0]

    def _output_observation(self, validation: dict[str, object]) -> OutputObservation:
        output = _output_observation(validation)
        expected_shape = self._config.output.shape
        if output.shape[1:] != expected_shape:
            raise ValueError(f"RGB output shape {output.shape[1:]} does not match configured {expected_shape}")
        return output


def _output_observation(validation: dict[str, object]) -> OutputObservation:
    shape = validation.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or not all(isinstance(value, int) for value in shape):
        raise ValueError("adapter did not report a rank-four output shape")
    return OutputObservation(shape=tuple(shape))


def _runtime_metadata(adapter: AugmentationAdapter) -> dict[str, str]:
    import importlib.metadata

    package = adapter.implementation_id.split("_")[0]
    package_names = {"albumentationsx": "albumentations", "pillow": "pillow"}
    try:
        version = importlib.metadata.version(package_names.get(package, package))
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    try:
        import torch

        device = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError):
        device = "unknown"
    return {"library_version": version, "cuda_device": device}


def _cuda_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _clear_device_state() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party augmentation APIs consume NumPy's global RNG.
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
