from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np

_MODEL_BATCH_DIMENSIONS = 4

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from augbench.adapters.runtime import AugmentationAdapter, RecipeRuntime


class DatasetStageError(RuntimeError):
    """Preserve the failing input stage across a DataLoader worker boundary."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"AUGBENCH_DATASET_STAGE={stage}: {cause}")


class RecipeDataset:
    def __init__(
        self,
        *,
        items: Sequence[Any],
        adapter: AugmentationAdapter,
        runtime: RecipeRuntime,
    ) -> None:
        if not items:
            raise ValueError("recipe dataset requires at least one item")
        self._items = items
        self._adapter = adapter
        self._runtime = runtime

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        item = self._items[index]
        try:
            sample = self._adapter.load_source(item)
        except Exception as error:
            raise DatasetStageError("source", error) from error
        if self._runtime.sample_transform is not None:
            try:
                sample = self._runtime.sample_transform(sample)
            except Exception as error:
                raise DatasetStageError("augmentation", error) from error
        return sample


@dataclass(frozen=True)
class TorchDataLoaderConfig:
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    persistent_workers: bool
    pin_memory: bool
    seed: int
    host_batch_transform: Callable[[Any], Any] | None = None
    shuffle: bool = False
    drop_last: bool = True


class TorchDataLoaderSource:
    def __init__(
        self,
        *,
        dataset: RecipeDataset,
        config: TorchDataLoaderConfig,
    ) -> None:
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if config.num_workers == 0 and (config.prefetch_factor is not None or config.persistent_workers):
            raise ValueError("prefetch and persistent workers require num_workers > 0")
        self._dataset = dataset
        self._batch_size = config.batch_size
        self._num_workers = config.num_workers
        self._prefetch_factor = config.prefetch_factor
        self._persistent_workers = config.persistent_workers
        self._pin_memory = config.pin_memory
        self._seed = config.seed
        self._host_batch_transform = config.host_batch_transform
        self._shuffle = config.shuffle
        self._drop_last = config.drop_last
        self._loader: Any | None = None
        self._iterator: Any | None = None

    def open(self) -> None:
        import torch
        from torch.utils.data import DataLoader

        generator = torch.Generator().manual_seed(self._seed)
        worker_options: dict[str, Any] = {}
        if self._num_workers > 0:
            worker_options = {
                "persistent_workers": self._persistent_workers,
                "prefetch_factor": self._prefetch_factor or 2,
                "worker_init_fn": _seed_worker,
            }
        loader = DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
            pin_memory=self._pin_memory,
            collate_fn=(
                _collate_images
                if self._host_batch_transform is None
                else partial(_collate_then_transform, transform=self._host_batch_transform)
            ),
            generator=generator,
            **worker_options,
        )
        self._loader = loader
        self._iterator = iter(loader)

    def next_batch(self) -> Any:
        if self._iterator is None:
            raise RuntimeError("DataLoader source is not open")
        return next(self._iterator)

    def close(self) -> None:
        if self._iterator is not None:
            shutdown = getattr(self._iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        self._iterator = None
        self._loader = None


class ReadyGpuBatchConsumer:
    _runtime: RecipeRuntime

    def __init__(self, *, runtime: RecipeRuntime, expected_channels: int) -> None:
        self._runtime = runtime
        self._expected_channels = expected_channels
        self._last_validation: dict[str, object] | None = None

    def prepare(self, batch: Any) -> Any:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("ready-batch scopes require CUDA")
        device_batch = batch.to("cuda", non_blocking=True)
        if self._runtime.batch_transform is not None:
            device_batch = self._runtime.batch_transform(device_batch)
        device_batch = canonicalize_model_batch(device_batch, expected_channels=self._expected_channels)
        ready = materialize(device_batch)
        self._last_validation = output_validation(ready)
        return ready

    def __call__(self, batch: Any) -> int:
        ready = self.prepare(batch)
        return _batch_size(ready)

    @staticmethod
    def synchronize() -> None:
        import torch

        torch.cuda.synchronize()

    @property
    def last_validation(self) -> dict[str, object]:
        if self._last_validation is None:
            raise RuntimeError("no ready GPU batch has been validated")
        return self._last_validation


def canonicalize_model_batch(batch: Any, *, expected_channels: int) -> Any:
    shape = tuple(batch.shape)
    if len(shape) != _MODEL_BATCH_DIMENSIONS or shape[1] != expected_channels:
        raise ValueError(f"RGB batch must be BCHW with {expected_channels} channels, got {shape}")
    _require_ready_gpu_float16(batch)
    return batch


def output_validation(batch: Any) -> dict[str, object]:
    return {
        "family": "rgb",
        "shape": list(batch.shape),
        "dtype": str(batch.dtype).removeprefix("torch."),
        "device": str(batch.device),
        "layout": "BCHW",
    }


def _require_ready_gpu_float16(batch: Any) -> None:
    import torch

    if not isinstance(batch, torch.Tensor):
        raise TypeError(f"RGB ready batch must be a torch.Tensor, got {type(batch).__name__}")
    if batch.device.type != "cuda":
        raise ValueError(f"RGB ready batch must be on CUDA, got {batch.device}")
    if batch.dtype != torch.float16:
        raise ValueError(f"RGB ready batch must be float16, got {batch.dtype}")


def materialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    contiguous = getattr(value, "contiguous", None)
    if callable(contiguous):
        return contiguous()
    if isinstance(value, (tuple, list)):
        return type(value)(materialize(item) for item in value)
    return value


def _batch_size(batch: Any) -> int:
    try:
        size = len(batch)
    except TypeError as error:
        raise TypeError(f"batch type {type(batch).__name__} has no leading batch dimension") from error
    if size < 1:
        raise ValueError("batch must not be empty")
    return size


def _seed_worker(_worker_id: int) -> None:
    import torch

    seed = cast("int", torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party transforms consume NumPy's process-global RNG.


def _collate_then_transform(samples: Sequence[Any], *, transform: Callable[[Any], Any]) -> Any:
    return transform(_collate_images(samples))


def _collate_images(samples: Sequence[Any]) -> Any:
    """Stack image samples into fresh storage before DataLoader transfers it."""
    if not samples:
        raise ValueError("cannot collate an empty image batch")

    import torch
    from torch.utils.data._utils.collate import default_collate

    first = samples[0]
    if isinstance(first, (np.ndarray, torch.Tensor)):
        return torch.stack([torch.as_tensor(sample) for sample in samples])
    return default_collate(samples)
