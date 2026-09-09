from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

from augbench import rgb_executor

if TYPE_CHECKING:
    import pytest


def test_preflight_uses_one_transient_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    arguments: dict[str, object] = {}

    class Dataset:
        def __init__(self, **_kwargs: object) -> None: ...

    class Loader:
        def __init__(self, **kwargs: object) -> None:
            arguments.update(kwargs)

        def open(self) -> None: ...

        def next_batch(self) -> str:
            return "batch"

        def close(self) -> None: ...

    class Consumer:
        last_validation: ClassVar[dict[str, list[int]]] = {"shape": [256, 3, 224, 224]}

        def __init__(self, **_kwargs: object) -> None: ...

        def __call__(self, _batch: str) -> int:
            return 256

        @staticmethod
        def synchronize() -> None: ...

    monkeypatch.setattr(rgb_executor, "RecipeDataset", Dataset)
    monkeypatch.setattr(rgb_executor, "TorchDataLoaderSource", Loader)
    monkeypatch.setattr(rgb_executor, "ReadyGpuBatchConsumer", Consumer)

    executor = object.__new__(rgb_executor.RGBCellExecutor)
    executor._config = SimpleNamespace(  # type: ignore[attr-defined]
        execution=SimpleNamespace(batch_size=256),
        output=SimpleNamespace(channels=3),
    )
    executor._output_observation = lambda _validation: None  # type: ignore[method-assign]

    rgb_executor.RGBCellExecutor._preflight_dataloader(
        executor,
        adapter=object(),
        runtime=SimpleNamespace(host_batch_transform=None),
        sources=(),
    )

    config = arguments["config"]
    assert config.num_workers == 1
    assert config.prefetch_factor == 1
    assert config.persistent_workers is False
