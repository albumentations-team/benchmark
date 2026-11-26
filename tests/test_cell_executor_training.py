from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from augbench import cell_executor
from augbench.run_records import CellKey, GpuMemory, OutputObservation, Throughput

if TYPE_CHECKING:
    from augbench.adapters.registry import AdapterRegistry


class _Monitor:
    events: list[str]

    def __init__(self) -> None:
        self.steady_peak_mib = 700.0

    def start(self) -> None:
        self.events.append("start")

    def mark_steady(self) -> None:
        self.events.append("mark_steady")

    def stop(self) -> GpuMemory:
        self.events.append("stop")
        return GpuMemory(peak_mib=900.0, baseline_mib=100.0, poll_interval_ms=50, valid_samples=4)


class _Session:
    model_name = "resnet18"

    def __init__(self, events: list[str], *, finite: bool = True) -> None:
        self._events = events
        self._finite = finite

    def reset(self, seed: int) -> None:
        self._events.append(f"reset:{seed}")

    def loss_is_finite(self) -> bool:
        return self._finite


class _Executor(cell_executor.CellExecutor):
    """The real ``execute`` with every collaborator that needs a GPU replaced."""

    def __init__(self, events: list[str], session: _Session | None) -> None:
        adapter = SimpleNamespace(build_recipe=lambda _recipe: SimpleNamespace())
        self._events = events
        self._session = session
        self._adapters = cast("AdapterRegistry", SimpleNamespace(load=lambda _implementation: adapter))

    def _recipe(self, _recipe_id: str) -> Any:
        return SimpleNamespace(supported_implementations=("pillow_cpu",))

    def _sources_for_seed(self, _seed: int) -> tuple[Any, ...]:
        return ()

    def _training_session(self) -> Any:
        return self._session

    def _measure_ceiling(self, _session: Any, _seed: int) -> float:
        self._events.append("ceiling")
        return 321.0

    def _execute_dataloader(self, *arguments: Any) -> tuple[Throughput, OutputObservation]:
        self._events.append("loader")
        training = arguments[-1]
        if training is not None:
            training.waits.extend([0.25, 0.5])
        return Throughput(completed_items=100, duration_seconds=1.0), OutputObservation(shape=(256, 3, 224, 224))


def _executor(monkeypatch: pytest.MonkeyPatch, events: list[str], session: _Session | None) -> _Executor:
    monkeypatch.setattr(cell_executor, "NvmlProcessMemoryMonitor", type("Monitor", (_Monitor,), {"events": events}))
    monkeypatch.setattr(cell_executor, "_clear_device_state", lambda: None)
    monkeypatch.setattr(cell_executor, "_seed_everything", lambda _seed: None)
    monkeypatch.setattr(cell_executor, "_runtime_metadata", lambda _adapter: {"library_version": "1"})
    return _Executor(events, session)


def _cell() -> CellKey:
    return CellKey(run_id="a" * 64, family="imagenet_train", implementation="pillow_cpu", recipe_id="Resize", seed=137)


def test_training_cell_measures_the_ceiling_before_the_loader_and_records_the_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    executor = _executor(monkeypatch, events, _Session(events))

    result = executor.execute(_cell())

    assert events == ["start", "ceiling", "mark_steady", "reset:137", "loader", "stop"]
    assert result.training is not None
    assert result.training.model == "resnet18"
    assert result.training.ceiling_images_per_second == 321.0
    assert result.training.loader_wait_seconds == 0.75
    assert result.training.steady_peak_mib == 700.0
    assert result.gpu_memory.peak_mib == 900.0


def test_input_pipeline_cell_has_no_training_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    executor = _executor(monkeypatch, events, None)

    result = executor.execute(_cell().model_copy(update={"family": "rgb"}))

    assert events == ["start", "loader", "stop"]
    assert result.training is None


def test_a_non_finite_loss_fails_the_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    executor = _executor(monkeypatch, events, _Session(events, finite=False))

    with pytest.raises(RuntimeError, match="non-finite"):
        executor.execute(_cell())
