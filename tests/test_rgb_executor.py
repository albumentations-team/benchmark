from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest

from augbench import rgb_executor
from augbench.adapters.dali.common import DaliAdapter
from augbench.recipes.load import load_recipe_catalog
from augbench.run_config import load_family_config
from augbench.run_records import CellKey


@pytest.mark.parametrize(
    ("implementation", "expected_events"),
    [
        (
            "dali_gpu",
            [
                "empty_cache",
                "synchronize",
                "release_dali_pool",
                "open_graph",
                "synchronize",
                "close_graph",
                "empty_cache",
                "synchronize",
                "release_dali_pool",
                "sample_baseline",
            ],
        ),
        ("pillow_cpu", ["empty_cache", "synchronize", "sample_baseline"]),
    ],
)
def test_cell_cleans_device_state_before_sampling_baseline(
    implementation: str,
    expected_events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def sample_baseline() -> Never:
        events.append("sample_baseline")
        raise RuntimeError("baseline sampled")

    cuda = SimpleNamespace(
        empty_cache=lambda: events.append("empty_cache"),
        synchronize=lambda: events.append("synchronize"),
    )
    backend = SimpleNamespace(ReleaseUnusedMemory=lambda: events.append("release_dali_pool"))
    dali = SimpleNamespace(backend=backend) if implementation == "dali_gpu" else None
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, Tensor=SimpleNamespace, float16="float16"))
    monkeypatch.setitem(sys.modules, "nvidia.dali", dali)
    monkeypatch.setattr(rgb_executor, "_seed_everything", lambda _seed: None)
    monkeypatch.setattr(rgb_executor, "NvmlProcessMemoryMonitor", lambda: SimpleNamespace(start=sample_baseline))

    root = Path(__file__).resolve().parents[1]
    config = load_family_config(root / "configs/families/rgb.yaml")
    executor = rgb_executor.RGBCellExecutor(
        config=config,
        recipes=load_recipe_catalog(root / config.recipes),
        dataset_files=(Path("image.jpg"),) * config.dataset.item_count,
    )
    cell = CellKey(run_id="a" * 64, family="rgb", implementation=implementation, recipe_id="Resize224", seed=137)

    if implementation == "dali_gpu":
        batch = SimpleNamespace(
            shape=(config.execution.batch_size, *config.output.shape),
            device=SimpleNamespace(type="cuda"),
            dtype="float16",
        )
        source = SimpleNamespace(
            open=lambda: events.append("open_graph"),
            next_batch=lambda: (batch, None),
            close=lambda: events.append("close_graph"),
        )
        monkeypatch.setattr(DaliAdapter, "native_source", lambda _adapter, **_kwargs: source)
        executor.preflight(cell)

    with pytest.raises(RuntimeError, match=r"^baseline sampled$"):
        executor.execute(cell)

    assert events == expected_events
