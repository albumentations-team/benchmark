from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from benchmark.pipeline_runner import PipelineBenchmarkRunner

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_pipeline_runner_executes_tiny_memory_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_file = tmp_path / "pipeline.json"
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    runner = PipelineBenchmarkRunner(
        library="testlib",
        data_dir=tmp_path,
        output_file=output_file,
        transforms=[{"name": "Identity", "transform": lambda item: item}],
        call_fn=lambda transform, item: transform(item),
        media="image",
        scenario="image-rgb",
        num_items=2,
        num_runs=1,
        batch_size=2,
        workers=0,
        min_time=0.0,
        min_batches=1,
        pipeline_scope="memory_dataloader_augment",
    )

    monkeypatch.setattr(runner, "_paths", lambda: paths)
    monkeypatch.setattr(runner, "_load_item", lambda _path: np.zeros((4, 4, 3), dtype=np.uint8))

    payload = runner.run()

    assert output_file.exists()
    assert payload["results"]["Identity"]["supported"] is True
    written = json.loads(output_file.read_text(encoding="utf-8"))
    assert written["metadata"]["benchmark_params"]["pipeline_scope"] == "memory_dataloader_augment"
    assert written["metadata"]["benchmark_params"]["num_images"] == 2


def test_pipeline_runner_resolves_none_device_without_torch(tmp_path: Path) -> None:
    runner = PipelineBenchmarkRunner(
        library="testlib",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[],
        call_fn=lambda _transform, item: item,
        media="image",
        scenario="image-rgb",
        device="none",
    )

    assert runner._resolved_device() is None
    assert runner._last_device is None


def test_pipeline_slow_preflight_uses_shared_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = PipelineBenchmarkRunner(
        library="testlib",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[],
        call_fn=lambda _transform, item: item,
        media="image",
        scenario="image-rgb",
    )

    monkeypatch.setattr(runner, "_load_item", lambda _path: object())

    threshold, preflight_items, max_preflight_secs = runner._slow_skip_config()

    assert threshold == 0.1
    assert preflight_items == 10
    assert max_preflight_secs == 60.0
