"""Tests for pyperf micro benchmark helpers."""

from __future__ import annotations

import argparse
import json
import time
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pyperf")

from benchmark.pyperf_micro_runner import _merge_pyperf_payload, _preflight_slow_transform
from benchmark.runner import MediaType

if TYPE_CHECKING:
    from pathlib import Path


def test_preflight_slow_transform_returns_visible_skip_payload() -> None:
    args = argparse.Namespace(
        disable_slow_skip=False,
        slow_threshold_sec_per_item=0.001,
        slow_preflight_items=1,
    )

    def call_fn(_transform: Any, item: Any) -> Any:
        time.sleep(0.01)
        return item

    result = _preflight_slow_transform(
        transform=object(),
        transform_name="SlowTransform",
        media=[object()],
        call_fn=call_fn,
        media_type=MediaType.IMAGE,
        args=args,
    )

    assert result is not None
    assert result["early_stopped"] is True
    assert result["num_successful_runs"] == 0
    assert result["slow_marker"] == "<1000 img/s"
    assert "SlowTransform slower than threshold" in result["early_stop_reason"]


def test_merge_pyperf_payload_allows_missing_file_for_slow_skips(tmp_path: Path) -> None:
    combined_pyperf: dict[str, object] = {"benchmarks": []}

    _merge_pyperf_payload(combined_pyperf, tmp_path / "SlowTransform.pyperf.json")

    assert combined_pyperf == {"benchmarks": []}


def test_merge_pyperf_payload_appends_existing_benchmarks(tmp_path: Path) -> None:
    pyperf_path = tmp_path / "FastTransform.pyperf.json"
    pyperf_path.write_text(
        json.dumps({"benchmarks": [{"metadata": {"name": "FastTransform"}}], "metadata": {"host": "vm"}}),
        encoding="utf-8",
    )
    combined_pyperf: dict[str, object] = {"benchmarks": []}

    _merge_pyperf_payload(combined_pyperf, pyperf_path)

    assert combined_pyperf == {
        "benchmarks": [{"metadata": {"name": "FastTransform"}}],
        "metadata": {"host": "vm"},
    }
