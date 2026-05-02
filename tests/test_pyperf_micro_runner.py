"""Tests for pyperf micro benchmark helpers."""

from __future__ import annotations

import argparse
import json
import time
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pytest

pytest.importorskip("pyperf")

from benchmark.pyperf_micro_runner import (
    _make_micro_output_contiguous,
    _merge_pyperf_payload,
    _merge_transform_payload,
    _preflight_slow_transform,
    _pyperf_value_throughputs,
)
from benchmark.runner import MediaType

if TYPE_CHECKING:
    from pathlib import Path


def test_pyperf_value_throughputs_use_normalized_per_item_times() -> None:
    assert _pyperf_value_throughputs([0.25, 0.5, 0.0]) == [4.0, 2.0]


def test_make_micro_output_contiguous_copies_numpy_views() -> None:
    output = np.zeros((4, 4, 3), dtype=np.uint8)[:, ::-1]

    contiguous = _make_micro_output_contiguous(output)

    assert contiguous.flags.c_contiguous
    assert contiguous.shape == output.shape


def test_make_micro_output_contiguous_converts_pillow_images() -> None:
    pil_image = pytest.importorskip("PIL.Image")
    image = pil_image.new("RGB", (4, 3))

    output = _make_micro_output_contiguous(image)

    assert isinstance(output, np.ndarray)
    assert output.flags.c_contiguous
    assert output.shape == (3, 4, 3)


def test_make_micro_output_contiguous_calls_tensor_contiguous() -> None:
    class TensorLike:
        def __init__(self) -> None:
            self.called = False

        def contiguous(self) -> Self:
            self.called = True
            return self

    output = TensorLike()

    assert _make_micro_output_contiguous(output) is output
    assert output.called


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
    assert result["slow_marker"] == "≤1000 img/s"
    assert "SlowTransform slower than threshold" in result["early_stop_reason"]
    assert ">=" in result["early_stop_reason"]


def test_merge_pyperf_payload_allows_missing_file_for_slow_skips(tmp_path: Path) -> None:
    combined_pyperf: dict[str, object] = {"benchmarks": []}

    _merge_pyperf_payload(combined_pyperf, tmp_path / "SlowTransform.pyperf.json")

    assert combined_pyperf == {"benchmarks": []}


def test_merge_transform_payload_keeps_first_transform_result() -> None:
    first_payload = {
        "metadata": {"library": "albumentationsx"},
        "results": {
            "Resize": {
                "supported": True,
                "median_throughput": 100.0,
            },
        },
    }
    second_payload = {
        "metadata": {"library": "albumentationsx"},
        "results": {
            "HorizontalFlip": {
                "supported": True,
                "median_throughput": 200.0,
            },
        },
    }

    payload = _merge_transform_payload(None, first_payload)
    payload = _merge_transform_payload(payload, second_payload)

    assert payload["metadata"] == {"library": "albumentationsx"}
    assert payload["results"] == {
        "Resize": {
            "supported": True,
            "median_throughput": 100.0,
        },
        "HorizontalFlip": {
            "supported": True,
            "median_throughput": 200.0,
        },
    }


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
