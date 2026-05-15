"""Tests for pyperf micro benchmark helpers."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pytest

pytest.importorskip("pyperf")

from benchmark.devices import DeviceUnavailableError
from benchmark.pyperf_micro_runner import (
    _load_media,
    _make_micro_output_contiguous,
    _merge_pyperf_payload,
    _merge_transform_payload,
    _preflight_slow_transform,
    _prepare_device_media,
    _pyperf_value_throughputs,
    _run_filtered_transforms,
)
from benchmark.runner import MediaType

if TYPE_CHECKING:
    from pathlib import Path


class _FakePyperfRunner:
    def __init__(self) -> None:
        self.args = argparse.Namespace(worker=False, values=1)

    def bench_time_func(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("bench_time_func should not run in this test")


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


def test_prepare_device_media_moves_tensor_like_samples() -> None:
    class TensorLike:
        def __init__(self) -> None:
            self.device: str | None = None

        def to(self, device: str) -> TensorLike:
            self.device = device
            return self

    sample = TensorLike()
    args = argparse.Namespace(resolved_device="cuda")

    assert _prepare_device_media(args, [sample]) == [sample]
    assert sample.device == "cuda"


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


def test_load_media_passes_clip_length_to_video_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}

    class FakeMediaLoader:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def load(self) -> list[object]:
            return [object()]

    monkeypatch.setattr("benchmark.pyperf_micro_runner.BenchmarkMediaLoader", FakeMediaLoader)
    args = argparse.Namespace(
        data_dir=tmp_path,
        media="video",
        num_items=2,
        num_channels=3,
        clip_length=8,
    )

    media = _load_media(args, "kornia")

    assert len(media) == 1
    assert seen["clip_length"] == 8


def test_preflight_exception_records_unsupported_result(tmp_path: Path) -> None:
    def broken_call(_transform: Any, _item: Any) -> Any:
        raise RuntimeError("bad dtype")

    media_cache = tmp_path / "media.pkl"
    media_cache.write_bytes(pickle.dumps([object()]))
    args = argparse.Namespace(
        media="video",
        media_cache=media_cache,
        data_dir=tmp_path,
        num_items=1,
        num_channels=3,
        clip_length=16,
        scenario="video-16f",
        disable_slow_skip=False,
        slow_threshold_sec_per_item=None,
        slow_preflight_items=None,
        json_output=tmp_path / "out.json",
        device="none",
    )
    _run_filtered_transforms(
        runner=cast("Any", _FakePyperfRunner()),
        args=args,
        library="torchvision",
        call_fn=broken_call,
        transforms=[{"name": "JpegCompression", "transform": object()}],
    )

    output = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    result = output["results"]["JpegCompression"]
    assert result["supported"] is False
    assert "bad dtype" in result["reason"]


def test_cuda_unavailable_records_unsupported_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_cache = tmp_path / "media.pkl"
    media_cache.write_bytes(pickle.dumps([object()]))
    args = argparse.Namespace(
        media="image",
        media_cache=media_cache,
        data_dir=tmp_path,
        num_items=1,
        num_channels=3,
        clip_length=16,
        scenario="image-rgb",
        disable_slow_skip=False,
        slow_threshold_sec_per_item=None,
        slow_preflight_items=None,
        json_output=tmp_path / "out.json",
        device="cuda",
    )

    def unavailable(_device: str) -> str | None:
        raise DeviceUnavailableError("Requested --device cuda, but CUDA is not available")

    monkeypatch.setattr("benchmark.pyperf_micro_runner.resolve_device", unavailable)

    _run_filtered_transforms(
        runner=cast("Any", _FakePyperfRunner()),
        args=args,
        library="torchvision",
        call_fn=lambda _transform, item: item,
        transforms=[{"name": "Resize", "transform": object()}],
    )

    output = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert output["results"]["Resize"]["supported"] is False
    assert "CUDA is not available" in output["results"]["Resize"]["reason"]


def test_pyperf_main_keeps_torchvision_gpu_jpeg_for_runtime_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark import pyperf_micro_runner

    spec_file = tmp_path / "spec.py"
    spec_file.write_text(
        """
LIBRARY = 'torchvision'
def __call__(transform, image):
    return transform(image)
TRANSFORMS = [
    {'name': 'Resize', 'transform': object()},
    {'name': 'JpegCompression', 'transform': object()},
]
""",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class FakeRunner:
        args = type("Args", (), {"worker": False, "processes": 1, "values": 1, "warmups": 1, "min_time": 0.001})()

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def parse_args(self) -> object:
            return argparse.Namespace(
                specs_file=spec_file,
                data_dir=tmp_path,
                json_output=tmp_path / "out.json",
                media="image",
                scenario="image-rgb",
                num_channels=3,
                clip_length=16,
                device="cuda",
                transforms="",
                media_cache=tmp_path / "media.pkl",
                num_items=1,
                slow_threshold_sec_per_item=None,
                slow_preflight_items=None,
                disable_slow_skip=False,
            )

    def fake_run_transform_subprocesses(**kwargs: Any) -> None:
        captured.update(kwargs)

    pyperf_module = pytest.importorskip("pyperf")
    monkeypatch.setattr(pyperf_module, "Runner", FakeRunner)
    monkeypatch.setattr(pyperf_micro_runner, "_run_transform_subprocesses", fake_run_transform_subprocesses)

    pyperf_micro_runner.main()

    assert [transform["name"] for transform in captured["transforms"]] == ["Resize", "JpegCompression"]
