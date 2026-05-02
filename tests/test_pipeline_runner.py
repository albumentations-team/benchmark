from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmark.pipeline_runner import PipelineBenchmarkRunner

if TYPE_CHECKING:
    from pathlib import Path


def test_pipeline_runner_executes_tiny_memory_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch.utils.data")

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


def test_gpu_image_loader_path_defers_transform_until_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch.utils.data")
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    loader_transforms: list[object | None] = []
    batch_transforms: list[object] = []
    transform = object()
    runner = PipelineBenchmarkRunner(
        library="kornia",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[{"name": "Identity", "transform": transform}],
        call_fn=lambda _transform, _item: (_ for _ in ()).throw(RuntimeError("per-sample transform used")),
        media="image",
        scenario="image-rgb",
        num_items=2,
        num_runs=1,
        batch_size=2,
        workers=0,
        min_time=0.0,
        min_batches=1,
        pipeline_scope="memory_dataloader_augment",
        device="cuda",
    )

    monkeypatch.setattr(runner, "_paths", lambda: paths)
    monkeypatch.setattr(runner, "_preload_items", lambda _paths: [np.zeros((3, 4, 5), dtype=np.float32)] * 2)
    monkeypatch.setattr(runner, "_preflight_slow_transform", lambda **_kwargs: None)

    def skip_warm_loader(_loader: object, batch_transform: object | None = None) -> None:
        _ = batch_transform

    monkeypatch.setattr(runner, "_warm_loader_once", skip_warm_loader)
    monkeypatch.setattr(runner, "_resolved_device", lambda: "cuda")

    original_loader = runner._loader

    def capture_loader(
        paths_arg: list[Path],
        transform_arg: object | None,
        preloaded: list[object] | None = None,
    ) -> object:
        loader_transforms.append(transform_arg)
        return original_loader(paths_arg, transform_arg, preloaded)

    def capture_batch(_batch: object, transform_arg: object) -> int:
        batch_transforms.append(transform_arg)
        return 2

    monkeypatch.setattr(runner, "_loader", capture_loader)
    monkeypatch.setattr(runner, "_materialize_gpu_image_batch", capture_batch)

    result = runner._run_transform({"name": "Identity", "transform": transform}, paths)

    assert result["supported"] is True
    assert loader_transforms == [None, None]
    assert batch_transforms == [transform]


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

    assert threshold == 0.05
    assert preflight_items == 10
    assert max_preflight_secs == 60.0


def test_materialize_batch_counts_tensor_batch_dimension(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    runner = PipelineBenchmarkRunner(
        library="testlib",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[],
        call_fn=lambda _transform, item: item,
        media="image",
        scenario="image-rgb",
        pipeline_scope="decode_dataloader_augment",
    )

    assert runner._materialize_batch(torch.zeros((2, 3, 4, 5))) == 2


def test_pipeline_runner_rejects_container_recipe_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch.utils.data")
    output_file = tmp_path / "pipeline.json"
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    runner = PipelineBenchmarkRunner(
        library="testlib",
        data_dir=tmp_path,
        output_file=output_file,
        transforms=[{"name": "DictOutput", "transform": lambda item: {"image": item}}],
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

    result = payload["results"]["DictOutput"]
    assert result["supported"] is False
    assert "Pipeline recipes must return one fixed-shape tensor or ndarray per sample" in result["reason"]


def test_to_tensor_does_not_guess_image_layout(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    runner = PipelineBenchmarkRunner(
        library="albumentationsx",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[],
        call_fn=lambda _transform, item: item,
        media="image",
        scenario="image-rgb",
    )

    tensor = runner._to_tensor(np.zeros((2, 4, 5, 3), dtype=np.uint8))

    assert tuple(tensor.shape) == (2, 4, 5, 3)


def test_to_tensor_keeps_collated_chw_image_batch_shape(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    runner = PipelineBenchmarkRunner(
        library="torchvision",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[],
        call_fn=lambda _transform, item: item,
        media="image",
        scenario="image-rgb",
    )

    tensor = runner._to_tensor(np.zeros((2, 3, 4, 5), dtype=np.uint8))

    assert tuple(tensor.shape) == (2, 3, 4, 5)


def test_video_clip_loader_keeps_torchvision_uint8_for_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from benchmark import pipeline_runner

    class FakeDecodedClip:
        frames = np.zeros((4, 5, 6, 3), dtype=np.uint8)

    monkeypatch.setattr(pipeline_runner, "decode_video", lambda *_args, **_kwargs: FakeDecodedClip())

    tensor = pipeline_runner._video_clip_for_library(tmp_path / "video.mp4", "torchvision", 4)

    assert tuple(tensor.shape) == (4, 3, 5, 6)
    assert tensor.dtype == torch.uint8
