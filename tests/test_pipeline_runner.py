from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

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


def test_gpu_image_loader_path_splits_cpu_prep_and_gpu_batch_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch.utils.data")
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    loader_transforms: list[object | None] = []
    batch_transforms: list[object] = []

    class CpuPrep:
        def __call__(self, _item: object) -> np.ndarray:
            return np.zeros((3, 4, 4), dtype=np.float32)

    class SplitTransform:
        cpu_transform = CpuPrep()
        gpu_transform = object()

    transform = SplitTransform()
    runner = PipelineBenchmarkRunner(
        library="kornia",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[{"name": "Identity", "transform": transform}],
        call_fn=lambda transform_arg, item: transform_arg(item),
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

    def fake_resolved_device() -> str:
        runner._last_device = "cuda"
        return "cuda"

    monkeypatch.setattr(runner, "_resolved_device", fake_resolved_device)

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
    assert loader_transforms == [transform.cpu_transform, transform.cpu_transform]
    assert batch_transforms == [transform.gpu_transform]


def test_torchvision_gpu_image_pipeline_uses_split_recipe(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class CpuPrep(nn.Module):
        def forward(self, image: Any) -> Any:
            return image[..., :4, :4]

    class GpuTransform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_shapes: list[tuple[int, ...]] = []

        def forward(self, batch: Any) -> Any:
            self.batch_shapes.append(tuple(batch.shape))
            return batch

    gpu_transform = GpuTransform()
    transform = nn.Module()
    transform.cpu_transform = CpuPrep()
    transform.gpu_transform = gpu_transform
    runner = PipelineBenchmarkRunner(
        library="torchvision",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[],
        call_fn=lambda transform_arg, item: transform_arg(item),
        media="image",
        scenario="image-rgb",
        device="cuda",
    )

    dataset_transform, batch_transform = runner._split_gpu_image_transform(transform)
    batch = torch.ones((2, 3, 6, 6), dtype=torch.float32)

    assert dataset_transform is not None
    assert tuple(cast("Any", dataset_transform)(batch[0]).shape) == (3, 4, 4)
    assert batch_transform is gpu_transform


def test_torchvision_gpu_batch_recipe_applies_measured_transform_per_sample() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch import nn

    from benchmark.transforms.torchvision_pipeline_impl import _GpuBatchRecipe

    class MeasuredTransform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_shapes: list[tuple[int, ...]] = []

        def forward(self, image: Any) -> Any:
            self.input_shapes.append(tuple(image.shape))
            return image + 1

    measured_transform = MeasuredTransform()
    recipe = _GpuBatchRecipe(measured_transform)
    batch = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)

    output = recipe(batch)

    assert measured_transform.input_shapes == [(3, 4, 4), (3, 4, 4)]
    assert tuple(output.shape) == (2, 3, 4, 4)
    assert output.dtype == torch.float32


def test_gpu_image_loader_uses_cpu_prep_before_default_collate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    runner = PipelineBenchmarkRunner(
        library="kornia",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[],
        call_fn=lambda transform_arg, item: transform_arg(item),
        media="image",
        scenario="image-rgb",
        num_items=2,
        batch_size=2,
        workers=0,
        device="cuda",
    )

    def fake_resolved_device() -> str:
        runner._last_device = "cuda"
        return "cuda"

    monkeypatch.setattr(runner, "_resolved_device", fake_resolved_device)
    preloaded = [
        torch.ones((3, 4, 5), dtype=torch.float32),
        torch.ones((3, 2, 7), dtype=torch.float32),
    ]

    def cpu_prep(_item: object) -> object:
        return torch.zeros((3, 4, 4), dtype=torch.float32)

    loader = runner._loader(paths, transform=cpu_prep, preloaded=preloaded)
    batch = next(iter(loader))

    assert tuple(batch.shape) == (2, 3, 4, 4)


def test_gpu_image_batch_move_preserves_uint8_for_torchvision_transforms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    runner = PipelineBenchmarkRunner(
        library="torchvision",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[],
        call_fn=lambda transform_arg, item: transform_arg(item),
        media="image",
        scenario="image-rgb",
        device="cuda",
    )
    monkeypatch.setattr(runner, "_resolved_device", lambda: None)

    batch = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)

    moved = runner._move_batch_to_resolved_device(batch, scale_uint8=False)

    assert moved.dtype == torch.uint8


def test_torchvision_gpu_batch_recipe_keeps_uint8_until_measured_transform() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch import nn

    from benchmark.transforms.torchvision_pipeline_impl import _GpuBatchRecipe

    class Uint8OnlyTransform(nn.Module):
        def forward(self, image: Any) -> Any:
            if image.dtype != torch.uint8:
                msg = "Input tensor dtype should be uint8"
                raise RuntimeError(msg)
            return image

    recipe = _GpuBatchRecipe(Uint8OnlyTransform())
    batch = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)

    output = recipe(batch)

    assert output.dtype == torch.float32


def test_pipeline_main_filters_torchvision_gpu_jpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark import pipeline_runner

    spec_file = tmp_path / "spec.py"
    spec_file.write_text(
        """
LIBRARY = 'torchvision'
def __call__(transform, image):
    return transform(image)
TRANSFORMS = [
    {'name': 'RandomCrop224+Resize+Normalize+ToTensor', 'transform': object()},
    {'name': 'RandomCrop224+JpegCompression+Normalize+ToTensor', 'transform': object()},
]
""",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self) -> None:
            return None

    monkeypatch.setattr(pipeline_runner, "PipelineBenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pipeline_runner",
            "--specs-file",
            str(spec_file),
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.json"),
            "--media",
            "image",
            "--scenario",
            "image-rgb",
            "--device",
            "cuda",
        ],
    )

    pipeline_runner.main()

    assert [transform["name"] for transform in captured["transforms"]] == [
        "RandomCrop224+Resize+Normalize+ToTensor",
    ]


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


def test_pipeline_runner_records_cuda_peak_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch.utils.data")
    from benchmark import pipeline_runner

    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    reset_calls: list[str | None] = []

    runner = PipelineBenchmarkRunner(
        library="testlib",
        data_dir=tmp_path,
        output_file=tmp_path / "pipeline.json",
        transforms=[{"name": "Identity", "transform": lambda item: item}],
        call_fn=lambda transform_arg, item: transform_arg(item),
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
    monkeypatch.setattr(runner, "_load_item", lambda _path: np.zeros((3, 4, 4), dtype=np.uint8))

    def fake_resolved_device() -> str:
        runner._last_device = "cuda"
        return "cuda"

    monkeypatch.setattr(runner, "_resolved_device", fake_resolved_device)
    monkeypatch.setattr(pipeline_runner, "_torch_synchronize", lambda _device=None: None)
    monkeypatch.setattr(pipeline_runner, "cuda_memory_allocated", lambda _device: 128)

    def fake_reset(device: str | None) -> None:
        reset_calls.append(device)

    monkeypatch.setattr(pipeline_runner, "reset_peak_memory_stats", fake_reset)
    monkeypatch.setattr(
        pipeline_runner,
        "cuda_memory_stats",
        lambda _device: {
            "gpu_memory_allocated_before_bytes": None,
            "gpu_memory_allocated_after_bytes": 256,
            "gpu_peak_memory_allocated_bytes": 1024,
            "gpu_peak_memory_reserved_bytes": 2048,
        },
    )

    payload = runner.run()

    gpu_memory = payload["results"]["Identity"]["gpu_memory"]
    assert reset_calls == ["cuda"]
    assert gpu_memory["measured"] is True
    assert gpu_memory["peak_allocated_bytes"] == 1024
    assert gpu_memory["peak_reserved_bytes"] == 2048
    assert gpu_memory["runs"][0]["gpu_memory_allocated_before_bytes"] == 128
    assert payload["metadata"]["benchmark_params"]["gpu_memory_peak_measured"] is True


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
