from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.transforms.kornia_unstable import KORNIA_BENCHMARK_EXCLUDED_NAMES
from benchmark.transforms.video_recipe_specs import (
    is_supported_by_library,
    recipe_augmentation_specs,
    recipe_name,
    spec_by_name,
)


def test_video_recipe_names_are_pipeline_recipes() -> None:
    names = {recipe_name(spec) for spec in recipe_augmentation_specs()}

    assert "RandomCrop224+HorizontalFlip+Normalize+ToTensor" in names
    assert "RandomCrop224+Normalize+ToTensor" in names
    assert "Normalize+Normalize+ToTensor" not in names


def test_albumentationsx_video_pipeline_returns_tensor_clip() -> None:
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")
    from benchmark.transforms import albumentationsx_video_pipeline_impl as impl

    transform = _transform_by_name(impl.TRANSFORMS, "RandomCrop224+HorizontalFlip+Normalize+ToTensor")
    video = np.zeros((4, 256, 256, 3), dtype=np.uint8)

    output = impl.__call__(transform, video)

    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (4, 3, 224, 224)
    assert output.dtype == torch.float32


def test_torchvision_video_pipeline_registers_recipe_transforms() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from benchmark.transforms import torchvision_video_pipeline_impl as impl

    names = {entry["name"] for entry in impl.TRANSFORMS}

    assert "RandomCrop224+HorizontalFlip+Normalize+ToTensor" in names
    assert "RandomCrop224+JpegCompression+Normalize+ToTensor" in names


def test_pytorchvideo_pipeline_registers_canonical_recipe() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from benchmark.transforms import pytorchvideo_pipeline_impl as impl

    assert impl.LIBRARY == "pytorchvideo"
    assert [entry["name"] for entry in impl.TRANSFORMS] == ["PyTorchVideoCanonical+Normalize+ToTensor"]


def test_pytorchvideo_pipeline_returns_fixed_shape_cthw_tensor() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from benchmark.transforms import pytorchvideo_pipeline_impl as impl

    transform = _transform_by_name(impl.TRANSFORMS, "PyTorchVideoCanonical+Normalize+ToTensor")
    video = torch.zeros((3, 16, 256, 256), dtype=torch.uint8)

    output = impl.__call__(transform, video)

    assert tuple(output.shape) == (3, 16, 224, 224)
    assert output.dtype == torch.float32


def test_dali_video_pipeline_uses_recipe_names_and_records_unsupported() -> None:
    from benchmark.adapters.dali_video import run_dali_video_transform
    from benchmark.dali_pipeline_worker import _dali_transforms_from_specs

    transforms = _dali_transforms_from_specs(media="video", transforms_filter=("HorizontalFlip", "Elastic"))
    by_name = {str(entry["name"]): entry for entry in transforms}

    assert "RandomCrop224+HorizontalFlip+Normalize+ToTensor" in by_name
    assert "RandomCrop224+Elastic+Normalize+ToTensor" in by_name

    result = run_dali_video_transform(
        transform_name="RandomCrop224+Elastic+Normalize+ToTensor",
        params=by_name["RandomCrop224+Elastic+Normalize+ToTensor"]["transform"],
        paths=[Path("a.mp4")],
        clip_length=16,
        batch_size=1,
        num_runs=1,
        workers=1,
    )

    assert result["supported"] is False
    assert "does not implement transform 'Elastic'" in result["reason"]


def test_dali_video_pipeline_records_invalid_reader_backend_as_unsupported() -> None:
    from benchmark.adapters.dali_video import run_dali_video_transform

    result = run_dali_video_transform(
        transform_name="RandomCrop224+HorizontalFlip+Normalize+ToTensor",
        params={"name": "HorizontalFlip", "params": {}},
        paths=[Path("a.mp4")],
        clip_length=16,
        batch_size=1,
        num_runs=1,
        workers=1,
        reader_backend="old.video_reader",
    )

    assert result["supported"] is False
    assert "Unsupported DALI video reader backend" in result["reason"]


def test_dali_experimental_reader_does_not_receive_dtype() -> None:
    from benchmark.adapters.dali_video import _read_video_batch

    calls: dict[str, dict[str, object]] = {}

    class StableReaders:
        @staticmethod
        def video(**kwargs: object) -> object:
            calls["stable"] = kwargs
            return object()

    class ExperimentalReaders:
        @staticmethod
        def video(**kwargs: object) -> object:
            calls["experimental"] = kwargs
            return object()

    class Experimental:
        readers = ExperimentalReaders()

    class FakeFn:
        readers = StableReaders()
        experimental = Experimental()

    class FakeTypes:
        RGB = "rgb"
        UINT8 = "uint8"

    _read_video_batch(
        FakeFn(),
        FakeTypes(),
        paths=[Path("a.mp4")],
        clip_length=16,
        reader_backend="experimental.readers.video",
    )
    _read_video_batch(
        FakeFn(),
        FakeTypes(),
        paths=[Path("a.mp4")],
        clip_length=16,
        reader_backend="readers.video",
    )

    assert "dtype" not in calls["experimental"]
    assert calls["stable"]["dtype"] == "uint8"


def test_kornia_video_pipeline_keeps_shear_for_runtime_classification() -> None:
    pytest.importorskip("kornia")
    from benchmark.transforms import kornia_video_pipeline_impl as impl

    names = {entry["name"] for entry in impl.TRANSFORMS}

    assert "RandomCrop224+Shear+Normalize+ToTensor" in names


def test_kornia_video_pipeline_uses_per_clip_randomness() -> None:
    pytest.importorskip("kornia")
    from benchmark.transforms import kornia_video_pipeline_impl as impl

    transform = _transform_by_name(impl.TRANSFORMS, "RandomCrop224+HorizontalFlip+Normalize+ToTensor")
    same_on_batch_values = _random_same_on_batch_values(transform)

    assert same_on_batch_values
    assert same_on_batch_values == [False] * len(same_on_batch_values)


def test_kornia_video_micro_uses_per_clip_randomness(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_TRANSFORMS_FILTER", "HorizontalFlip")
    pytest.importorskip("kornia")
    from benchmark.transforms import kornia_video_impl as impl

    transform = _transform_by_name(impl.build_transforms("kornia", media="video"), "HorizontalFlip")
    same_on_batch_values = _random_same_on_batch_values(transform)

    assert same_on_batch_values
    assert same_on_batch_values == [False] * len(same_on_batch_values)


def test_kornia_video_uses_official_video_sequential_api() -> None:
    source = Path("benchmark/transforms/kornia_video_impl.py").read_text(encoding="utf-8")
    pipeline_source = Path("benchmark/transforms/kornia_video_pipeline_impl.py").read_text(encoding="utf-8")

    assert "Kaug.VideoSequential" in source
    assert 'data_format="BTCHW"' in source
    assert "same_on_frame=True" in source
    assert "wrap_video_transform(transform)" in source
    assert "KorniaVideoSequential" in pipeline_source


def test_kornia_video_adapter_uses_float32_parameters_and_contiguous_inputs() -> None:
    source = Path("benchmark/transforms/kornia_video_impl.py").read_text(encoding="utf-8")

    assert "torch.float16" not in source
    assert "dtype=torch.float32" in source
    assert ".to(device).contiguous()" in source
    assert "clip_limit=clip" in source
    assert 'value=float(params["fill"])' in source
    assert 'bits=float(params["bits"])' in source
    assert "brightness=(2.0, 2.0)" in source


def test_kornia_video_problematic_dtype_rows_smoke_on_tiny_cpu_clip(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_TRANSFORMS_FILTER", "CLAHE,Erasing,Posterize,Snow,Affine")
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from benchmark.transforms import kornia_video_impl as impl

    names = {"CLAHE", "Erasing", "Posterize", "Snow", "Affine"}
    transforms = [entry for entry in impl.TRANSFORMS if entry["name"] in names]
    video = torch.zeros((2, 3, 64, 64), dtype=torch.float32)

    assert {entry["name"] for entry in transforms} == names
    for entry in transforms:
        output = impl.__call__(entry["transform"], video)
        assert output.dtype == torch.float32
        assert output.device == impl.device
        assert output.is_contiguous()


def test_torchvision_video_jpeg_is_cpu_only_when_cuda_is_requested() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from benchmark.transforms import torchvision_video_impl as impl

    transform = impl.create_transform(spec_by_name("JpegCompression"))
    assert transform is not None
    cpu_video = torch.zeros((2, 3, 16, 16), dtype=torch.uint8)
    cpu_output = impl.__call__(transform, cpu_video)
    assert cpu_output.device.type == "cpu"

    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="CPU tensor"):
            impl.__call__(transform, cpu_video.to("cuda"))


def test_albumentationsx_video_pipeline_keeps_shear_in_shared_rows() -> None:
    pytest.importorskip("albumentations")
    from benchmark.transforms import albumentationsx_video_pipeline_impl as impl

    names = {entry["name"] for entry in impl.TRANSFORMS}

    assert "RandomCrop224+Shear+Normalize+ToTensor" in names


def test_shear_video_dataloader_stays_in_shared_rows() -> None:
    shear = spec_by_name("Shear")
    assert is_supported_by_library(shear, "albumentationsx")
    assert is_supported_by_library(shear, "kornia")
    assert shear in recipe_augmentation_specs()


def test_kornia_unstable_video_pipeline_transforms_are_unsupported() -> None:
    for name in KORNIA_BENCHMARK_EXCLUDED_NAMES:
        assert not is_supported_by_library(spec_by_name(name), "kornia")


def _transform_by_name(transforms: list[dict[str, Any]], name: str) -> Any:
    for entry in transforms:
        if entry["name"] == name:
            return entry["transform"]
    raise AssertionError(f"Missing transform {name!r}")


def _random_same_on_batch_values(transform: Any) -> list[bool]:
    return [
        module.same_on_batch
        for module in transform.modules()
        if hasattr(module, "same_on_batch")
        and module.__class__.__name__ != "Normalize"
        and module.same_on_batch is not None
    ]
