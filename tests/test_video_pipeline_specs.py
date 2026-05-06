from __future__ import annotations

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


def test_kornia_video_pipeline_shear_is_not_registered() -> None:
    pytest.importorskip("kornia")
    from benchmark.transforms import kornia_video_pipeline_impl as impl

    names = {entry["name"] for entry in impl.TRANSFORMS}

    assert "RandomCrop224+Shear+Normalize+ToTensor" not in names


def test_albumentationsx_video_pipeline_shear_is_not_registered() -> None:
    pytest.importorskip("albumentations")
    from benchmark.transforms import albumentationsx_video_pipeline_impl as impl

    names = {entry["name"] for entry in impl.TRANSFORMS}

    assert "RandomCrop224+Shear+Normalize+ToTensor" not in names


def test_shear_video_dataloader_requires_two_libraries() -> None:
    """AlbumentationsX implements Shear, but Kornia video pipeline does not → <2 libs → no shared recipe row."""
    shear = spec_by_name("Shear")
    assert is_supported_by_library(shear, "albumentationsx")
    assert not is_supported_by_library(shear, "kornia")
    assert shear not in recipe_augmentation_specs()


def test_kornia_unstable_video_pipeline_transforms_are_unsupported() -> None:
    for name in KORNIA_BENCHMARK_EXCLUDED_NAMES:
        assert not is_supported_by_library(spec_by_name(name), "kornia")


def _transform_by_name(transforms: list[dict[str, Any]], name: str) -> Any:
    for entry in transforms:
        if entry["name"] == name:
            return entry["transform"]
    raise AssertionError(f"Missing transform {name!r}")
