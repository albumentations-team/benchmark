from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PIL")

from PIL import Image

from benchmark.transforms.specs import TransformSpec


def _create_transform(spec: TransformSpec) -> Any | None:
    from benchmark.transforms.pillow_impl import create_transform

    return create_transform(spec)


def test_direct_resize_is_supported() -> None:
    transform = _create_transform(TransformSpec("Resize", {"target_size": 8, "interpolation": "nearest"}))
    assert transform is not None

    result = transform(Image.new("RGB", (4, 6)))

    assert result.size == (8, 8)


def test_direct_pad_is_supported() -> None:
    transform = _create_transform(TransformSpec("Pad", {"padding": 2, "fill": 0}))
    assert transform is not None

    result = transform(Image.new("RGB", (4, 6)))

    assert result.size == (8, 10)


def test_direct_transpose_is_supported() -> None:
    transform = _create_transform(TransformSpec("Transpose", {}))
    assert transform is not None

    result = transform(Image.new("RGB", (4, 6)))

    assert result.size == (6, 4)


def test_pillow_skips_non_native_or_composite_transforms() -> None:
    assert _create_transform(TransformSpec("RandomCrop128", {"height": 128, "width": 128})) is None
    assert _create_transform(TransformSpec("CenterCrop128", {"height": 128, "width": 128})) is None
    assert _create_transform(TransformSpec("RandomResizedCrop", {"size": (512, 512)})) is None
    assert _create_transform(TransformSpec("SquareSymmetry", {})) is None
    assert _create_transform(TransformSpec("RandomRotate90", {"times": (0, 3)})) is None
    assert _create_transform(TransformSpec("SafeRotate", {"limit": 45})) is None
    assert _create_transform(TransformSpec("PadIfNeeded", {"min_height": 1024, "min_width": 1024, "fill": 0})) is None
    assert _create_transform(TransformSpec("ShiftScaleRotate", {"shift_limit": 0.0625})) is None
    assert _create_transform(TransformSpec("LongestMaxSize", {"max_size": 512})) is None
    assert _create_transform(TransformSpec("SmallestMaxSize", {"max_size": 512})) is None
    assert _create_transform(TransformSpec("CropAndPad", {"px": (-10, 20, -10, 20)})) is None
    assert _create_transform(TransformSpec("Defocus", {"radius": (3, 7), "alias_blur": (0.1, 0.5)})) is None
    assert (
        _create_transform(TransformSpec("SaltAndPepper", {"amount": (0.01, 0.06), "salt_vs_pepper": (0.4, 0.6)}))
        is None
    )
