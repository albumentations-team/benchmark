from __future__ import annotations

import pytest

from benchmark.devices import ensure_supported_device


def test_image_gpu_device_is_limited_to_tensor_libraries() -> None:
    ensure_supported_device("torchvision", "image", "cuda")
    ensure_supported_device("kornia", "image", "auto")

    with pytest.raises(ValueError, match="albumentationsx image benchmarks do not support --device cuda"):
        ensure_supported_device("albumentationsx", "image", "cuda")


def test_video_device_support_remains_backwards_compatible() -> None:
    ensure_supported_device("albumentationsx", "video", "cuda")
