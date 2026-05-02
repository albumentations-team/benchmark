from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_albumentationsx_specs_do_not_use_removed_range_aliases() -> None:
    root = Path(__file__).parents[1]
    specs = [
        root / "benchmark/transforms/albumentationsx_impl.py",
        root / "benchmark/transforms/albumentationsx_video_impl.py",
        root / "benchmark/transforms/albumentationsx_multichannel_impl.py",
    ]
    removed_constructor_args = [
        "limit=",
        "blur_limit=",
        "sigma_limit=",
        "clip_limit=",
        "gamma_limit=",
        "brightness_limit=",
        "contrast_limit=",
        "hue_shift_limit=",
        "sat_shift_limit=",
        "val_shift_limit=",
        "r_shift_limit=",
        "g_shift_limit=",
        "b_shift_limit=",
        "distort_limit=",
        "scale_limit=",
        "shift_limit=",
        "rotate_limit=",
        "primary_distortion_limit=",
        "secondary_distortion_limit=",
        "num_shadows_limit=",
    ]

    for spec in specs:
        source = spec.read_text(encoding="utf-8")
        for arg in removed_constructor_args:
            assert arg not in source, f"{spec} still uses removed AlbumentationsX 2.2.0 arg {arg}"


def test_albumentationsx_specs_include_motion_blur() -> None:
    pytest.importorskip("albumentations")

    from benchmark.transforms.albumentationsx_impl import TRANSFORMS as IMAGE_TRANSFORMS
    from benchmark.transforms.albumentationsx_video_impl import TRANSFORMS as VIDEO_TRANSFORMS

    assert "MotionBlur" in {transform["name"] for transform in IMAGE_TRANSFORMS}
    assert "MotionBlur" in {transform["name"] for transform in VIDEO_TRANSFORMS}


def test_albumentationsx_micro_spec_does_not_use_to_tensor() -> None:
    root = Path(__file__).parents[1]
    spec_paths = [
        root / "benchmark/transforms/albumentationsx_impl.py",
        root / "benchmark/transforms/albumentationsx_multichannel_impl.py",
    ]

    for spec_path in spec_paths:
        source = spec_path.read_text(encoding="utf-8")
        assert "ToTensor" not in source


def test_albumentationsx_pipeline_recipe_returns_chw_tensor() -> None:
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")

    from benchmark.transforms.albumentationsx_pipeline_impl import TRANSFORMS, __call__

    transform = next(item["transform"] for item in TRANSFORMS if item["name"] == "RandomCrop224+Normalize+ToTensor")
    result = __call__(transform, np.zeros((256, 256, 3), dtype=np.uint8))

    assert isinstance(result, torch.Tensor)
    assert tuple(result.shape) == (3, 224, 224)
