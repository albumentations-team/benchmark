from __future__ import annotations

from pathlib import Path

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
