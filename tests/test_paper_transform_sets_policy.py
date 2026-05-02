from __future__ import annotations

from pathlib import Path

from benchmark.transforms.kornia_unstable import KORNIA_BENCHMARK_EXCLUDED_NAMES


def _paper_names(path: str) -> set[str]:
    text = Path(path).read_text(encoding="utf-8")
    block = text.split("```text", maxsplit=1)[1].split("```", maxsplit=1)[0]
    return {line.strip() for line in block.splitlines() if line.strip()}


def test_center_crop_is_not_in_paper_transform_sets() -> None:
    for path in (
        "docs/paper_transform_sets/rgb.md",
        "docs/paper_transform_sets/9ch.md",
        "docs/paper_transform_sets/video.md",
    ):
        assert "CenterCrop224" not in _paper_names(path)


def test_kornia_video_pipeline_unstable_rows_stay_in_micro_paper_sets() -> None:
    expected = {
        "docs/paper_transform_sets/rgb.md": KORNIA_BENCHMARK_EXCLUDED_NAMES,
        "docs/paper_transform_sets/9ch.md": {
            "CornerIllumination",
            "Erasing",
            "LinearIllumination",
            "MotionBlur",
            "Perspective",
            "Posterize",
            "RandomJigsaw",
            "RandomRotate90",
            "Shear",
        },
        "docs/paper_transform_sets/video.md": KORNIA_BENCHMARK_EXCLUDED_NAMES,
    }

    for path, expected_names in expected.items():
        assert _paper_names(path) & KORNIA_BENCHMARK_EXCLUDED_NAMES == expected_names


def test_shear_stays_in_image_paper_sets_for_non_kornia_gpu_paths() -> None:
    assert "Shear" in _paper_names("docs/paper_transform_sets/rgb.md")
    assert "Shear" in _paper_names("docs/paper_transform_sets/9ch.md")
