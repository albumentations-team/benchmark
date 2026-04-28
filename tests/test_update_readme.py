from __future__ import annotations

from typing import TYPE_CHECKING

from tools.update_readme import patch_readme

if TYPE_CHECKING:
    from pathlib import Path


def _readme_with_markers() -> str:
    return """# README

<!-- IMAGE_BENCHMARK_TABLE_START -->

old image

<!-- IMAGE_BENCHMARK_TABLE_END -->

<!-- VIDEO_BENCHMARK_TABLE_START -->

old video

<!-- VIDEO_BENCHMARK_TABLE_END -->

<!-- IMAGE_SPEEDUP_SUMMARY_START -->

old image summary

<!-- IMAGE_SPEEDUP_SUMMARY_END -->
"""


def test_patch_readme_can_update_rgb_without_touching_video(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(_readme_with_markers())

    changed = patch_readme(
        readme,
        image_table="new rgb table",
        video_table=None,
        image_summary="new rgb summary",
        video_summary=None,
    )

    content = readme.read_text()
    assert changed
    assert "new rgb table" in content
    assert "new rgb summary" in content
    assert "old video" in content


def test_patch_readme_reports_no_change_when_no_sections_requested(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    original = _readme_with_markers()
    readme.write_text(original)

    changed = patch_readme(
        readme,
        image_table=None,
        video_table=None,
        image_summary=None,
        video_summary=None,
    )

    assert not changed
    assert readme.read_text() == original
