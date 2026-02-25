r"""Update README.md with full benchmark tables from result JSONs.

Usage:
    python -m tools.update_readme
    python -m tools.update_readme --image-results output/ --video-results output_videos/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tools.compare import format_comparison_table, load_results_dir


def patch_readme(
    readme_path: Path,
    image_table: str | None,
    video_table: str | None,
    image_summary: str | None,
    video_summary: str | None,
    multichannel_table: str | None = None,
) -> bool:
    """Patch README between markers. Returns True if changed."""
    content = readme_path.read_text()

    def replace_section(marker_start: str, marker_end: str, new_content: str | None) -> str:
        pattern = re.compile(
            rf"({re.escape(marker_start)}).*?({re.escape(marker_end)})",
            re.DOTALL,
        )
        if new_content is None:
            return content
        replacement = f"{marker_start}\n{new_content.strip()}\n{marker_end}"
        new_content_str, n = pattern.subn(replacement, content, count=1)
        return new_content_str if n else content

    orig = content

    if image_table is not None:
        content = replace_section(
            "<!-- IMAGE_BENCHMARK_TABLE_START -->",
            "<!-- IMAGE_BENCHMARK_TABLE_END -->",
            image_table,
        )
    if video_table is not None:
        content = replace_section(
            "<!-- VIDEO_BENCHMARK_TABLE_START -->",
            "<!-- VIDEO_BENCHMARK_TABLE_END -->",
            video_table,
        )
    if multichannel_table is not None:
        content = replace_section(
            "<!-- MULTICHANNEL_BENCHMARK_TABLE_START -->",
            "<!-- MULTICHANNEL_BENCHMARK_TABLE_END -->",
            multichannel_table,
        )
    if image_summary is not None:
        content = replace_section(
            "<!-- IMAGE_SPEEDUP_SUMMARY_START -->",
            "<!-- IMAGE_SPEEDUP_SUMMARY_END -->",
            image_summary,
        )
    if video_summary is not None:
        content = replace_section(
            "<!-- VIDEO_SPEEDUP_SUMMARY_START -->",
            "<!-- VIDEO_SPEEDUP_SUMMARY_END -->",
            video_summary,
        )

    if content != orig:
        readme_path.write_text(content)
        return True
    return False


def compute_summary_text(_table: str, media: str) -> str:
    """Brief summary for Performance Highlights section."""
    return f"See the full benchmark table above for {media} results."


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.update_readme",
        description="Update README with full benchmark tables from result JSONs",
    )
    parser.add_argument("--readme", default="README.md", type=Path, help="README path")
    parser.add_argument("--image-results", default="output", type=Path, help="Directory with *_results.json (image)")
    parser.add_argument(
        "--video-results",
        default="output_videos",
        type=Path,
        help="Directory with *_video_results.json",
    )
    parser.add_argument(
        "--multichannel-results",
        type=Path,
        default=None,
        help=(
            "Directory with multichannel *_results.json (e.g. output/multichannel). "
            "If unset, derived as image-results/multichannel."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    readme = repo_root / args.readme
    image_results = repo_root / args.image_results
    video_results = repo_root / args.video_results
    multichannel_results = (
        (repo_root / args.multichannel_results) if args.multichannel_results else (image_results / "multichannel")
    )

    # Load and filter by media type
    image_loaded = load_results_dir(image_results)
    image_loaded = {k: v for k, v in image_loaded.items() if v["media"] == "image"}

    video_loaded = load_results_dir(video_results)
    video_loaded = {k: v for k, v in video_loaded.items() if v["media"] == "video"}

    multichannel_loaded: dict[str, dict[str, object]] = {}
    if multichannel_results.exists():
        multichannel_loaded = load_results_dir(multichannel_results)
        multichannel_loaded = {k: v for k, v in multichannel_loaded.items() if v["media"] == "image"}

    image_table = format_comparison_table(image_loaded) if image_loaded else None
    video_table = format_comparison_table(video_loaded) if video_loaded else None
    multichannel_table = format_comparison_table(multichannel_loaded) if multichannel_loaded else None

    # Summary text for Performance Highlights
    image_summary = compute_summary_text(image_table, "image") if image_table else None
    video_summary = compute_summary_text(video_table, "video") if video_table else None

    changed = patch_readme(
        readme,
        image_table=image_table,
        video_table=video_table,
        image_summary=image_summary,
        video_summary=video_summary,
        multichannel_table=multichannel_table,
    )
    if changed:
        print(f"Updated {readme}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
