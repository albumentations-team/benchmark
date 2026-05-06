r"""Update README.md with full RGB benchmark tables from result JSONs.

Usage:
    python -m tools.update_readme
    python -m tools.update_readme --image-results output/rgb_micro --dataloader-results output/rgb_dataloader
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from tools.compare import format_comparison_table, load_results_dir

# Libraries to keep out of public docs (internal / historical reference only)
_DOCS_EXCLUDED: frozenset[str] = frozenset({"albumentations_mit"})
_DEFAULT_PUBLISHED_ROOT = Path("results/published")


def latest_results_dir(root: Path, pattern: str) -> Path | None:
    """Return the latest matching published results directory by lexicographic snapshot name."""
    if not root.exists():
        return None
    matches = sorted(path for path in root.glob(pattern) if path.is_dir())
    return matches[-1] if matches else None


def patch_readme(
    readme_path: Path,
    image_table: str | None,
    video_table: str | None,
    image_summary: str | None,
    video_summary: str | None,
    multichannel_table: str | None = None,
    dataloader_table: str | None = None,
    dataloader_summary: str | None = None,
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
        # Blank lines after markers help some Markdown engines start a new block (tables after HTML comments).
        replacement = f"{marker_start}\n\n{new_content.strip()}\n\n{marker_end}"
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
    if dataloader_table is not None:
        content = replace_section(
            "<!-- DATALOADER_BENCHMARK_TABLE_START -->",
            "<!-- DATALOADER_BENCHMARK_TABLE_END -->",
            dataloader_table,
        )
    if image_summary is not None:
        content = replace_section(
            "<!-- IMAGE_SPEEDUP_SUMMARY_START -->",
            "<!-- IMAGE_SPEEDUP_SUMMARY_END -->",
            image_summary,
        )
    if dataloader_summary is not None:
        content = replace_section(
            "<!-- DATALOADER_SPEEDUP_SUMMARY_START -->",
            "<!-- DATALOADER_SPEEDUP_SUMMARY_END -->",
            dataloader_summary,
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
        description="Update README with full RGB benchmark tables from result JSONs",
    )
    parser.add_argument("--readme", default="README.md", type=Path, help="README path")
    parser.add_argument(
        "--image-results",
        default=None,
        type=Path,
        help="Directory with RGB micro *_results.json files. Defaults to the latest paper-rgb-micro-* snapshot.",
    )
    parser.add_argument(
        "--dataloader-results",
        default=None,
        type=Path,
        help=(
            "Directory with RGB DataLoader *_results.json files. "
            "Defaults to the latest paper-rgb-dataloader-* snapshot."
        ),
    )
    parser.add_argument(
        "--published-results-root",
        default=os.environ.get("PAPER_RGB_RESULTS_ROOT", str(_DEFAULT_PUBLISHED_ROOT)),
        type=Path,
        help="Root used to discover latest paper-rgb-* snapshots when result directories are omitted.",
    )
    parser.add_argument(
        "--video-results",
        default="output_videos",
        type=Path,
        help="Accepted for backwards compatibility; public README output is RGB-only.",
    )
    parser.add_argument(
        "--multichannel-results",
        type=Path,
        default=None,
        help="Accepted for backwards compatibility; public README output is RGB-only.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    readme = repo_root / args.readme
    published_results_root = repo_root / args.published_results_root
    image_results = (
        repo_root / args.image_results
        if args.image_results is not None
        else latest_results_dir(published_results_root, "paper-rgb-micro-*")
    )
    dataloader_results = (
        repo_root / args.dataloader_results
        if args.dataloader_results is not None
        else latest_results_dir(published_results_root, "paper-rgb-dataloader-*")
    )

    def _load(directory: Path, media: str, *, exclude_docs: bool) -> dict[str, dict[str, object]]:
        loaded = load_results_dir(directory)
        return {
            k: v
            for k, v in loaded.items()
            if v["media"] == media and (not exclude_docs or v["library"] not in _DOCS_EXCLUDED)
        }

    # Public tables exclude internal/historical libraries.
    image_loaded = _load(image_results, "image", exclude_docs=True) if image_results is not None else {}
    dataloader_loaded: dict[str, dict[str, object]] = {}
    if dataloader_results is not None and dataloader_results.exists():
        dataloader_loaded = _load(dataloader_results, "image", exclude_docs=True)

    image_table = format_comparison_table(image_loaded) if image_loaded else None
    dataloader_table = format_comparison_table(dataloader_loaded, name_header="Recipe") if dataloader_loaded else None

    # Summary text for Performance Highlights
    image_summary = compute_summary_text(image_table, "RGB micro") if image_table else None
    dataloader_summary = compute_summary_text(dataloader_table, "RGB DataLoader") if dataloader_table else None

    changed = patch_readme(
        readme,
        image_table=image_table,
        video_table=None,
        image_summary=image_summary,
        video_summary=None,
        multichannel_table=None,
        dataloader_table=dataloader_table,
        dataloader_summary=dataloader_summary,
    )
    if changed:
        print(f"Updated {readme}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
