r"""Unified benchmark result comparison tool.

Handles both image and video result files. Detects media type automatically
based on filename suffix (_video_results.json vs _results.json).

Usage:
    # Compare all results in a directory (quick table)
    python -m tools.compare --results-dir ./results

    # Compare only specific libraries or transforms
    python -m tools.compare --results-dir ./results --libraries albumentationsx kornia

    # Regression check between two runs (exits 1 on regression)
    python -m tools.compare \
        --baseline ./results/baseline \
        --current ./results/current \
        --threshold 0.05 --fail-on-regression
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_HAVE_PANDAS = importlib.util.find_spec("pandas") is not None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _is_video(path: Path) -> bool:
    return "_video" in path.stem


def load_result_file(path: Path) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """Return (library, media_type, metadata, results_dict)."""
    with path.open() as f:
        data = json.load(f)

    stem = path.stem
    if "_video_results" in stem:
        library = stem.replace("_video_results", "")
        media = "video"
    else:
        library = stem.replace("_results", "")
        media = "image"

    return library, media, data.get("metadata", {}), data.get("results", {})


def load_results_dir(directory: Path) -> dict[str, dict[str, Any]]:
    """Load all *.json result files from a directory.

    Returns:
        {library: {"media": "image"|"video", "metadata": {...}, "results": {...}}}
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            library, media, metadata, results = load_result_file(path)
            key = f"{library}_video" if media == "video" else library
            out[key] = {"library": library, "media": media, "metadata": metadata, "results": results}
        except Exception as e:
            print(f"Warning: could not load {path}: {e}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Table generation (single directory)
# ---------------------------------------------------------------------------


def _extract_version(metadata: dict[str, Any], library: str) -> str:
    return metadata.get("library_versions", {}).get(library, "?")


def print_comparison_table(
    loaded: dict[str, dict[str, Any]],
    libraries_filter: list[str] | None = None,
    transforms_filter: list[str] | None = None,
) -> None:
    """Print a side-by-side comparison table for all results in *loaded*."""
    if not loaded:
        print("No results found.")
        return

    # Filter libraries
    if libraries_filter:
        allowed = set(libraries_filter)
        loaded = {k: v for k, v in loaded.items() if v["library"] in allowed}

    # Gather all transform names
    all_transforms: set[str] = set()
    for entry in loaded.values():
        all_transforms.update(entry["results"].keys())

    if transforms_filter:
        all_transforms &= set(transforms_filter)

    sorted_transforms = sorted(all_transforms)
    lib_keys = sorted(loaded.keys())

    # Build header — embed unit in each column so mixed image/video tables are unambiguous
    def col_header(key: str) -> str:
        entry = loaded[key]
        version = _extract_version(entry["metadata"], entry["library"])
        unit = "vid/s" if entry["media"] == "video" else "img/s"
        suffix = " (video)" if entry["media"] == "video" else ""
        return f"{entry['library']}{suffix} {version} [{unit}]"

    headers = ["Transform", *[col_header(k) for k in lib_keys], "Speedup (albx/fastest other)"]

    # Build rows
    rows: list[list[str]] = []

    for transform in sorted_transforms:
        row_vals: dict[str, float] = {}
        row_stds: dict[str, float] = {}
        for key in lib_keys:
            r = loaded[key]["results"].get(transform, {})
            if r.get("supported") and not r.get("early_stopped"):
                row_vals[key] = r.get("median_throughput", 0.0)
                row_stds[key] = r.get("std_throughput", 0.0)

        # Need at least 2 libraries to compare
        if len(row_vals) < 2:
            continue

        max_val = max(row_vals.values())

        row: list[str] = [transform]
        for key in lib_keys:
            if key in row_vals:
                v = row_vals[key]
                s = row_stds[key]
                cell = f"{v:.0f} ± {s:.0f}"
                if v == max_val:
                    cell = f"**{cell}**"
            else:
                cell = "-"
            row.append(cell)

        # Speedup: albumentationsx vs fastest other
        alb_key = next((k for k in lib_keys if loaded[k]["library"] == "albumentationsx" and k in row_vals), None)
        if alb_key:
            alb_val = row_vals[alb_key]
            others = [v for k, v in row_vals.items() if k != alb_key]
            if others:
                speedup = alb_val / max(others)
                row.append(f"{speedup:.2f}x")
            else:
                row.append("N/A")
        else:
            row.append("N/A")

        rows.append(row)

    if not rows:
        print("No transforms with multi-library support found.")
        return

    # Print as markdown table (or plain if pandas unavailable)
    if _HAVE_PANDAS:
        import pandas as pd

        df = pd.DataFrame(rows, columns=headers)
        print(df.to_markdown(index=False))
    else:
        col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
        fmt_row = lambda cells: " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))  # noqa: E731
        print(fmt_row(headers))
        print("-+-".join("-" * w for w in col_widths))
        for row in rows:
            print(fmt_row(row))

    print("\n(single CPU thread, median ± std; units shown per column)")


# ---------------------------------------------------------------------------
# Regression comparison (two directories)
# ---------------------------------------------------------------------------


def compare_regression(
    baseline_dir: Path,
    current_dir: Path,
    libraries_filter: list[str] | None,
    transforms_filter: list[str] | None,
    threshold: float,
    fail_on_regression: bool,
) -> None:
    baseline = load_results_dir(baseline_dir)
    current = load_results_dir(current_dir)

    common_keys = set(baseline) & set(current)
    if not common_keys:
        print("No common libraries between baseline and current.")
        sys.exit(0)

    rows: list[dict[str, Any]] = []
    regression_found = False

    for key in sorted(common_keys):
        lib = baseline[key]["library"]
        if libraries_filter and lib not in libraries_filter:
            continue

        b_results = baseline[key]["results"]
        c_results = current[key]["results"]
        common_transforms = sorted(set(b_results) & set(c_results))

        for transform in common_transforms:
            if transforms_filter and transform not in transforms_filter:
                continue

            b = b_results[transform]
            c = c_results[transform]

            if not b.get("supported") or not c.get("supported"):
                continue
            if b.get("early_stopped") or c.get("early_stopped"):
                continue

            b_tps = b.get("median_throughput", 0.0)
            c_tps = c.get("median_throughput", 0.0)
            if b_tps == 0:
                continue

            delta = (c_tps - b_tps) / b_tps
            is_regression = delta < -threshold

            if is_regression:
                regression_found = True

            status = "REGRESSION" if is_regression else ("faster" if delta > threshold else "same")
            rows.append(
                {
                    "key": key,
                    "library": lib,
                    "transform": transform,
                    "baseline": b_tps,
                    "current": c_tps,
                    "delta_pct": delta * 100,
                    "status": status,
                },
            )

    if not rows:
        print("No comparable transforms found.")
        sys.exit(0)

    # Sort by delta (regressions first)
    rows.sort(key=lambda r: r["delta_pct"])

    tw = max(len(r["transform"]) for r in rows)
    lw = max(len(r["library"]) for r in rows)

    header = (
        f"{'library':<{lw}}  {'transform':<{tw}}  {'baseline':>12}  {'current':>12}  {'delta %':>10}  {'status':<12}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        print(
            f"{r['library']:<{lw}}  "
            f"{r['transform']:<{tw}}  "
            f"{r['baseline']:>12.1f}  "
            f"{r['current']:>12.1f}  "
            f"{r['delta_pct']:>+10.1f}  "
            f"{r['status']:<12}",
        )

    print(sep)
    regressions = [r for r in rows if r["status"] == "REGRESSION"]
    faster = [r for r in rows if r["status"] == "faster"]
    print(
        f"\nSummary: {len(faster)} faster, {len(regressions)} regressions, "
        f"{len(rows) - len(faster) - len(regressions)} same  (threshold ±{threshold:.0%})",
    )

    if regression_found and fail_on_regression:
        print(f"\n{len(regressions)} regression(s) exceed threshold — exiting with code 1")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.compare",
        description="Compare benchmark results (images and/or videos)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--results-dir", "-r", metavar="DIR", help="Show comparison table for all results in a directory")
    mode.add_argument("--baseline", metavar="DIR", help="Baseline results directory (regression mode)")

    parser.add_argument("--current", metavar="DIR", help="Current results directory (required with --baseline)")
    parser.add_argument("--libraries", nargs="+", metavar="LIB", help="Filter to specific libraries")
    parser.add_argument("--transforms", nargs="+", metavar="TRANSFORM", help="Filter to specific transforms")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum delta to flag as regression/improvement (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if any regression exceeds --threshold",
    )

    args = parser.parse_args()

    if args.results_dir:
        loaded = load_results_dir(Path(args.results_dir))
        print_comparison_table(loaded, libraries_filter=args.libraries, transforms_filter=args.transforms)
    else:
        if not args.current:
            parser.error("--current is required when using --baseline")
        compare_regression(
            baseline_dir=Path(args.baseline),
            current_dir=Path(args.current),
            libraries_filter=args.libraries,
            transforms_filter=args.transforms,
            threshold=args.threshold,
            fail_on_regression=args.fail_on_regression,
        )


if __name__ == "__main__":
    main()
