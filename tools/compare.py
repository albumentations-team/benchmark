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
import math
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
    elif "_micro_results" in stem:
        library = stem.replace("_micro_results", "")
        media = "image"
    elif "_pipeline_results" in stem:
        library = stem.replace("_pipeline_results", "")
        media = "image"
    else:
        library = stem.replace("_results", "")
        media = "image"

    return library, media, data.get("metadata", {}), data.get("results", {})


def load_results_dir(directory: Path) -> dict[str, dict[str, Any]]:
    """Load benchmark summary result files from a directory.

    Returns:
        {library: {"media": "image"|"video", "metadata": {...}, "results": {...}}}
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*_results.json")):
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


# Human-readable column titles (also avoids `_` in headers tripping strict MD emphasis parsers).
_MARKDOWN_TABLE_LIBRARY_LABELS: dict[str, str] = {
    "albumentations_mit": "Albumentations (MIT)",
    "albumentationsx": "AlbumentationsX",
}


def _markdown_table_library_label(library: str) -> str:
    return _MARKDOWN_TABLE_LIBRARY_LABELS.get(library, library)


def _speedup_ratio_sigma_bounds(
    ref_mu: float,
    ref_std: float,
    comp_mu: float,
    comp_std: float,
) -> tuple[float, float, float]:
    """Lower bound, point estimate, upper bound for ref/competitor throughput ratio.

    Uses independent +/-1 standard deviation corners: low = max(ref - ref_std, 0) / (comp + comp_std),
    high = (ref + ref_std) / max(comp - comp_std, epsilon). Assumes non-negative throughputs.
    """
    if comp_mu <= 0 or ref_mu < 0:
        return (float("nan"), float("nan"), float("nan"))
    mid = ref_mu / comp_mu
    comp_hi = comp_mu + max(comp_std, 0.0)
    comp_lo_raw = comp_mu - max(comp_std, 0.0)
    comp_lo = comp_lo_raw if comp_lo_raw > 0 else max(comp_mu * 1e-9, 1e-18)
    ref_lo = max(ref_mu - max(ref_std, 0.0), 0.0)
    ref_hi = ref_mu + max(ref_std, 0.0)
    low = ref_lo / comp_hi if comp_hi > 0 else mid
    high = ref_hi / comp_lo if comp_lo > 0 else mid
    return (low, mid, high)


def _format_speedup_cell(low: float, mid: float, high: float) -> str:
    if math.isnan(mid):
        return "N/A"
    span = high - low
    rel = max(abs(mid), 1e-9)
    if span <= 0 or span < 0.005 * rel:
        return f"{mid:.2f}x"
    return f"{mid:.2f}x ({low:.2f}-{high:.2f}x)"


def _format_result_cell(result: dict[str, Any], *, is_fastest: bool = False) -> str:
    if not result.get("supported"):
        return "-"
    if result.get("early_stopped"):
        marker = result.get("slow_marker")
        if isinstance(marker, str) and marker:
            return marker
        threshold = result.get("slow_threshold_throughput")
        unit = result.get("slow_threshold_unit", "img/s")
        if isinstance(threshold, (int, float)) and threshold > 0:
            return f"≤{threshold:.0f} {unit}"
        return "slow-skipped"
    v = float(result.get("median_throughput", 0.0))
    s = float(result.get("std_throughput", 0.0))
    cell = f"{v:.0f} ± {s:.0f}"
    return f"**{cell}**" if is_fastest else cell


def format_comparison_table(
    loaded: dict[str, dict[str, Any]],
    libraries_filter: list[str] | None = None,
    transforms_filter: list[str] | None = None,
    *,
    speedup_header: str = "Speedup (albx / fastest, +/-1sd)",
    speedup_ref_library: str = "albumentationsx",
) -> str:
    """Return markdown table string for all results in *loaded*."""
    if not loaded:
        return ""

    if libraries_filter:
        allowed = set(libraries_filter)
        loaded = {k: v for k, v in loaded.items() if v["library"] in allowed}

    all_transforms: set[str] = set()
    for entry in loaded.values():
        all_transforms.update(entry["results"].keys())

    if transforms_filter:
        all_transforms &= set(transforms_filter)

    sorted_transforms = sorted(all_transforms)
    lib_keys = sorted(loaded.keys())

    def col_header(key: str) -> str:
        entry = loaded[key]
        version = _extract_version(entry["metadata"], entry["library"])
        unit = "vid/s" if entry["media"] == "video" else "img/s"
        suffix = " (video)" if entry["media"] == "video" else ""
        lib_label = _markdown_table_library_label(entry["library"])
        return f"{lib_label}{suffix} {version} [{unit}]"

    headers = ["Transform", *[col_header(k) for k in lib_keys], speedup_header]

    rows: list[list[str]] = []
    for transform in sorted_transforms:
        row_vals: dict[str, float] = {}
        row_stds: dict[str, float] = {}
        row_results: dict[str, dict[str, Any]] = {}
        for key in lib_keys:
            r = loaded[key]["results"].get(transform, {})
            if r.get("supported"):
                row_results[key] = r
            if r.get("supported") and not r.get("early_stopped"):
                row_vals[key] = r.get("median_throughput", 0.0)
                row_stds[key] = r.get("std_throughput", 0.0)

        if not row_results:
            continue

        max_val = max(row_vals.values()) if row_vals else None

        row: list[str] = [transform]
        for key in lib_keys:
            if key in row_results:
                is_fastest = max_val is not None and row_vals.get(key) == max_val
                cell = _format_result_cell(
                    row_results[key],
                    is_fastest=is_fastest,
                )
            else:
                cell = "-"
            row.append(cell)

        ref_key = next(
            (k for k in lib_keys if loaded[k]["library"] == speedup_ref_library and k in row_vals),
            None,
        )
        if ref_key:
            ref_val = row_vals[ref_key]
            ref_s = row_stds[ref_key]
            other_keys = [k for k in lib_keys if k != ref_key and k in row_vals]
            if other_keys:
                comp_key = max(other_keys, key=lambda k: row_vals[k])
                comp_val = row_vals[comp_key]
                comp_s = row_stds[comp_key]
                low, mid, high = _speedup_ratio_sigma_bounds(ref_val, ref_s, comp_val, comp_s)
                row.append(_format_speedup_cell(low, mid, high))
            else:
                row.append("N/A")
        else:
            row.append("N/A")

        rows.append(row)

    if not rows:
        return ""

    if _HAVE_PANDAS:
        import pandas as pd

        df = pd.DataFrame(rows, columns=headers)
        return df.to_markdown(index=False)
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt_row = lambda cells: " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))  # noqa: E731
    lines = [fmt_row(headers), "-+-".join("-" * w for w in col_widths)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def format_head_to_head_table(
    loaded: dict[str, dict[str, Any]],
    transforms_filter: list[str] | None = None,
) -> str:
    """Head-to-head AlbumentationsX vs Albumentations (MIT) comparison table."""
    return format_comparison_table(
        loaded,
        libraries_filter=["albumentationsx", "albumentations_mit"],
        transforms_filter=transforms_filter,
        speedup_header="Speedup (albx / MIT, +/-1sd)",
        speedup_ref_library="albumentationsx",
    )


def print_comparison_table(
    loaded: dict[str, dict[str, Any]],
    libraries_filter: list[str] | None = None,
    transforms_filter: list[str] | None = None,
) -> None:
    """Print a side-by-side comparison table for all results in *loaded*."""
    table = format_comparison_table(loaded, libraries_filter, transforms_filter)
    if table:
        print(table)
        print("\n(single CPU thread, median +/- std; units per column; speedup +/-1sd range vs fastest other)")
    else:
        print("No results found.")


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
