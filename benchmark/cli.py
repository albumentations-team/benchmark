r"""Benchmark CLI — single entry point for running and comparing benchmarks.

Usage examples:

    # Run all built-in libraries, image mode, CPU
    python -m benchmark.cli run --data-dir /images --output ./results

    # Run one library, video mode
    python -m benchmark.cli run --data-dir /videos --output ./results \
        --media video --libraries albumentationsx

    # Run a single transform in one library (fast feedback loop)
    python -m benchmark.cli run --data-dir /images --output ./results/current \
        --libraries albumentationsx --transforms GaussianBlur

    # Compare two result directories for regressions
    python -m benchmark.cli compare \
        --baseline ./results/baseline \
        --current ./results/current \
        --threshold 0.05 --fail-on-regression
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Built-in library → spec file mapping (image / video)
_IMAGE_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_impl.py",
    "torchvision": "benchmark/transforms/torchvision_impl.py",
    "kornia": "benchmark/transforms/kornia_impl.py",
    "imgaug": "benchmark/transforms/imgaug_impl.py",
    "augly": "benchmark/transforms/augly_impl.py",
}

_VIDEO_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_video_impl.py",
    "torchvision": "benchmark/transforms/torchvision_video_impl.py",
    "kornia": "benchmark/transforms/kornia_video_impl.py",
}

_REQUIREMENTS: dict[str, str] = {
    "albumentationsx": "requirements/albumentationsx.txt",
    "torchvision": "requirements/torchvision.txt",
    "kornia": "requirements/kornia.txt",
    "imgaug": "requirements/imgaug.txt",
    "augly": "requirements/augly.txt",
}

_VIDEO_REQUIREMENTS: dict[str, str] = {
    "albumentationsx": "requirements/albumentationsx.txt",
    "torchvision": "requirements/torchvision-video.txt",
    "kornia": "requirements/kornia-video.txt",
}


# ---------------------------------------------------------------------------
# venv / runner helpers
# ---------------------------------------------------------------------------


def _extract_library(spec_file: Path) -> str:
    """Extract LIBRARY string from a spec file without importing it."""
    for line in spec_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("LIBRARY"):
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"').strip("'")
    raise ValueError(f"Could not find LIBRARY assignment in {spec_file}")


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(library: str, media: str, repo_root: Path) -> Path:
    suffix = "_video" if media == "video" else ""
    venv_dir = repo_root / f".venv_{library}{suffix}"

    if not venv_dir.exists():
        logger.info("Creating venv for %s (%s)...", library, media)
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    python = _venv_python(venv_dir)

    # Install / upgrade uv
    subprocess.run([str(python), "-m", "pip", "install", "-q", "-U", "uv"], check=True)

    # Base requirements
    base_req = repo_root / "requirements" / "requirements.txt"
    subprocess.run(
        [str(python), "-m", "uv", "pip", "install", "-q", "-U", "-r", str(base_req)],
        check=True,
    )

    # Library-specific requirements
    req_map = _VIDEO_REQUIREMENTS if media == "video" else _REQUIREMENTS
    lib_req = repo_root / req_map[library]
    subprocess.run(
        [str(python), "-m", "uv", "pip", "install", "-q", "-U", "--force-reinstall", "-r", str(lib_req)],
        check=True,
    )

    return python


def _run_single(
    library: str,
    spec_file: Path,
    data_dir: Path,
    output_file: Path,
    media: str,
    num_items: int | None,
    num_runs: int,
    max_warmup: int | None,
    warmup_window: int,
    warmup_threshold: float,
    min_warmup_windows: int,
    repo_root: Path,
    transforms_filter: list[str] | None = None,
) -> None:
    python = _ensure_venv(library, media, repo_root)

    cmd = [
        str(python),
        "-m",
        "benchmark.runner",
        "--specs-file",
        str(spec_file),
        "--data-dir",
        str(data_dir),
        "--output",
        str(output_file),
        "--media",
        media,
        "--num-runs",
        str(num_runs),
        "--warmup-window",
        str(warmup_window),
        "--warmup-threshold",
        str(warmup_threshold),
        "--min-warmup-windows",
        str(min_warmup_windows),
    ]
    if num_items is not None:
        cmd += ["--num-items", str(num_items)]
    if max_warmup is not None:
        cmd += ["--max-warmup", str(max_warmup)]

    env_extra: dict[str, str] = {}
    if transforms_filter:
        env_extra["BENCHMARK_TRANSFORMS_FILTER"] = ",".join(transforms_filter)

    import os

    env = {**os.environ, **env_extra}

    logger.info("Running %s %s benchmark → %s", library, media, output_file)
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# GCP cloud helper
# ---------------------------------------------------------------------------


def _cmd_run_gcp(args: argparse.Namespace, repo_root: Path, local_output_dir: Path, media: str) -> None:
    """Run benchmarks on a GCP instance."""
    from benchmark.cloud.gcp import GCPRunner
    from benchmark.cloud.instance import GCPInstanceConfig

    if not args.gcp_project:
        logger.error("--gcp-project is required when using --cloud gcp")
        sys.exit(1)

    remote_data_dir = args.gcp_remote_data_dir or args.data_dir

    config = GCPInstanceConfig(
        project=args.gcp_project,
        zone=args.gcp_zone,
        machine_type=args.gcp_machine_type,
        accelerator_type=args.gcp_gpu_type,
        accelerator_count=1 if args.gcp_gpu_type else 0,
        image_family="pytorch-latest-gpu" if args.gcp_gpu_type else "pytorch-latest-cpu",
    )

    # Build CLI args that will be forwarded to the remote run
    remote_args: list[str] = ["--media", media]
    if args.libraries:
        remote_args += ["--libraries", *args.libraries]
    if args.transforms:
        remote_args += ["--transforms", *args.transforms]
    if args.num_items:
        remote_args += ["--num-items", str(args.num_items)]
    if args.num_runs != 5:
        remote_args += ["--num-runs", str(args.num_runs)]

    runner = GCPRunner(config)
    runner.run(
        repo_root=repo_root,
        remote_cli_args=remote_args,
        local_output_dir=local_output_dir,
        remote_data_dir=remote_data_dir,
        keep_instance=args.gcp_keep_instance,
    )


# ---------------------------------------------------------------------------
# `run` subcommand
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).parent.parent.resolve()
    media: str = args.media
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cloud path: delegate the whole run to a GCP instance
    # ------------------------------------------------------------------
    if args.cloud == "gcp":
        _cmd_run_gcp(args, repo_root, output_dir, media)
        return

    # Custom spec file path takes priority
    if args.spec:
        spec_file = Path(args.spec)
        library = _extract_library(spec_file)
        output_file = output_dir / f"{spec_file.stem}.json"
        _run_single(
            library=library,
            spec_file=spec_file,
            data_dir=Path(args.data_dir),
            output_file=output_file,
            media=media,
            num_items=args.num_items,
            num_runs=args.num_runs,
            max_warmup=args.max_warmup,
            warmup_window=args.warmup_window,
            warmup_threshold=args.warmup_threshold,
            min_warmup_windows=args.min_warmup_windows,
            repo_root=repo_root,
            transforms_filter=args.transforms,
        )
        return

    # Built-in libraries
    spec_map = _VIDEO_SPECS if media == "video" else _IMAGE_SPECS
    available = list(spec_map.keys())

    requested: list[str] = args.libraries or available
    unknown = set(requested) - set(available)
    if unknown:
        logger.error("Unknown libraries for %s mode: %s. Available: %s", media, sorted(unknown), available)
        sys.exit(1)

    suffix = "_video" if media == "video" else ""
    for library in requested:
        spec_file = repo_root / spec_map[library]
        output_file = output_dir / f"{library}{suffix}_results.json"
        _run_single(
            library=library,
            spec_file=spec_file,
            data_dir=Path(args.data_dir),
            output_file=output_file,
            media=media,
            num_items=args.num_items,
            num_runs=args.num_runs,
            max_warmup=args.max_warmup,
            warmup_window=args.warmup_window,
            warmup_threshold=args.warmup_threshold,
            min_warmup_windows=args.min_warmup_windows,
            repo_root=repo_root,
            transforms_filter=args.transforms,
        )

    logger.info("All benchmarks complete. Results in: %s", output_dir)


# ---------------------------------------------------------------------------
# `compare` subcommand  (Todo 4: regression detection)
# ---------------------------------------------------------------------------


def _load_result_dir(directory: Path) -> dict[str, dict[str, Any]]:
    """Load all *_results.json files from a directory.

    Returns: {library: {transform_name: result_dict}}
    """
    data: dict[str, dict[str, Any]] = {}
    for f in sorted(directory.glob("*.json")):
        try:
            with f.open() as fp:
                raw = json.load(fp)
            # Strip _results / _video_results suffixes to get a clean library key
            key = f.stem.replace("_video_results", "").replace("_results", "")
            data[key] = raw.get("results", {})
        except Exception as e:
            logger.warning("Could not load %s: %s", f, e)
    return data


def cmd_compare(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline)
    current_dir = Path(args.current)

    if not baseline_dir.exists():
        logger.error("Baseline directory not found: %s", baseline_dir)
        sys.exit(1)
    if not current_dir.exists():
        logger.error("Current directory not found: %s", current_dir)
        sys.exit(1)

    baseline = _load_result_dir(baseline_dir)
    current = _load_result_dir(current_dir)

    libraries_filter: set[str] | None = set(args.libraries) if args.libraries else None
    transforms_filter: set[str] | None = set(args.transforms) if args.transforms else None
    threshold: float = args.threshold

    # Collect rows
    rows: list[dict[str, Any]] = []
    regression_found = False

    all_libs = sorted(set(baseline) & set(current))
    for lib in all_libs:
        if libraries_filter and lib not in libraries_filter:
            continue

        b_results = baseline[lib]
        c_results = current[lib]

        all_transforms = sorted(set(b_results) & set(c_results))
        for transform in all_transforms:
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
                    "library": lib,
                    "transform": transform,
                    "baseline": b_tps,
                    "current": c_tps,
                    "delta_pct": delta * 100,
                    "status": status,
                },
            )

    if not rows:
        logger.info("No common transforms found between baseline and current.")
        sys.exit(0)

    # Print table
    lw = max(len(r["library"]) for r in rows)
    tw = max(len(r["transform"]) for r in rows)
    col_widths = {
        "library": max(len("library"), lw),
        "transform": max(len("transform"), tw),
        "baseline": 12,
        "current": 12,
        "delta": 10,
        "status": 12,
    }

    header = (
        f"{'library':<{col_widths['library']}}  "
        f"{'transform':<{col_widths['transform']}}  "
        f"{'baseline':>{col_widths['baseline']}}  "
        f"{'current':>{col_widths['current']}}  "
        f"{'delta %':>{col_widths['delta']}}  "
        f"{'status':<{col_widths['status']}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in sorted(rows, key=lambda x: x["delta_pct"]):
        unit = "img/s" if "video" not in r["library"] else "vid/s"
        print(
            f"{r['library']:<{col_widths['library']}}  "
            f"{r['transform']:<{col_widths['transform']}}  "
            f"{r['baseline']:>{col_widths['baseline']}.1f}  "
            f"{r['current']:>{col_widths['current']}.1f}  "
            f"{r['delta_pct']:>+{col_widths['delta']}.1f}  "
            f"{r['status']:<{col_widths['status']}}"
            f"  ({unit})",
        )

    print(sep)

    # Summary
    regressions = [r for r in rows if r["status"] == "REGRESSION"]
    faster = [r for r in rows if r["status"] == "faster"]
    print(
        f"\nSummary: {len(faster)} faster, {len(regressions)} regressions, "
        f"{len(rows) - len(faster) - len(regressions)} same",
    )

    if regression_found and args.fail_on_regression:
        print(f"\n{len(regressions)} regression(s) exceed threshold {threshold:.0%} — exiting with code 1")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.cli",
        description="Image/video augmentation benchmark suite",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    run_p = subparsers.add_parser("run", help="Run benchmarks")
    run_p.add_argument("--data-dir", "-d", required=True, help="Directory with images or videos")
    run_p.add_argument("--output", "-o", required=True, help="Directory to write result JSON files")
    run_p.add_argument(
        "--media",
        choices=["image", "video"],
        default="image",
        help="Media type (default: image)",
    )
    run_p.add_argument(
        "--libraries",
        nargs="+",
        metavar="LIB",
        help="Which built-in libraries to run (default: all for the selected media type)",
    )
    run_p.add_argument(
        "--transforms",
        nargs="+",
        metavar="TRANSFORM",
        help="Run only these transforms (by name). Default: all.",
    )
    run_p.add_argument(
        "--spec",
        "-s",
        metavar="FILE",
        help="Custom spec file (overrides --libraries; library inferred from LIBRARY variable)",
    )
    # Cloud options
    run_p.add_argument("--cloud", choices=["gcp"], default=None, help="Run on cloud (currently: gcp)")
    run_p.add_argument("--gcp-project", metavar="PROJECT", help="GCP project ID (required with --cloud gcp)")
    run_p.add_argument("--gcp-zone", metavar="ZONE", default="us-central1-a", help="GCP zone (default: us-central1-a)")
    run_p.add_argument("--gcp-machine-type", metavar="TYPE", default="n1-standard-8")
    run_p.add_argument(
        "--gcp-gpu-type",
        metavar="TYPE",
        default=None,
        help="GPU accelerator type (e.g. nvidia-tesla-t4)",
    )
    run_p.add_argument("--gcp-remote-data-dir", metavar="PATH", help="Path to data on the remote instance")
    run_p.add_argument("--gcp-keep-instance", action="store_true", help="Do not delete instance after run (debug)")

    run_p.add_argument("--num-items", "-n", type=int, help="Number of images/videos (default depends on media type)")
    run_p.add_argument("--num-runs", "-r", type=int, default=5, help="Benchmark runs per transform (default: 5)")
    run_p.add_argument("--max-warmup", type=int, help="Max warmup iterations (default depends on media type)")
    run_p.add_argument("--warmup-window", type=int, default=5)
    run_p.add_argument("--warmup-threshold", type=float, default=0.05)
    run_p.add_argument("--min-warmup-windows", type=int, default=3)

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    cmp_p = subparsers.add_parser("compare", help="Compare two result directories")
    cmp_p.add_argument("--baseline", required=True, help="Baseline results directory")
    cmp_p.add_argument("--current", required=True, help="Current results directory")
    cmp_p.add_argument(
        "--libraries",
        nargs="+",
        metavar="LIB",
        help="Filter to specific libraries",
    )
    cmp_p.add_argument(
        "--transforms",
        nargs="+",
        metavar="TRANSFORM",
        help="Filter to specific transforms",
    )
    cmp_p.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum delta fraction to consider a change significant (default: 0.05 = 5%%)",
    )
    cmp_p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if any regression exceeds --threshold",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
