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
import time
from pathlib import Path

from tqdm import tqdm

from benchmark.term import configure_logging, tqdm_kwargs

logger = logging.getLogger(__name__)

# Built-in library → spec file mapping (image / video)
_IMAGE_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_impl.py",
    "albumentations_mit": "benchmark/transforms/albumentations_mit_impl.py",
    "torchvision": "benchmark/transforms/torchvision_impl.py",
    "kornia": "benchmark/transforms/kornia_impl.py",
}

_MULTICHANNEL_IMAGE_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_multichannel_impl.py",
    "albumentations_mit": "benchmark/transforms/albumentations_mit_multichannel_impl.py",
    "torchvision": "benchmark/transforms/torchvision_multichannel_impl.py",
    "kornia": "benchmark/transforms/kornia_multichannel_impl.py",
}

_VIDEO_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_video_impl.py",
    "albumentations_mit": "benchmark/transforms/albumentations_mit_video_impl.py",
    "torchvision": "benchmark/transforms/torchvision_video_impl.py",
    "kornia": "benchmark/transforms/kornia_video_impl.py",
}

_REQUIREMENTS: dict[str, str] = {
    "albumentationsx": "requirements/albumentationsx.txt",
    "albumentations_mit": "requirements/albumentations_mit.txt",
    "torchvision": "requirements/torchvision.txt",
    "kornia": "requirements/kornia.txt",
}

_VIDEO_REQUIREMENTS: dict[str, str] = {
    "albumentationsx": "requirements/albumentationsx.txt",
    "albumentations_mit": "requirements/albumentations_mit.txt",
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

    logger.info("Installing dependencies for %s (%s)...", library, media)
    subprocess.run([str(python), "-m", "pip", "install", "-q", "-U", "uv"], check=True)

    base_req = repo_root / "requirements" / "requirements.txt"
    subprocess.run(
        [str(python), "-m", "uv", "pip", "install", "-q", "-U", "-r", str(base_req)],
        check=True,
    )

    req_map = _VIDEO_REQUIREMENTS if media == "video" else _REQUIREMENTS
    lib_req = repo_root / req_map[library]
    subprocess.run(
        [str(python), "-m", "uv", "pip", "install", "-q", "-U", "--force-reinstall", "-r", str(lib_req)],
        check=True,
    )
    logger.info("Dependencies ready for %s", library)

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
    verbose: bool = False,
    num_channels: int = 3,
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
    if num_channels != 3:
        cmd += ["--num-channels", str(num_channels)]

    import os

    env_extra: dict[str, str] = {}
    if transforms_filter:
        env_extra["BENCHMARK_TRANSFORMS_FILTER"] = ",".join(transforms_filter)
    if verbose:
        env_extra["BENCHMARK_VERBOSE"] = "1"

    env = {**os.environ, **env_extra}

    logger.info("Running %s %s benchmark → %s", library, media, output_file)
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# GCP cloud helper
# ---------------------------------------------------------------------------


def build_gcp_benchmark_cli_argv(
    args: argparse.Namespace,
    *,
    data_dir: str,
    output: str,
    repo_root: Path,
) -> list[str]:
    """Build argv for ``python -m benchmark.cli run`` on the VM (no cloud flags)."""
    argv: list[str] = [
        "--data-dir",
        data_dir,
        "--output",
        output,
        "--media",
        args.media,
        "--num-runs",
        str(args.num_runs),
        "--warmup-window",
        str(args.warmup_window),
        "--warmup-threshold",
        str(args.warmup_threshold),
        "--min-warmup-windows",
        str(args.min_warmup_windows),
        "--num-channels",
        str(args.num_channels),
    ]
    if args.num_items is not None:
        argv += ["--num-items", str(args.num_items)]
    if args.max_warmup is not None:
        argv += ["--max-warmup", str(args.max_warmup)]
    if args.libraries:
        argv += ["--libraries", *args.libraries]
    if args.transforms:
        argv += ["--transforms", *args.transforms]
    if args.spec:
        spec_path = Path(args.spec).resolve()
        try:
            rel = spec_path.relative_to(repo_root.resolve())
        except ValueError as e:
            msg = "--spec must be inside the repository when using --cloud gcp"
            raise ValueError(msg) from e
        argv += ["--spec", str(rel)]
    if getattr(args, "multichannel", False):
        argv.append("--multichannel")
    if args.verbose:
        argv.append("--verbose")
    return argv


def _cmd_run_gcp(args: argparse.Namespace, repo_root: Path, local_output_dir: Path) -> None:
    """Run benchmarks on a GCP instance (detached by default)."""
    from benchmark.cloud.gcp import GCPRunner, build_gcp_job_dict, new_run_id
    from benchmark.cloud.instance import GCPInstanceConfig

    if not args.gcp_project:
        logger.error("--gcp-project is required when using --cloud gcp")
        sys.exit(1)

    remote_repo_dir = args.gcp_remote_repo_dir

    if args.gcp_attached:
        if not args.gcp_remote_data_dir:
            logger.error("--gcp-remote-data-dir is required with --gcp-attached (path to dataset on the VM)")
            sys.exit(1)
        remote_data_dir = args.gcp_remote_data_dir
        remote_output = f"{remote_repo_dir}/results"
        try:
            bench_argv = build_gcp_benchmark_cli_argv(
                args,
                data_dir=remote_data_dir,
                output=remote_output,
                repo_root=repo_root,
            )
        except ValueError as e:
            logger.error("%s", e)  # noqa: TRY400
            sys.exit(1)

        config = GCPInstanceConfig(
            project=args.gcp_project,
            zone=args.gcp_zone,
            machine_type=args.gcp_machine_type,
            accelerator_type=args.gcp_gpu_type,
            accelerator_count=1 if args.gcp_gpu_type else 0,
            image_family="pytorch-latest-gpu" if args.gcp_gpu_type else "pytorch-latest-cpu",
            disk_size_gb=args.gcp_disk_size_gb,
        )
        runner = GCPRunner(config)
        runner.run_attached(
            repo_root=repo_root,
            remote_cli_args=bench_argv,
            local_output_dir=local_output_dir,
            remote_repo_dir=remote_repo_dir,
            keep_instance=args.gcp_keep_instance,
        )
        return

    if not args.gcp_gcs_data_uri or not args.gcp_gcs_results_uri:
        logger.error("Detached GCP runs require --gcp-gcs-data-uri and --gcp-gcs-results-uri")
        sys.exit(1)

    run_id = new_run_id()
    machine_slug = args.gcp_machine_type.replace("/", "-").lower()[:24]
    instance_name = f"benchmark-{machine_slug}-{run_id[:12]}".lower().replace("_", "-")

    try:
        bench_argv = build_gcp_benchmark_cli_argv(
            args,
            data_dir="/root/benchmark-data",
            output="/root/benchmark-work/results",
            repo_root=repo_root,
        )
    except ValueError as e:
        logger.error("%s", e)  # noqa: TRY400
        sys.exit(1)

    submission = {
        "argv": sys.argv,
        "start_timestamp_unix": time.time(),
    }
    instance_meta = {
        "project": args.gcp_project,
        "zone": args.gcp_zone,
        "machine_type": args.gcp_machine_type,
        "accelerator_type": args.gcp_gpu_type,
        "instance_name": instance_name,
    }
    job = build_gcp_job_dict(
        run_id=run_id,
        gcs_data_uri=args.gcp_gcs_data_uri,
        benchmark_cli_args=bench_argv,
        terminate_instance=not args.gcp_keep_instance,
        submission=submission,
        instance_meta=instance_meta,
    )

    config = GCPInstanceConfig(
        project=args.gcp_project,
        zone=args.gcp_zone,
        machine_type=args.gcp_machine_type,
        accelerator_type=args.gcp_gpu_type,
        accelerator_count=1 if args.gcp_gpu_type else 0,
        image_family="pytorch-latest-gpu" if args.gcp_gpu_type else "pytorch-latest-cpu",
        disk_size_gb=args.gcp_disk_size_gb,
        instance_name_override=instance_name,
    )
    runner = GCPRunner(config)

    if args.gcp_dry_run:
        prefix = runner.run_detached(
            repo_root=repo_root,
            gcs_data_uri=args.gcp_gcs_data_uri,
            gcs_results_base_uri=args.gcp_gcs_results_uri,
            job=dict(job),
            dry_run=True,
        )
        logger.info("Dry run complete (no uploads or VM). Run prefix would be: %s", prefix)
        return

    run_prefix = runner.run_detached(
        repo_root=repo_root,
        gcs_data_uri=args.gcp_gcs_data_uri,
        gcs_results_base_uri=args.gcp_gcs_results_uri,
        job=dict(job),
        dry_run=False,
    )

    meta_path = local_output_dir / "gcp_last_run.json"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_prefix": run_prefix,
        "run_id": run_id,
        "instance_name": instance_name,
        "zone": args.gcp_zone,
        "project": args.gcp_project,
        "gcs_data_uri": args.gcp_gcs_data_uri,
        "gcs_results_base_uri": args.gcp_gcs_results_uri.rstrip("/"),
        "terminate_instance": not args.gcp_keep_instance,
        "fetch_results_hint": f"gcloud storage cp -r {run_prefix}/results/* {local_output_dir}/",
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Detached GCP run submitted. Instance %s starting; artifacts under %s\n"
        "Local metadata written to %s\n"
        "When finished, fetch results e.g.:\n  %s",
        instance_name,
        run_prefix,
        meta_path,
        payload["fetch_results_hint"],
    )


# ---------------------------------------------------------------------------
# `run` subcommand
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).parent.parent.resolve()
    media: str = args.media
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --multichannel: use 9ch specs, output to output/multichannel/
    if getattr(args, "multichannel", False) and media == "image":
        args.num_channels = 9
        if args.cloud != "gcp":
            output_dir = output_dir / "multichannel"
            output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cloud path: delegate the whole run to a GCP instance
    # ------------------------------------------------------------------
    if args.cloud == "gcp":
        _cmd_run_gcp(args, repo_root, output_dir)
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
            verbose=args.verbose,
            num_channels=args.num_channels,
        )
        return

    # Built-in libraries
    if getattr(args, "multichannel", False) and media == "image":
        spec_map = _MULTICHANNEL_IMAGE_SPECS
    else:
        spec_map = _VIDEO_SPECS if media == "video" else _IMAGE_SPECS
    available = list(spec_map.keys())

    requested: list[str] = args.libraries or available
    unknown = set(requested) - set(available)
    if unknown:
        logger.error("Unknown libraries for %s mode: %s. Available: %s", media, sorted(unknown), available)
        sys.exit(1)

    suffix = "_video" if media == "video" else ""
    logger.info("Running %s benchmarks for %d libraries: %s", media, len(requested), requested)
    for library in tqdm(requested, desc="Libraries", unit="lib", **tqdm_kwargs()):
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
            verbose=args.verbose,
            num_channels=args.num_channels,
        )

    logger.info("All benchmarks complete. Results in: %s", output_dir)


# ---------------------------------------------------------------------------
# `compare` subcommand  (Todo 4: regression detection)
# ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline)
    current_dir = Path(args.current)

    if not baseline_dir.exists():
        logger.error("Baseline directory not found: %s", baseline_dir)
        sys.exit(1)
    if not current_dir.exists():
        logger.error("Current directory not found: %s", current_dir)
        sys.exit(1)

    from tools.compare import compare_regression

    compare_regression(
        baseline_dir=baseline_dir,
        current_dir=current_dir,
        libraries_filter=args.libraries,
        transforms_filter=args.transforms,
        threshold=args.threshold,
        fail_on_regression=args.fail_on_regression,
    )


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
    run_p.add_argument(
        "--gcp-remote-data-dir",
        metavar="PATH",
        help="Attached mode: dataset path on the VM (required with --gcp-attached)",
    )
    run_p.add_argument(
        "--gcp-remote-repo-dir",
        metavar="PATH",
        default="~/benchmark",
        help="Remote directory for the repo extract and results (default: ~/benchmark)",
    )
    run_p.add_argument(
        "--gcp-gcs-data-uri",
        metavar="GS_URI",
        help="Detached mode: gs:// URI of dataset directory to rsync onto the VM (required unless --gcp-attached)",
    )
    run_p.add_argument(
        "--gcp-gcs-results-uri",
        metavar="GS_URI",
        help="Detached mode: gs:// URI prefix for run artifacts (required unless --gcp-attached)",
    )
    run_p.add_argument(
        "--gcp-attached",
        action="store_true",
        help="Use blocking SSH workflow (upload repo, run, download results) instead of detached startup-script",
    )
    run_p.add_argument(
        "--gcp-dry-run",
        action="store_true",
        help="Detached mode: print job.json and exit without uploading or creating a VM",
    )
    run_p.add_argument(
        "--gcp-disk-size-gb",
        type=int,
        default=100,
        metavar="N",
        help="Boot disk size in GB (default: 100)",
    )
    run_p.add_argument("--gcp-keep-instance", action="store_true", help="Do not delete instance after run (debug)")

    run_p.add_argument("--num-items", "-n", type=int, help="Number of images/videos (default depends on media type)")
    run_p.add_argument("--num-runs", "-r", type=int, default=5, help="Benchmark runs per transform (default: 5)")
    run_p.add_argument("--max-warmup", type=int, help="Max warmup iterations (default depends on media type)")
    run_p.add_argument("--warmup-window", type=int, default=5)
    run_p.add_argument("--warmup-threshold", type=float, default=0.05)
    run_p.add_argument("--min-warmup-windows", type=int, default=3)
    run_p.add_argument(
        "--num-channels",
        type=int,
        default=3,
        help=(
            "Number of image channels (must be multiple of 3). Values > 3 stack the RGB source image "
            "to synthesize multi-channel data, e.g. 9 for 3x stacked RGB (default: 3)"
        ),
    )
    run_p.add_argument(
        "--multichannel",
        action="store_true",
        help=(
            "Use multi-channel specs (9ch) and output to <output>/multichannel/. "
            "Implies --num-channels 9 for image mode."
        ),
    )

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

    configure_logging(
        logging.DEBUG if args.verbose else logging.INFO,
        fmt="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
