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
import sys
import time
from pathlib import Path
from typing import Literal, TypedDict, cast

from tqdm import tqdm

from benchmark import envs
from benchmark.jobs import BenchmarkJob
from benchmark.matrix import (
    IMAGE_SPECS as _IMAGE_SPECS,
)
from benchmark.matrix import (
    MULTICHANNEL_IMAGE_SPECS as _MULTICHANNEL_IMAGE_SPECS,
)
from benchmark.matrix import (
    VIDEO_SPECS as _VIDEO_SPECS,
)
from benchmark.matrix import (
    library_env_group,
    paper_transform_set_file,
    requirements_for_env_group,
    spec_map_for_scenario,
)
from benchmark.orchestrator import execute_job
from benchmark.term import configure_logging, tqdm_kwargs

logger = logging.getLogger(__name__)


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


def _read_markdown_text_block(path: Path) -> list[str]:
    lines: list[str] = []
    in_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "```text":
            in_block = True
            continue
        if in_block and stripped == "```":
            break
        if in_block and stripped:
            lines.append(stripped)
    if not lines:
        raise ValueError(f"No text transform block found in {path}")
    return lines


def _paper_transform_names(repo_root: Path, scenario_name: str, mode: str) -> list[str]:
    transform_set_path = repo_root / paper_transform_set_file(scenario_name)
    names = _read_markdown_text_block(transform_set_path)
    if mode == "pipeline" and scenario_name in {"image-rgb", "image-9ch"}:
        from benchmark.transforms.image_recipe_specs import recipe_name, spec_by_name

        return [recipe_name(spec_by_name(name)) for name in names if name != "Normalize"]
    return names


def _apply_transform_set(args: argparse.Namespace, repo_root: Path, scenario_name: str, mode: str) -> None:
    if args.transform_set is None or args.transforms:
        return
    if args.transform_set == "paper":
        args.transforms = _paper_transform_names(repo_root, scenario_name, mode)
        logger.info("Using %s paper transform set (%d transforms)", scenario_name, len(args.transforms))
        return
    msg = f"Unknown transform set {args.transform_set!r}"
    raise ValueError(msg)


def _compile_requirements(python: Path, requirements_path: Path) -> None:
    envs.compile_requirements(python, requirements_path)


def _requirements_cache_key(
    *,
    python: Path,
    requirements_paths: list[Path],
    env_group: str,
    media: Literal["image", "video"],
) -> str:
    return envs.requirements_cache_key(
        python=python,
        requirements_paths=requirements_paths,
        env_group=env_group,
        media=media,
    )


def _library_env_group(library: str, media: str) -> str:
    return library_env_group(library, media)


def _requirements_for_env_group(env_group: str, media: str, repo_root: Path) -> list[Path]:
    return requirements_for_env_group(env_group, media, repo_root)


class _SlowSkipKwargs(TypedDict):
    slow_threshold_sec_per_item: float | None
    slow_preflight_items: int | None
    disable_slow_skip: bool


def _slow_skip_kwargs(args: argparse.Namespace) -> _SlowSkipKwargs:
    return {
        "slow_threshold_sec_per_item": args.slow_threshold_sec_per_item,
        "slow_preflight_items": args.slow_preflight_items,
        "disable_slow_skip": args.disable_slow_skip,
    }


def _run_single(
    library: str,
    spec_file: Path,
    data_dir: Path,
    output_file: Path,
    media: str,
    num_items: int | None,
    num_runs: int,
    repo_root: Path,
    transforms_filter: list[str] | None = None,
    verbose: bool = False,
    num_channels: int = 3,
    scenario: str = "manual",
    refresh_requirements: bool = True,
    slow_threshold_sec_per_item: float | None = None,
    slow_preflight_items: int | None = None,
    disable_slow_skip: bool = False,
) -> None:
    media_kind = cast("Literal['image', 'video']", media)
    job = BenchmarkJob(
        library=library,
        scenario=scenario,
        mode="micro",
        media=media_kind,
        data_dir=data_dir,
        output_file=output_file,
        num_items=num_items,
        num_runs=num_runs,
        num_channels=num_channels,
        spec_file=spec_file,
        transforms_filter=tuple(transforms_filter or ()),
        refresh_requirements=refresh_requirements,
        slow_threshold_sec_per_item=slow_threshold_sec_per_item,
        slow_preflight_items=slow_preflight_items,
        disable_slow_skip=disable_slow_skip,
    )
    execute_job(job, repo_root=repo_root, verbose=verbose)


def _spec_map_for_scenario(scenario_name: str, mode: str) -> dict[str, str]:
    return spec_map_for_scenario(scenario_name, mode)


def _pipeline_output_file(output_dir: Path, library: str, args: argparse.Namespace) -> Path:
    num_items = f"n{args.num_items}" if args.num_items is not None else "nall"
    device = f"_dev-{args.device}" if args.device != "none" else ""
    stem = f"{library}_{args.pipeline_scope}_{num_items}_r{args.num_runs}_w{args.workers}_b{args.batch_size}{device}"
    return output_dir / f"{stem}_results.json"


def _run_scenario_library(
    *,
    library: str,
    spec_map: dict[str, str],
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
    scenario_name: str,
    media: str,
    num_channels: int,
    clip_length: int,
) -> None:
    backend: Literal["dali_pipeline"] | None = (
        "dali_pipeline" if args.mode == "pipeline" and library == "dali" else None
    )
    media_name = cast("Literal['image', 'video']", media)
    spec_file = None if backend == "dali_pipeline" else repo_root / spec_map[library]
    output_file = (
        _pipeline_output_file(output_dir, library, args)
        if args.mode == "pipeline"
        else output_dir / f"{library}_micro_results.json"
    )

    if args.mode == "micro":
        if spec_file is None:
            msg = f"{library} micro job requires a spec file"
            raise ValueError(msg)
        _run_single(
            library=library,
            spec_file=spec_file,
            data_dir=Path(args.data_dir),
            output_file=output_file,
            media=media_name,
            num_items=args.num_items,
            num_runs=args.num_runs,
            repo_root=repo_root,
            transforms_filter=args.transforms,
            verbose=args.verbose,
            num_channels=num_channels,
            scenario=scenario_name,
            refresh_requirements=args.refresh_requirements,
            **_slow_skip_kwargs(args),
        )
        return

    job = BenchmarkJob.from_args(
        library=library,
        scenario_name=scenario_name,
        mode="pipeline",
        media=media_name,
        data_dir=Path(args.data_dir),
        output_file=output_file,
        args=args,
        num_channels=num_channels,
        clip_length=clip_length,
        spec_file=spec_file,
        backend=backend,
    )
    execute_job(job, repo_root=repo_root)


def _cmd_run_scenario(args: argparse.Namespace, repo_root: Path, output_dir: Path) -> None:
    from benchmark.decode_runner import VideoDecodeRunner
    from benchmark.scenarios import get_scenario, resolve_decoders, resolve_libraries, resolve_mode

    scenario = get_scenario(args.scenario)
    args.mode = resolve_mode(scenario, args.mode)
    try:
        _apply_transform_set(args, repo_root, scenario.name, args.mode)
    except ValueError as e:
        logger.error("%s", e)  # noqa: TRY400
        sys.exit(1)
    if args.thread_policy is None:
        args.thread_policy = "micro-single" if args.mode == "micro" else "pipeline-default"
    clip_length = args.clip_length or scenario.clip_length or 16
    num_channels = scenario.num_channels

    if args.mode == "decode":
        decoders = resolve_decoders(scenario, args.decoders)
        runner = VideoDecodeRunner(
            data_dir=Path(args.data_dir),
            decoders=decoders,
            output_dir=output_dir,
            num_items=args.num_items,
            num_runs=args.num_runs,
            clip_length=clip_length,
            scenario=scenario.name,
            min_time=args.min_time,
        )
        runner.run()
        return

    libraries = resolve_libraries(scenario, args.mode, args.libraries)
    spec_map = _spec_map_for_scenario(scenario.name, args.mode)
    scenario_output_dir = output_dir / scenario.name / args.mode
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running scenario %s/%s for libraries: %s", scenario.name, args.mode, libraries)
    for library in tqdm(libraries, desc=f"{scenario.name}/{args.mode}", unit="lib", **tqdm_kwargs()):
        _run_scenario_library(
            library=library,
            spec_map=spec_map,
            args=args,
            repo_root=repo_root,
            output_dir=scenario_output_dir,
            scenario_name=scenario.name,
            media=scenario.media,
            num_channels=num_channels,
            clip_length=clip_length,
        )


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
        "--num-channels",
        str(args.num_channels),
    ]
    if args.num_items is not None:
        argv += ["--num-items", str(args.num_items)]
    if args.libraries:
        argv += ["--libraries", *args.libraries]
    if args.transforms:
        argv += ["--transforms", *args.transforms]
    if getattr(args, "transform_set", None):
        argv += ["--transform-set", args.transform_set]
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
    if getattr(args, "scenario", None):
        argv += ["--scenario", args.scenario]
    if getattr(args, "mode", None):
        argv += ["--mode", args.mode]
    if getattr(args, "pipeline_scope", None):
        argv += ["--pipeline-scope", args.pipeline_scope]
    if getattr(args, "device", None):
        argv += ["--device", args.device]
    if getattr(args, "thread_policy", None):
        argv += ["--thread-policy", args.thread_policy]
    if getattr(args, "batch_size", None):
        argv += ["--batch-size", str(args.batch_size)]
    if getattr(args, "workers", None) is not None:
        argv += ["--workers", str(args.workers)]
    if getattr(args, "min_time", None):
        argv += ["--min-time", str(args.min_time)]
    if getattr(args, "min_batches", None):
        argv += ["--min-batches", str(args.min_batches)]
    if getattr(args, "clip_length", None):
        argv += ["--clip-length", str(args.clip_length)]
    if getattr(args, "decoders", None):
        argv += ["--decoders", *args.decoders]
    if not getattr(args, "refresh_requirements", True):
        argv.append("--no-refresh-requirements")
    if getattr(args, "slow_threshold_sec_per_item", None) is not None:
        argv += ["--slow-threshold-sec-per-item", str(args.slow_threshold_sec_per_item)]
    if getattr(args, "slow_preflight_items", None) is not None:
        argv += ["--slow-preflight-items", str(args.slow_preflight_items)]
    if getattr(args, "disable_slow_skip", False):
        argv.append("--disable-slow-skip")
    return argv


def _default_gcp_venv_cache_uri(results_uri: str) -> str:
    base = results_uri.rstrip("/")
    parent = base.rsplit("/", 1)[0] if "/" in base.removeprefix("gs://") else base
    return f"{parent}/augmentation-cache"


def _cmd_run_gcp(args: argparse.Namespace, repo_root: Path, local_output_dir: Path) -> None:
    """Run benchmarks on a GCP instance (detached by default)."""
    from benchmark.cloud.gcp import GCPRunner, build_gcp_job_dict, new_run_id
    from benchmark.cloud.instance import GCPInstanceConfig

    if not args.gcp_project:
        logger.error("--gcp-project is required when using --cloud gcp")
        sys.exit(1)

    remote_repo_dir = args.gcp_remote_repo_dir
    image_family = "pytorch-2-9-cu129-ubuntu-2404-nvidia-580" if args.gcp_gpu_type else "ubuntu-2404-lts-amd64"
    image_project = "deeplearning-platform-release" if args.gcp_gpu_type else "ubuntu-os-cloud"

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
            image_family=image_family,
            image_project=image_project,
            disk_size_gb=args.gcp_disk_size_gb,
            preemptible=args.gcp_preemptible,
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

    def _gcp_staged_data_dir() -> str:
        """Where the dataset ended up on the VM after staging from GCS."""
        p = (args.gcp_gcs_data_uri or "").rstrip("/")
        if not p.startswith("gs://"):
            return "/root/benchmark-data"
        base = p.rsplit("/", 1)[-1].lower()
        if base.endswith(".tar") and base.startswith("val"):
            return "/root/benchmark-data/val"
        if base in {"val", "train", "test"}:
            return "/root/benchmark-data/val"
        return "/root/benchmark-data"

    run_id = new_run_id()
    machine_slug = args.gcp_machine_type.replace("/", "-").lower()[:24]
    instance_name = f"benchmark-{machine_slug}-{run_id[:12]}".lower().replace("_", "-")

    try:
        bench_argv = build_gcp_benchmark_cli_argv(
            args,
            data_dir=_gcp_staged_data_dir(),
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
        keep_instance_on_failure=args.gcp_keep_on_failure,
        venv_cache_uri=""
        if args.gcp_no_venv_cache
        else args.gcp_venv_cache_uri or _default_gcp_venv_cache_uri(args.gcp_gcs_results_uri),
        force_venv_cache_rebuild=args.gcp_force_venv_cache_rebuild,
        submission=submission,
        instance_meta=instance_meta,
    )

    config = GCPInstanceConfig(
        project=args.gcp_project,
        zone=args.gcp_zone,
        machine_type=args.gcp_machine_type,
        accelerator_type=args.gcp_gpu_type,
        accelerator_count=1 if args.gcp_gpu_type else 0,
        image_family=image_family,
        image_project=image_project,
        disk_size_gb=args.gcp_disk_size_gb,
        preemptible=args.gcp_preemptible,
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
        "keep_instance_on_failure": args.gcp_keep_on_failure,
        "venv_cache_uri": job["venv_cache_uri"],
        "force_venv_cache_rebuild": args.gcp_force_venv_cache_rebuild,
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

    if args.scenario:
        _cmd_run_scenario(args, repo_root, output_dir)
        logger.info("Scenario benchmark complete. Results in: %s", output_dir)
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
            repo_root=repo_root,
            transforms_filter=args.transforms,
            verbose=args.verbose,
            num_channels=args.num_channels,
            scenario=f"{media}-manual",
            refresh_requirements=args.refresh_requirements,
            **_slow_skip_kwargs(args),
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
            repo_root=repo_root,
            transforms_filter=args.transforms,
            verbose=args.verbose,
            num_channels=args.num_channels,
            scenario=f"{media}-manual",
            refresh_requirements=args.refresh_requirements,
            **_slow_skip_kwargs(args),
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


def cmd_doctor(args: argparse.Namespace) -> None:
    from benchmark.reliability import doctor_report

    report = doctor_report(Path(__file__).parent.parent)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Benchmark doctor")
        print(f"ok: {report['ok']}")
        for warning in report["warnings"]:
            print(f"warning: {warning}")
    if args.fail_on_warning and report["warnings"]:
        sys.exit(1)


def cmd_validate_results(args: argparse.Namespace) -> None:
    from benchmark.reliability import audit_results

    report = audit_results(Path(args.path))
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"checked: {report.files_checked} result file(s)")
        for warning in report.warnings:
            print(f"warning: {warning}")
        for issue in report.issues:
            print(f"error: {issue}")
    if not report.ok:
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
        "--transform-set",
        choices=["paper"],
        help=(
            "Use a named transform set. 'paper' selects transforms present in 2+ selected libraries for the scenario."
        ),
    )
    run_p.add_argument(
        "--spec",
        "-s",
        metavar="FILE",
        help="Custom spec file (overrides --libraries; library inferred from LIBRARY variable)",
    )
    run_p.add_argument(
        "--scenario",
        choices=["image-rgb", "image-9ch", "video-decode-16f", "video-16f"],
        help="Run a benchmark scenario such as image-rgb, image-9ch, video-decode-16f, or video-16f.",
    )
    run_p.add_argument(
        "--mode",
        choices=["micro", "pipeline", "decode"],
        help="Scenario benchmark mode. Defaults depend on --scenario.",
    )
    run_p.add_argument("--batch-size", type=int, default=32, help="Pipeline dataloader batch size")
    run_p.add_argument("--workers", type=int, default=0, help="Pipeline dataloader worker count")
    run_p.add_argument("--min-time", type=float, default=0.0, help="Minimum measured seconds per run")
    run_p.add_argument("--min-batches", type=int, default=1, help="Minimum measured dataloader batches per run")
    run_p.add_argument(
        "--pipeline-scope",
        choices=["memory_dataloader_augment", "decode_dataloader_augment", "decode_dataloader_augment_batch_copy"],
        default="decode_dataloader_augment",
        help="Pipeline measurement scope for --mode pipeline",
    )
    run_p.add_argument(
        "--device",
        choices=["none", "cuda", "mps", "auto"],
        default="none",
        help="Device copy target for batch-copy pipeline scope",
    )
    run_p.add_argument(
        "--thread-policy",
        choices=["micro-single", "pipeline-default", "pipeline-single-worker"],
        help="Thread policy. Defaults to micro-single for micro and pipeline-default for pipeline.",
    )
    run_p.add_argument("--clip-length", type=int, help="Video frames per clip for scenario benchmarks")
    run_p.add_argument(
        "--decoders",
        nargs="+",
        metavar="DECODER",
        help="Video decoders for --scenario video-decode-16f",
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
        help=(
            "Detached mode: gs:// URI of the dataset archive/object to download to the VM "
            "(required unless --gcp-attached)"
        ),
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
    run_p.add_argument(
        "--gcp-keep-on-failure",
        action="store_true",
        help="Detached mode: keep the VM alive only when the startup script or benchmark fails.",
    )
    run_p.add_argument(
        "--gcp-preemptible",
        action="store_true",
        help="Use a preemptible GCP VM. Default is a regular VM for benchmark stability and quota compatibility.",
    )
    run_p.add_argument(
        "--gcp-venv-cache-uri",
        metavar="GS_URI",
        help="GCS prefix for reusable VM venv cache (default: sibling augmentation-cache bucket prefix).",
    )
    run_p.add_argument("--gcp-no-venv-cache", action="store_true", help="Disable GCS venv cache restore/populate.")
    run_p.add_argument(
        "--gcp-force-venv-cache-rebuild",
        action="store_true",
        help="Bypass venv cache lookup and upload a fresh cache after a successful run.",
    )

    run_p.add_argument("--num-items", "-n", type=int, help="Number of images/videos (default depends on media type)")
    run_p.add_argument("--num-runs", "-r", type=int, default=5, help="Benchmark runs per transform (default: 5)")
    run_p.add_argument(
        "--slow-threshold-sec-per-item",
        type=float,
        default=None,
        help="Skip micro/pipeline transforms slower than this many seconds per image/video in preflight.",
    )
    run_p.add_argument(
        "--slow-preflight-items",
        type=int,
        default=None,
        help="Number of images/videos used for slow-transform preflight.",
    )
    run_p.add_argument(
        "--disable-slow-skip",
        action="store_true",
        help="Run exhaustive measurements even when preflight says a transform is slow.",
    )
    run_p.set_defaults(refresh_requirements=True)
    run_p.add_argument(
        "--no-refresh-requirements",
        action="store_false",
        dest="refresh_requirements",
        help="Skip regenerating requirements/*.txt from requirements/*.in before checking the venv dependency cache",
    )
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

    # ------------------------------------------------------------------
    # doctor
    # ------------------------------------------------------------------
    doctor_p = subparsers.add_parser("doctor", help="Check benchmark environment reliability")
    doctor_p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    doctor_p.add_argument("--fail-on-warning", action="store_true", help="Exit 1 when doctor reports warnings")

    # ------------------------------------------------------------------
    # validate-results
    # ------------------------------------------------------------------
    validate_p = subparsers.add_parser("validate-results", help="Audit benchmark result JSON files")
    validate_p.add_argument("path", help="Result JSON file or directory")
    validate_p.add_argument("--json", action="store_true", help="Print machine-readable JSON")

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
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "validate-results":
        cmd_validate_results(args)


if __name__ == "__main__":
    main()
