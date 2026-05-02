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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped,unused-ignore]
from pydantic import ValidationError
from tqdm import tqdm

from benchmark import envs
from benchmark.cloud.paths import VM_RESULTS, staged_data_dir_for_gcs_uri
from benchmark.config import (
    BenchmarkRunConfig,
    apply_cli_overrides,
    build_run_cli_argv_from_config,
    build_run_plan,
    install_run_config_env,
    load_run_config,
    remote_run_config_payload,
    repo_relative_spec_path,
    resolve_config_transform_set,
    run_config_from_args,
    run_config_payload,
    write_resolved_config,
)
from benchmark.devices import ensure_supported_device
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
    requirements_for_env_group,
    spec_map_for_scenario,
)
from benchmark.orchestrator import execute_job
from benchmark.output_naming import manual_micro_output_file, micro_output_file, pipeline_output_file
from benchmark.term import configure_logging, tqdm_kwargs

logger = logging.getLogger(__name__)


def _collect_provided_flags(argv: list[str]) -> set[str]:
    return {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}


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


def _manual_micro_config(config: BenchmarkRunConfig) -> BenchmarkRunConfig:
    if config.resolved_mode() == "micro":
        return config
    data = config.model_dump()
    data["selection"]["mode"] = "micro"
    data["execution"]["workers"] = 0
    return BenchmarkRunConfig.model_validate(data)


def _run_micro_job(
    *,
    library: str,
    spec_file: Path,
    data_dir: Path,
    output_file: Path,
    run_config: BenchmarkRunConfig,
    repo_root: Path,
    verbose: bool,
) -> None:
    config = _manual_micro_config(run_config)
    job = BenchmarkJob.from_run_config(
        library=library,
        config=config,
        data_dir=data_dir,
        output_file=output_file,
        num_channels=config.data.num_channels,
        clip_length=config.data.clip_length or 16,
        spec_file=spec_file,
        backend="pyperf",
    )
    execute_job(job, repo_root=repo_root, verbose=verbose)


def _spec_map_for_scenario(scenario_name: str, mode: str) -> dict[str, str]:
    return spec_map_for_scenario(scenario_name, mode)


def _required_data_dir(config: BenchmarkRunConfig) -> Path:
    if not config.data.data_dir:
        msg = "data.data_dir is required for benchmark execution"
        raise ValueError(msg)
    return Path(config.data.data_dir)


def _run_scenario_library(
    *,
    library: str,
    spec_map: dict[str, str],
    run_config: BenchmarkRunConfig,
    repo_root: Path,
    output_dir: Path,
    media: Literal["image", "video"],
    num_channels: int,
    clip_length: int,
    verbose: bool,
) -> None:
    mode = run_config.resolved_mode()
    backend: Literal["dali_pipeline"] | None = "dali_pipeline" if mode == "pipeline" and library == "dali" else None
    spec_file = None if backend == "dali_pipeline" else repo_root / spec_map[library]
    output_file = (
        pipeline_output_file(
            output_dir,
            library,
            pipeline_scope=run_config.execution.pipeline_scope,
            num_items=run_config.data.num_items,
            num_runs=run_config.execution.num_runs,
            workers=run_config.execution.workers,
            batch_size=run_config.execution.batch_size,
            device=run_config.execution.device,
        )
        if mode == "pipeline"
        else micro_output_file(output_dir, library, device=run_config.execution.device)
    )
    try:
        ensure_supported_device(library, media, run_config.execution.device)
    except ValueError as e:
        logger.error("%s", e)  # noqa: TRY400
        sys.exit(1)

    if mode == "micro":
        if spec_file is None:
            msg = f"{library} micro job requires a spec file"
            raise ValueError(msg)
        job = BenchmarkJob.from_run_config(
            library=library,
            config=run_config,
            data_dir=_required_data_dir(run_config),
            output_file=output_file,
            num_channels=num_channels,
            clip_length=clip_length,
            spec_file=spec_file,
            backend="pyperf",
        )
        execute_job(job, repo_root=repo_root, verbose=verbose)
        return

    job = BenchmarkJob.from_run_config(
        library=library,
        config=run_config,
        data_dir=_required_data_dir(run_config),
        output_file=output_file,
        num_channels=num_channels,
        clip_length=clip_length,
        spec_file=spec_file,
        backend=backend,
    )
    execute_job(job, repo_root=repo_root)


def _cmd_run_scenario(
    *,
    run_config: BenchmarkRunConfig,
    repo_root: Path,
    output_dir: Path,
    verbose: bool,
) -> None:
    from benchmark.decode_runner import VideoDecodeRunner
    from benchmark.scenarios import get_scenario, resolve_decoders, resolve_libraries, resolve_mode

    if run_config.selection.scenario is None:
        msg = "selection.scenario is required for scenario execution"
        raise ValueError(msg)
    scenario = get_scenario(run_config.selection.scenario)
    mode = resolve_mode(scenario, run_config.selection.mode)
    clip_length = run_config.data.clip_length or scenario.clip_length or 16
    num_channels = scenario.num_channels

    if mode == "decode":
        decoders = resolve_decoders(scenario, run_config.selection.decoders)
        runner = VideoDecodeRunner(
            data_dir=_required_data_dir(run_config),
            decoders=decoders,
            output_dir=output_dir,
            num_items=run_config.data.num_items,
            num_runs=run_config.execution.num_runs,
            clip_length=clip_length,
            scenario=scenario.name,
            min_time=run_config.execution.min_time,
        )
        runner.run()
        return

    libraries = resolve_libraries(scenario, mode, run_config.selection.libraries)
    spec_map = _spec_map_for_scenario(scenario.name, mode)
    scenario_output_dir = output_dir / scenario.name / mode
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running scenario %s/%s for libraries: %s", scenario.name, mode, libraries)
    for library in tqdm(libraries, desc=f"{scenario.name}/{mode}", unit="lib", **tqdm_kwargs()):
        _run_scenario_library(
            library=library,
            spec_map=spec_map,
            run_config=run_config,
            repo_root=repo_root,
            output_dir=scenario_output_dir,
            media=scenario.media,
            num_channels=num_channels,
            clip_length=clip_length,
            verbose=verbose,
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
    media = args.media
    if getattr(args, "scenario", None):
        from benchmark.scenarios import get_scenario

        media = get_scenario(args.scenario).media

    argv: list[str] = [
        "--data-dir",
        data_dir,
        "--output",
        output,
        "--media",
        media,
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
        argv += ["--spec", repo_relative_spec_path(str(args.spec), repo_root)]
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


@dataclass(frozen=True)
class _GcpLaunchOptions:
    project: str | None
    zone: str
    machine_type: str
    gpu_type: str | None
    attached: bool
    disk_size_gb: int
    keep_instance: bool
    keep_on_failure: bool
    preemptible: bool
    remote_repo_dir: str
    venv_cache_uri: str | None
    no_venv_cache: bool
    force_venv_cache_rebuild: bool
    dry_run: bool
    remote_data_dir: str | None
    gcs_data_uri: str | None
    gcs_results_uri: str | None


def _gcp_launch_options(args: argparse.Namespace, run_config: BenchmarkRunConfig | None) -> _GcpLaunchOptions:
    cloud = run_config.cloud if run_config and run_config.cloud else None
    return _GcpLaunchOptions(
        project=cloud.project if cloud else args.gcp_project,
        zone=cloud.zone if cloud else args.gcp_zone,
        machine_type=cloud.machine_type if cloud else args.gcp_machine_type,
        gpu_type=cloud.gpu_type if cloud else args.gcp_gpu_type,
        attached=cloud.attached if cloud else args.gcp_attached,
        disk_size_gb=cloud.disk_size_gb if cloud else args.gcp_disk_size_gb,
        keep_instance=cloud.keep_instance if cloud else args.gcp_keep_instance,
        keep_on_failure=cloud.keep_on_failure if cloud else args.gcp_keep_on_failure,
        preemptible=cloud.preemptible if cloud else args.gcp_preemptible,
        remote_repo_dir=cloud.remote_repo_dir if cloud else args.gcp_remote_repo_dir,
        venv_cache_uri=cloud.venv_cache_uri if cloud else args.gcp_venv_cache_uri,
        no_venv_cache=cloud.no_venv_cache if cloud else args.gcp_no_venv_cache,
        force_venv_cache_rebuild=cloud.force_venv_cache_rebuild if cloud else args.gcp_force_venv_cache_rebuild,
        dry_run=cloud.dry_run if cloud else args.gcp_dry_run,
        remote_data_dir=run_config.data.remote_data_dir if run_config else args.gcp_remote_data_dir,
        gcs_data_uri=run_config.data.gcs_uri if run_config else args.gcp_gcs_data_uri,
        gcs_results_uri=run_config.output.gcs_results_uri if run_config else args.gcp_gcs_results_uri,
    )


def _cmd_run_gcp(
    args: argparse.Namespace,
    repo_root: Path,
    local_output_dir: Path,
    run_config: BenchmarkRunConfig | None = None,
) -> None:
    """Run benchmarks on a GCP instance (detached by default)."""
    from benchmark.cloud.gcp import GCPRunner, build_gcp_job_dict, new_run_id
    from benchmark.cloud.instance import GCPInstanceConfig, is_gpu_machine_type

    options = _gcp_launch_options(args, run_config)

    if not options.project:
        logger.error("--gcp-project is required when using --cloud gcp")
        sys.exit(1)

    gpu_machine = bool(options.gpu_type) or is_gpu_machine_type(options.machine_type)
    image_family = "pytorch-2-9-cu129-ubuntu-2404-nvidia-580" if gpu_machine else "ubuntu-2404-lts-amd64"
    image_project = "deeplearning-platform-release" if gpu_machine else "ubuntu-os-cloud"

    if options.attached:
        if not options.remote_data_dir:
            logger.error("--gcp-remote-data-dir is required with --gcp-attached (path to dataset on the VM)")
            sys.exit(1)
        remote_output = f"{options.remote_repo_dir}/results"
        try:
            bench_argv = (
                build_run_cli_argv_from_config(
                    run_config,
                    data_dir=options.remote_data_dir,
                    output=remote_output,
                    repo_root=repo_root,
                    verbose=getattr(args, "verbose", False),
                )
                if run_config
                else build_gcp_benchmark_cli_argv(
                    args,
                    data_dir=options.remote_data_dir,
                    output=remote_output,
                    repo_root=repo_root,
                )
            )
        except ValueError as e:
            logger.error("%s", e)  # noqa: TRY400
            sys.exit(1)
        config = GCPInstanceConfig(
            project=options.project,
            zone=options.zone,
            machine_type=options.machine_type,
            accelerator_type=options.gpu_type,
            accelerator_count=1 if options.gpu_type else 0,
            image_family=image_family,
            image_project=image_project,
            disk_size_gb=options.disk_size_gb,
            preemptible=options.preemptible,
        )
        runner = GCPRunner(config)
        runner.run_attached(
            repo_root=repo_root,
            remote_cli_args=bench_argv,
            local_output_dir=local_output_dir,
            remote_repo_dir=options.remote_repo_dir,
            keep_instance=options.keep_instance,
        )
        return

    if not options.gcs_data_uri or not options.gcs_results_uri:
        logger.error("Detached GCP runs require --gcp-gcs-data-uri and --gcp-gcs-results-uri")
        sys.exit(1)

    run_id = new_run_id()
    machine_slug = options.machine_type.replace("/", "-").lower()[:24]
    instance_name = f"benchmark-{machine_slug}-{run_id[:12]}".lower().replace("_", "-")

    try:
        staged_data_dir = staged_data_dir_for_gcs_uri(options.gcs_data_uri)
        bench_argv = (
            build_run_cli_argv_from_config(
                run_config,
                data_dir=staged_data_dir,
                output=VM_RESULTS,
                repo_root=repo_root,
                verbose=getattr(args, "verbose", False),
            )
            if run_config
            else build_gcp_benchmark_cli_argv(
                args,
                data_dir=staged_data_dir,
                output=VM_RESULTS,
                repo_root=repo_root,
            )
        )
    except ValueError as e:
        logger.error("%s", e)  # noqa: TRY400
        sys.exit(1)
    remote_config = (
        remote_run_config_payload(
            run_config,
            data_dir=staged_data_dir,
            output_dir=VM_RESULTS,
        )
        if run_config
        else None
    )

    submission = {
        "argv": sys.argv,
        "start_timestamp_unix": time.time(),
    }
    instance_meta = {
        "project": options.project,
        "zone": options.zone,
        "machine_type": options.machine_type,
        "accelerator_type": options.gpu_type,
        "instance_name": instance_name,
    }
    job = build_gcp_job_dict(
        run_id=run_id,
        gcs_data_uri=options.gcs_data_uri,
        benchmark_cli_args=bench_argv,
        run_config=remote_config,
        cloud_config=run_config_payload(run_config).get("cloud") if run_config and run_config.cloud else None,
        terminate_instance=not options.keep_instance,
        keep_instance_on_failure=options.keep_on_failure,
        venv_cache_uri=""
        if options.no_venv_cache
        else options.venv_cache_uri or _default_gcp_venv_cache_uri(options.gcs_results_uri),
        force_venv_cache_rebuild=options.force_venv_cache_rebuild,
        submission=submission,
        instance_meta=instance_meta,
    )

    config = GCPInstanceConfig(
        project=options.project,
        zone=options.zone,
        machine_type=options.machine_type,
        accelerator_type=options.gpu_type,
        accelerator_count=1 if options.gpu_type else 0,
        image_family=image_family,
        image_project=image_project,
        disk_size_gb=options.disk_size_gb,
        preemptible=options.preemptible,
        instance_name_override=instance_name,
    )
    runner = GCPRunner(config)

    if options.dry_run:
        prefix = runner.run_detached(
            repo_root=repo_root,
            gcs_data_uri=options.gcs_data_uri,
            gcs_results_base_uri=options.gcs_results_uri,
            job=dict(job),
            dry_run=True,
        )
        logger.info("Dry run complete (no uploads or VM). Run prefix would be: %s", prefix)
        return

    run_prefix = runner.run_detached(
        repo_root=repo_root,
        gcs_data_uri=options.gcs_data_uri,
        gcs_results_base_uri=options.gcs_results_uri,
        job=dict(job),
        dry_run=False,
    )

    meta_path = local_output_dir / "gcp_last_run.json"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_prefix": run_prefix,
        "run_id": run_id,
        "instance_name": instance_name,
        "zone": options.zone,
        "project": options.project,
        "gcs_data_uri": options.gcs_data_uri,
        "gcs_results_base_uri": options.gcs_results_uri.rstrip("/"),
        "terminate_instance": not options.keep_instance,
        "keep_instance_on_failure": options.keep_on_failure,
        "venv_cache_uri": job["venv_cache_uri"],
        "force_venv_cache_rebuild": options.force_venv_cache_rebuild,
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


def _resolve_run_config(args: argparse.Namespace) -> BenchmarkRunConfig:
    try:
        if getattr(args, "resolved_config", None):
            return load_run_config(Path(args.resolved_config))
        if getattr(args, "config", None):
            return apply_cli_overrides(load_run_config(Path(args.config)), args)
        return run_config_from_args(args)
    except (TypeError, ValidationError, ValueError) as e:
        logger.error("Invalid benchmark run config: %s", e)  # noqa: TRY400
        sys.exit(1)


def _log_run_summary(config: BenchmarkRunConfig) -> None:
    cloud = config.cloud.provider if config.cloud and config.cloud.enabled else "local"
    logger.info(
        "Resolved run: scenario=%s mode=%s libraries=%s data=%s output=%s device=%s cloud=%s",
        config.selection.scenario or "manual",
        config.selection.mode or "default",
        config.selection.libraries or "default",
        config.data.gcs_uri or config.data.data_dir,
        config.output.output_dir,
        config.execution.device,
        cloud,
    )


def _plan_payload(config: BenchmarkRunConfig, repo_root: Path) -> dict[str, object]:
    plan = build_run_plan(config, repo_root)
    return {
        "resolved_config": json.loads(config.model_dump_json(exclude_none=True)),
        "plan": plan.to_dict(),
    }


def _print_dry_run(config: BenchmarkRunConfig, repo_root: Path) -> None:
    payload = _plan_payload(config, repo_root)
    print(yaml.safe_dump(payload, sort_keys=False))


def cmd_run(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).parent.parent.resolve()
    verbose = args.verbose
    run_config = _resolve_run_config(args)
    run_config = resolve_config_transform_set(run_config, repo_root)
    _log_run_summary(run_config)
    install_run_config_env(run_config)
    if args.dry_run:
        _print_dry_run(run_config, repo_root)
        return
    media = run_config.resolved_media()
    output_dir = Path(run_config.output.output_dir or "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(run_config, output_dir / "resolved_config.yaml")

    # --multichannel: use 9ch specs, output to output/multichannel/
    if (
        run_config.selection.multichannel
        and media == "image"
        and not (run_config.cloud and run_config.cloud.provider == "gcp")
    ):
        output_dir = output_dir / "multichannel"
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cloud path: delegate the whole run to a GCP instance
    # ------------------------------------------------------------------
    if run_config.cloud and run_config.cloud.provider == "gcp":
        _cmd_run_gcp(args, repo_root, output_dir, run_config=run_config)
        return

    if run_config.selection.scenario:
        _cmd_run_scenario(run_config=run_config, repo_root=repo_root, output_dir=output_dir, verbose=verbose)
        logger.info("Scenario benchmark complete. Results in: %s", output_dir)
        return

    # Custom spec file path takes priority
    if run_config.selection.spec:
        spec_file = Path(run_config.selection.spec)
        if not spec_file.is_absolute():
            spec_file = repo_root / spec_file
        library = _extract_library(spec_file)
        try:
            ensure_supported_device(library, media, run_config.execution.device)
        except ValueError as e:
            logger.error("%s", e)  # noqa: TRY400
            sys.exit(1)
        output_file = output_dir / f"{spec_file.stem}.json"
        _run_micro_job(
            library=library,
            spec_file=spec_file,
            data_dir=_required_data_dir(run_config),
            output_file=output_file,
            run_config=run_config,
            repo_root=repo_root,
            verbose=verbose,
        )
        return

    # Built-in libraries
    if run_config.selection.multichannel and media == "image":
        spec_map = _MULTICHANNEL_IMAGE_SPECS
    else:
        spec_map = _VIDEO_SPECS if media == "video" else _IMAGE_SPECS
    available = list(spec_map.keys())

    requested: list[str] = run_config.selection.libraries or available
    unknown = set(requested) - set(available)
    if unknown:
        logger.error("Unknown libraries for %s mode: %s. Available: %s", media, sorted(unknown), available)
        sys.exit(1)

    logger.info("Running %s benchmarks for %d libraries: %s", media, len(requested), requested)
    for library in tqdm(requested, desc="Libraries", unit="lib", **tqdm_kwargs()):
        try:
            ensure_supported_device(library, media, run_config.execution.device)
        except ValueError as e:
            logger.error("%s", e)  # noqa: TRY400
            sys.exit(1)
        spec_file = repo_root / spec_map[library]
        output_file = manual_micro_output_file(output_dir, library, media=media, device=run_config.execution.device)
        _run_micro_job(
            library=library,
            spec_file=spec_file,
            data_dir=_required_data_dir(run_config),
            output_file=output_file,
            run_config=run_config,
            repo_root=repo_root,
            verbose=verbose,
        )

    logger.info("All benchmarks complete. Results in: %s", output_dir)


def cmd_plan(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).parent.parent.resolve()
    run_config = _resolve_run_config(args)
    run_config = resolve_config_transform_set(run_config, repo_root)
    _log_run_summary(run_config)
    payload = _plan_payload(run_config, repo_root)
    print(yaml.safe_dump(payload, sort_keys=False))


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
    run_p.add_argument("--config", type=Path, help="YAML benchmark run config")
    run_p.add_argument("--resolved-config", type=Path, help=argparse.SUPPRESS)
    run_p.add_argument("--dry-run", action="store_true", help="Print the resolved config and exit without running")
    run_p.add_argument("--data-dir", "-d", help="Directory with images or videos")
    run_p.add_argument("--output", "-o", help="Directory to write result JSON files")
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
    # plan
    # ------------------------------------------------------------------
    plan_p = subparsers.add_parser("plan", help="Print resolved config, generated jobs, and expected outputs")
    plan_p.add_argument("--config", type=Path, required=True, help="YAML benchmark run config")
    plan_p.add_argument("--resolved-config", type=Path, help=argparse.SUPPRESS)
    plan_p.add_argument("--output", "-o", help="Override output.output_dir")
    plan_p.add_argument("--num-items", "-n", type=int, help="Override data.num_items")
    plan_p.add_argument("--num-runs", "-r", type=int, help="Override execution.num_runs")
    plan_p.add_argument("--device", choices=["none", "cuda", "mps", "auto"], help="Override execution.device")
    plan_p.add_argument("--workers", type=int, help="Override execution.workers")
    plan_p.add_argument("--batch-size", type=int, help="Override execution.batch_size")
    plan_p.add_argument("--gcp-dry-run", action="store_true", help="Override cloud.dry_run")

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
    vars(args)["_provided_flags"] = _collect_provided_flags(sys.argv[1:])

    configure_logging(
        logging.DEBUG if args.verbose else logging.INFO,
        fmt="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "run":
        cmd_run(args)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "validate-results":
        cmd_validate_results(args)


if __name__ == "__main__":
    main()
