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

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml  # type: ignore[import-untyped,unused-ignore]
from pydantic import ValidationError
from tqdm import tqdm

from benchmark.config import (
    BenchmarkRunConfig,
    apply_cli_overrides,
    build_run_plan,
    install_run_config_env,
    load_run_config,
    resolve_config_transform_set,
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
    spec_map_for_scenario,
)
from benchmark.orchestrator import execute_job
from benchmark.output_naming import manual_micro_output_file, micro_output_file, pipeline_output_file
from benchmark.parser import build_parser, collect_provided_flags
from benchmark.term import configure_logging, tqdm_kwargs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse


def _extract_library(spec_file: Path) -> str:
    """Extract LIBRARY string from a spec file without importing it."""
    for line in spec_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("LIBRARY"):
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"').strip("'")
    raise ValueError(f"Could not find LIBRARY assignment in {spec_file}")


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
    spec_map = spec_map_for_scenario(scenario.name, mode)
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


def _cmd_run_gcp(
    args: argparse.Namespace,
    repo_root: Path,
    local_output_dir: Path,
    run_config: BenchmarkRunConfig | None = None,
) -> None:
    from benchmark.cloud.launch import run_gcp

    run_gcp(args=args, repo_root=repo_root, local_output_dir=local_output_dir, run_config=run_config)


# ---------------------------------------------------------------------------
# `run` subcommand
# ---------------------------------------------------------------------------


def _resolve_run_config(args: argparse.Namespace) -> BenchmarkRunConfig:
    if not getattr(args, "resolved_config", None) and not getattr(args, "config", None):
        logger.error("Invalid benchmark run config: run requires --config or --resolved-config")
        sys.exit(1)
    try:
        if getattr(args, "resolved_config", None):
            return load_run_config(Path(args.resolved_config))
        return apply_cli_overrides(load_run_config(Path(args.config)), args)
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    vars(args)["_provided_flags"] = collect_provided_flags(sys.argv[1:])

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
