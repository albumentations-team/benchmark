from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from benchmark.config.models import BenchmarkRunConfig


def build_run_cli_argv_from_args(
    args: argparse.Namespace,
    *,
    data_dir: str,
    output: str,
    repo_root: Path,
) -> list[str]:
    """Build a legacy ``benchmark.cli run`` argv from parsed flags."""
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


def repo_relative_spec_path(spec: str, repo_root: Path) -> str:
    spec_path = Path(spec).resolve()
    try:
        return str(spec_path.relative_to(repo_root.resolve()))
    except ValueError as e:
        msg = "--spec must be inside the repository when using --cloud gcp"
        raise ValueError(msg) from e


def build_run_cli_argv_from_config(
    config: BenchmarkRunConfig,
    *,
    data_dir: str,
    output: str,
    repo_root: Path,
    verbose: bool = False,
) -> list[str]:
    """Build a legacy ``benchmark.cli run`` argv from a typed config for compatibility."""
    media = config.resolved_media()
    argv: list[str] = [
        "--data-dir",
        data_dir,
        "--output",
        output,
        "--media",
        media,
        "--num-runs",
        str(config.execution.num_runs),
        "--num-channels",
        str(config.data.num_channels),
    ]
    if config.data.num_items is not None:
        argv += ["--num-items", str(config.data.num_items)]
    if config.selection.libraries:
        argv += ["--libraries", *config.selection.libraries]
    if config.selection.transforms:
        argv += ["--transforms", *config.selection.transforms]
    if config.selection.transform_set:
        argv += ["--transform-set", config.selection.transform_set]
    if config.selection.spec:
        argv += ["--spec", repo_relative_spec_path(config.selection.spec, repo_root)]
    if config.selection.multichannel:
        argv.append("--multichannel")
    if verbose:
        argv.append("--verbose")
    if config.selection.scenario:
        argv += ["--scenario", config.selection.scenario]
    if config.selection.mode:
        argv += ["--mode", config.selection.mode]
    if config.execution.pipeline_scope:
        argv += ["--pipeline-scope", config.execution.pipeline_scope]
    if config.execution.device:
        argv += ["--device", config.execution.device]
    if config.execution.thread_policy:
        argv += ["--thread-policy", config.execution.thread_policy]
    argv += ["--batch-size", str(config.execution.batch_size)]
    argv += ["--workers", str(config.execution.workers)]
    if config.execution.min_time:
        argv += ["--min-time", str(config.execution.min_time)]
    if config.execution.min_batches:
        argv += ["--min-batches", str(config.execution.min_batches)]
    if config.data.clip_length:
        argv += ["--clip-length", str(config.data.clip_length)]
    if config.selection.decoders:
        argv += ["--decoders", *config.selection.decoders]
    if not config.execution.refresh_requirements:
        argv.append("--no-refresh-requirements")
    if config.execution.slow_threshold_sec_per_item is not None:
        argv += ["--slow-threshold-sec-per-item", str(config.execution.slow_threshold_sec_per_item)]
    if config.execution.slow_preflight_items is not None:
        argv += ["--slow-preflight-items", str(config.execution.slow_preflight_items)]
    if config.execution.disable_slow_skip:
        argv.append("--disable-slow-skip")
    return argv
