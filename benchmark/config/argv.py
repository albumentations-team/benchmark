from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.config.models import BenchmarkRunConfig


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
