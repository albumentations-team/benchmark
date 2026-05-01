from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING

from benchmark import envs

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.jobs import BenchmarkJob

logger = logging.getLogger(__name__)


def execute_job(job: BenchmarkJob, *, repo_root: Path, verbose: bool = False) -> None:
    if job.backend == "dali_pipeline":
        _run_dali_pipeline_job(job)
        return

    python = envs.ensure_venv(job.library, job.media, repo_root, refresh_requirements=job.refresh_requirements)
    if job.mode == "micro":
        job.pyperf_output_file().unlink(missing_ok=True)
        cmd = job.micro_command(python)
    else:
        cmd = job.pipeline_command(python)
    env = {**os.environ, **job.env_extra(verbose=verbose)}
    logger.info("Running %s benchmark for %s %s -> %s", job.mode, job.library, job.media, job.output_file)
    subprocess.run(cmd, check=True, env=env)  # noqa: S603 - command is built from BenchmarkJob internals.


def _dali_transforms_from_specs(transforms_filter: tuple[str, ...] = ()) -> list[dict[str, object]]:
    from benchmark.transforms.specs import TRANSFORM_SPECS

    transforms: list[dict[str, object]] = [{"name": spec.name, "transform": spec.params} for spec in TRANSFORM_SPECS]
    if not transforms_filter:
        return transforms
    allowed = set(transforms_filter)
    return [transform for transform in transforms if str(transform["name"]) in allowed]


def _run_dali_pipeline_job(job: BenchmarkJob) -> None:
    from benchmark.pipeline_runner import PipelineBenchmarkRunner

    runner = PipelineBenchmarkRunner(
        library=job.library,
        data_dir=job.data_dir,
        output_file=job.output_file,
        transforms=_dali_transforms_from_specs(job.transforms_filter),
        call_fn=lambda _transform, item: item,
        media=job.media,
        scenario=job.scenario,
        num_items=job.num_items,
        num_runs=job.num_runs,
        batch_size=job.batch_size,
        workers=job.workers,
        min_time=job.min_time,
        min_batches=job.min_batches,
        num_channels=job.num_channels,
        clip_length=job.clip_length,
        pipeline_scope=job.pipeline_scope,  # type: ignore[arg-type]
        device=job.device,  # type: ignore[arg-type]
        thread_policy=job.thread_policy,  # type: ignore[arg-type]
        slow_threshold_sec_per_item=job.slow_threshold_sec_per_item,
        slow_preflight_items=job.slow_preflight_items,
        disable_slow_skip=job.disable_slow_skip,
    )
    runner.run()
