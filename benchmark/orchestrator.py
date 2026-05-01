from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark import envs

if TYPE_CHECKING:
    from benchmark.jobs import BenchmarkJob

logger = logging.getLogger(__name__)


def execute_job(job: BenchmarkJob, *, repo_root: Path, verbose: bool = False) -> None:
    if job.backend == "dali_pipeline":
        python = envs.ensure_venv(job.library, job.media, repo_root, refresh_requirements=job.refresh_requirements)
        payload = json.dumps(asdict(job), default=str)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(payload)
            job_path = Path(tmp.name)
        try:
            cmd = [
                str(python),
                "-m",
                "benchmark.dali_pipeline_worker",
                "--job-file",
                str(job_path),
            ]
            env = {**os.environ, **job.env_extra(verbose=verbose)}
            logger.info(
                "Running %s benchmark for %s %s -> %s (DALI subprocess)",
                job.mode,
                job.library,
                job.media,
                job.output_file,
            )
            subprocess.run(cmd, check=True, env=env)  # noqa: S603 - argv built from ensure_venv + fixed module name.
        finally:
            job_path.unlink(missing_ok=True)
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
