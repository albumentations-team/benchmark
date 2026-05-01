from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from benchmark.jobs import BenchmarkJob
from benchmark.orchestrator import execute_job

if TYPE_CHECKING:
    from pathlib import Path


def test_micro_job_builds_pyperf_command_with_filters_and_slow_skip(tmp_path: Path) -> None:
    job = BenchmarkJob(
        library="kornia",
        scenario="image-9ch",
        mode="micro",
        media="image",
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_items=10,
        num_runs=3,
        num_channels=9,
        spec_file=tmp_path / "spec.py",
        transforms_filter=("HorizontalFlip", "GaussianBlur"),
        slow_threshold_sec_per_item=0.2,
        slow_preflight_items=5,
        disable_slow_skip=True,
    )

    cmd = job.micro_command(tmp_path / ".venv" / "bin" / "python")

    assert cmd[cmd.index("--scenario") + 1] == "image-9ch"
    assert cmd[cmd.index("--num-channels") + 1] == "9"
    assert cmd[cmd.index("--transforms") + 1] == "HorizontalFlip,GaussianBlur"
    assert cmd[cmd.index("--slow-threshold-sec-per-item") + 1] == "0.2"
    assert cmd[cmd.index("--slow-preflight-items") + 1] == "5"
    assert "--disable-slow-skip" in cmd
    assert job.env_extra(verbose=True)["BENCHMARK_TRANSFORMS_FILTER"] == "HorizontalFlip,GaussianBlur"
    assert job.env_extra(verbose=True)["BENCHMARK_VERBOSE"] == "1"


def test_execute_job_deletes_pyperf_sidecar_before_micro_run(tmp_path: Path) -> None:
    sidecar = tmp_path / "out.pyperf.json"
    sidecar.write_text("stale", encoding="utf-8")
    job = BenchmarkJob(
        library="kornia",
        scenario="image-rgb",
        mode="micro",
        media="image",
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_items=1,
        num_runs=1,
        num_channels=3,
        spec_file=tmp_path / "spec.py",
    )

    with (
        patch("benchmark.envs.ensure_venv", return_value=tmp_path / "python"),
        patch("benchmark.orchestrator.subprocess.run") as run,
    ):
        execute_job(job, repo_root=tmp_path)

    assert not sidecar.exists()
    run.assert_called_once()


def test_execute_job_uses_dali_backend_without_subprocess(tmp_path: Path) -> None:
    runner = MagicMock()
    job = BenchmarkJob(
        library="dali",
        scenario="video-16f",
        mode="pipeline",
        media="video",
        data_dir=tmp_path / "videos",
        output_file=tmp_path / "dali.json",
        num_items=2,
        num_runs=1,
        num_channels=3,
        clip_length=16,
        pipeline_scope="decode_dataloader_augment",
        device="cuda",
        backend="dali_pipeline",
        slow_threshold_sec_per_item=3.0,
        slow_preflight_items=1,
        disable_slow_skip=True,
    )

    with (
        patch("benchmark.pipeline_runner.PipelineBenchmarkRunner", return_value=runner) as runner_cls,
        patch("benchmark.orchestrator.subprocess.run") as run,
    ):
        execute_job(job, repo_root=tmp_path)

    run.assert_not_called()
    runner.run.assert_called_once()
    kwargs = runner_cls.call_args.kwargs
    assert kwargs["library"] == "dali"
    assert kwargs["slow_threshold_sec_per_item"] == pytest.approx(3.0)
    assert kwargs["disable_slow_skip"] is True


def test_attached_gcp_run_deletes_instance_when_setup_fails(tmp_path: Path) -> None:
    from benchmark.cloud.gcp import GCPRunner
    from benchmark.cloud.instance import GCPInstanceConfig

    class FailingRunner(GCPRunner):
        def __init__(self) -> None:
            super().__init__(GCPInstanceConfig(project="p", zone="z"))
            self.events: list[str] = []

        def create_instance(self, **_kwargs: object) -> None:
            self.events.append("create")

        def wait_for_ssh(self, timeout: int = 300, poll_interval: int = 10) -> None:
            _ = (timeout, poll_interval)
            self.events.append("wait")
            raise RuntimeError("ssh failed")

        def delete_instance(self) -> None:
            self.events.append("delete")

    runner = FailingRunner()

    with pytest.raises(RuntimeError, match="ssh failed"):
        runner.run_attached(repo_root=tmp_path, remote_cli_args=[], local_output_dir=tmp_path)

    assert runner.events == ["create", "wait", "delete"]
