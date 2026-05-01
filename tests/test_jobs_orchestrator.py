from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.dali_pipeline_worker import benchmark_job_from_json_dict
from benchmark.jobs import BenchmarkJob
from benchmark.orchestrator import execute_job


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


def test_execute_job_runs_dali_via_subprocess_after_venv(tmp_path: Path) -> None:
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

    fake_python = tmp_path / "fake-venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)

    with (
        patch("benchmark.envs.ensure_venv", return_value=fake_python) as ensure,
        patch("benchmark.orchestrator.subprocess.run") as run,
    ):
        execute_job(job, repo_root=tmp_path)

    ensure.assert_called_once_with("dali", "video", tmp_path, refresh_requirements=True)
    run.assert_called_once()
    cmd = run.call_args[0][0]
    assert cmd[0] == str(fake_python)
    assert cmd[1:4] == ["-m", "benchmark.dali_pipeline_worker", "--job-file"]
    assert Path(cmd[4]).suffix == ".json"


def test_dali_job_json_roundtrip_preserves_paths(tmp_path: Path) -> None:
    job = BenchmarkJob(
        library="dali",
        scenario="video-16f",
        mode="pipeline",
        media="video",
        data_dir=tmp_path / "videos",
        output_file=tmp_path / "out.json",
        num_items=1,
        num_runs=2,
        num_channels=3,
        clip_length=8,
        spec_file=None,
        transforms_filter=("Resize",),
        backend="dali_pipeline",
    )
    raw = json.loads(json.dumps(asdict(job), default=str))
    assert benchmark_job_from_json_dict(raw) == job


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
