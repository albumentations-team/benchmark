from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.adapters.dali_image import run_dali_image_transform
from benchmark.config import load_run_config
from benchmark.config.models import BenchmarkRunConfig
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
        device="cuda",
        slow_threshold_sec_per_item=0.2,
        slow_preflight_items=5,
        disable_slow_skip=True,
    )

    cmd = job.micro_command(tmp_path / ".venv" / "bin" / "python")

    assert cmd[cmd.index("--scenario") + 1] == "image-9ch"
    assert cmd[cmd.index("--num-channels") + 1] == "9"
    assert cmd[cmd.index("--clip-length") + 1] == "16"
    assert cmd[cmd.index("--device") + 1] == "cuda"
    assert cmd[cmd.index("--transforms") + 1] == "HorizontalFlip,GaussianBlur"
    assert cmd[cmd.index("--slow-threshold-sec-per-item") + 1] == "0.2"
    assert cmd[cmd.index("--slow-preflight-items") + 1] == "5"
    assert "--disable-slow-skip" in cmd
    assert job.env_extra(verbose=True)["BENCHMARK_TRANSFORMS_FILTER"] == "HorizontalFlip,GaussianBlur"
    assert job.env_extra(verbose=True)["BENCHMARK_VERBOSE"] == "1"


def test_video_micro_job_passes_clip_length_to_pyperf_runner(tmp_path: Path) -> None:
    job = BenchmarkJob(
        library="kornia",
        scenario="video-16f",
        mode="micro",
        media="video",
        data_dir=tmp_path / "videos",
        output_file=tmp_path / "out.json",
        num_items=10,
        num_runs=1,
        num_channels=3,
        clip_length=8,
        spec_file=tmp_path / "spec.py",
    )

    cmd = job.micro_command(tmp_path / ".venv" / "bin" / "python")

    assert cmd[cmd.index("--clip-length") + 1] == "8"


def test_job_can_be_built_from_run_config(tmp_path: Path) -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_dataloader_gpu_smoke.yaml"))

    job = BenchmarkJob.from_run_config(
        library="kornia",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=3,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.mode == "pipeline"
    assert job.pipeline_scope == "decode_dataloader_augment"
    assert job.device == "cuda"


def test_kornia_gpu_rgb_image_micro_excludes_shear_but_keeps_median_blur(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-rgb",
                "mode": "micro",
                "libraries": ["kornia"],
                "transforms": ["Resize", "MedianBlur", "Shear", "HorizontalFlip"],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "cuda"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="kornia",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=3,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == ("Resize", "MedianBlur", "HorizontalFlip")
    cmd = job.micro_command(tmp_path / ".venv" / "bin" / "python")
    assert cmd[cmd.index("--transforms") + 1] == "Resize,MedianBlur,HorizontalFlip"


def test_kornia_cpu_image_keeps_shear(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-rgb",
                "mode": "micro",
                "libraries": ["kornia"],
                "transforms": ["Resize", "Shear"],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "none"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="kornia",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=3,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == ("Resize", "Shear")


def test_kornia_gpu_rgb_image_pipeline_excludes_shear_but_keeps_median_blur_recipe(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-rgb",
                "mode": "pipeline",
                "libraries": ["kornia"],
                "transforms": [
                    "RandomCrop224+Resize+Normalize+ToTensor",
                    "RandomCrop224+MedianBlur+Normalize+ToTensor",
                    "RandomCrop224+Shear+Normalize+ToTensor",
                ],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "cuda"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="kornia",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=3,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == (
        "RandomCrop224+Resize+Normalize+ToTensor",
        "RandomCrop224+MedianBlur+Normalize+ToTensor",
    )
    assert (
        job.env_extra()["BENCHMARK_TRANSFORMS_FILTER"]
        == "RandomCrop224+Resize+Normalize+ToTensor,RandomCrop224+MedianBlur+Normalize+ToTensor"
    )


def test_torchvision_gpu_image_micro_excludes_jpeg_compression(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-rgb",
                "mode": "micro",
                "libraries": ["torchvision"],
                "transforms": ["Resize", "JpegCompression", "HorizontalFlip"],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "cuda"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="torchvision",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=3,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == ("Resize", "HorizontalFlip")


def test_torchvision_gpu_image_pipeline_excludes_jpeg_compression_recipe(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-rgb",
                "mode": "pipeline",
                "libraries": ["torchvision"],
                "transforms": [
                    "RandomCrop224+Resize+Normalize+ToTensor",
                    "RandomCrop224+JpegCompression+Normalize+ToTensor",
                ],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "cuda"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="torchvision",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=3,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == ("RandomCrop224+Resize+Normalize+ToTensor",)


def test_torchvision_cpu_image_keeps_jpeg_compression(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-rgb",
                "mode": "micro",
                "libraries": ["torchvision"],
                "transforms": ["Resize", "JpegCompression"],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "none"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="torchvision",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=3,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == ("Resize", "JpegCompression")


def test_kornia_gpu_9ch_image_excludes_known_gpu_limitations(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-9ch",
                "mode": "micro",
                "libraries": ["kornia"],
                "transforms": ["Resize", "MedianBlur", "Shear", "HorizontalFlip"],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "cuda"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="kornia",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=9,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == ("Resize", "HorizontalFlip")


def test_kornia_gpu_9ch_image_pipeline_excludes_median_blur_and_shear_recipes(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-9ch",
                "mode": "pipeline",
                "libraries": ["kornia"],
                "transforms": [
                    "RandomCrop224+Resize+Normalize+ToTensor",
                    "RandomCrop224+MedianBlur+Normalize+ToTensor",
                    "RandomCrop224+Shear+Normalize+ToTensor",
                ],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "cuda"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="kornia",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=9,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == ("RandomCrop224+Resize+Normalize+ToTensor",)


def test_kornia_cpu_9ch_image_keeps_median_blur(tmp_path: Path) -> None:
    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "scenario": "image-9ch",
                "mode": "micro",
                "libraries": ["kornia"],
                "transforms": ["Resize", "MedianBlur"],
            },
            "data": {"data_dir": "/data"},
            "output": {"output_dir": "/out"},
            "execution": {"device": "none"},
        },
    )

    job = BenchmarkJob.from_run_config(
        library="kornia",
        config=config,
        data_dir=tmp_path / "data",
        output_file=tmp_path / "out.json",
        num_channels=9,
        clip_length=16,
        spec_file=tmp_path / "spec.py",
    )

    assert job.transforms_filter == ("Resize", "MedianBlur")


def test_job_from_run_config_rejects_decode_mode(tmp_path: Path) -> None:
    data = load_run_config(Path("configs/paper/gcp_g2_video_smoke.yaml")).model_dump()
    data["selection"] = {"scenario": "video-decode-16f", "mode": "decode"}
    from benchmark.config import BenchmarkRunConfig

    config = BenchmarkRunConfig.model_validate(data)

    with pytest.raises(ValueError, match="does not support mode 'decode'"):
        BenchmarkJob.from_run_config(
            library="opencv",
            config=config,
            data_dir=tmp_path / "data",
            output_file=tmp_path / "out.json",
            num_channels=3,
            clip_length=16,
            spec_file=None,
        )


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


def test_execute_image_dali_job_uses_image_venv(tmp_path: Path) -> None:
    job = BenchmarkJob(
        library="dali",
        scenario="image-rgb",
        mode="pipeline",
        media="image",
        data_dir=tmp_path / "images",
        output_file=tmp_path / "dali.json",
        num_items=2,
        num_runs=1,
        num_channels=3,
        pipeline_scope="decode_dataloader_augment",
        device="cuda",
        backend="dali_pipeline",
    )

    fake_python = tmp_path / "fake-venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)

    with (
        patch("benchmark.envs.ensure_venv", return_value=fake_python) as ensure,
        patch("benchmark.orchestrator.subprocess.run") as run,
    ):
        execute_job(job, repo_root=tmp_path)

    ensure.assert_called_once_with("dali", "image", tmp_path, refresh_requirements=True)
    run.assert_called_once()


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


def test_dali_image_job_json_roundtrip_preserves_paths(tmp_path: Path) -> None:
    job = BenchmarkJob(
        library="dali",
        scenario="image-rgb",
        mode="pipeline",
        media="image",
        data_dir=tmp_path / "images",
        output_file=tmp_path / "out.json",
        num_items=10,
        num_runs=1,
        num_channels=3,
        spec_file=None,
        transforms_filter=("RandomCrop224+Resize+Normalize+ToTensor",),
        pipeline_scope="decode_dataloader_augment",
        device="cuda",
        backend="dali_pipeline",
    )
    raw = json.loads(json.dumps(asdict(job), default=str))
    assert benchmark_job_from_json_dict(raw) == job


def test_dali_image_unsupported_transform_returns_unsupported_result(tmp_path: Path) -> None:
    result = run_dali_image_transform(
        transform_name="RandomCrop224+Perspective+Normalize+ToTensor",
        spec={"name": "Perspective", "params": {}},
        paths=[tmp_path / "image.jpg"],
        batch_size=1,
        num_runs=1,
        workers=1,
    )

    assert result["supported"] is False
    assert "Perspective" in result["reason"]


def test_g2_instance_create_uses_gpu_maintenance_policy_without_accelerator_flag() -> None:
    from benchmark.cloud.gcp import GCPRunner
    from benchmark.cloud.instance import GCPInstanceConfig

    runner = GCPRunner(GCPInstanceConfig(project="p", zone="z", machine_type="g2-standard-16"))

    with patch("benchmark.cloud.gcp._run") as run:
        runner.create_instance()

    cmd = run.call_args.args[0]
    assert "--maintenance-policy" in cmd
    assert cmd[cmd.index("--maintenance-policy") + 1] == "TERMINATE"
    assert "--accelerator" not in cmd


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


def test_detached_gcp_bootstrap_resolves_gcloud_executable(tmp_path: Path) -> None:
    from benchmark.cloud.gcp import _BOOTSTRAP_SH, _STARTUP_INLINE

    bash = shutil.which("bash")
    assert bash is not None

    for name, script in {"bootstrap": _BOOTSTRAP_SH, "startup": _STARTUP_INLINE}.items():
        path = tmp_path / f"{name}.sh"
        path.write_text(script, encoding="utf-8")

        result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, check=False)  # noqa: S603

        assert result.returncode == 0, result.stderr
        assert "resolve_gcloud()" in script
        assert '"$GCLOUD_BIN" --quiet storage' in script
        assert "gcloud --quiet storage" not in script


def test_detached_gcp_bootstrap_syncs_results_before_done_marker() -> None:
    from benchmark.cloud.gcp import _BOOTSTRAP_SH

    sync_index = _BOOTSTRAP_SH.index('gcs_rsync_retry "results"')
    marker_index = _BOOTSTRAP_SH.index('gcs_cp_retry "$marker_name"')
    stale_warning = "result rsync failed after terminal marker confirmation"

    assert sync_index < marker_index
    assert stale_warning not in _BOOTSTRAP_SH
