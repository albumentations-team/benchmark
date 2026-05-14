from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.config import (
    RUN_CONFIG_ENV_VAR,
    BenchmarkRunConfig,
    DataConfig,
    ExecutionConfig,
    OutputConfig,
    SelectionConfig,
    apply_cli_overrides,
    install_run_config_env,
    load_run_config,
    remote_run_config_payload,
    resolve_config_transform_set,
    run_config_payload_from_env,
)


def test_loads_example_config() -> None:
    config = load_run_config(Path("configs/examples/local_rgb_micro_cpu.yaml"))

    assert config.selection.scenario == "image-rgb"
    assert config.execution.num_runs == 1
    assert config.output.output_dir == "output/rgb_micro"


def test_all_checked_in_configs_validate() -> None:
    for path in Path("configs").rglob("*.yaml"):
        load_run_config(path)


def test_paper_production_configs_use_deadline_sizing() -> None:
    expected = {
        "prod_c4_rgb_micro_cpu.yaml": {
            "scenario": "image-rgb",
            "mode": "micro",
            "libraries": ["albumentationsx", "torchvision", "kornia", "pillow"],
            "num_items": 2000,
            "device": "none",
            "machine_type": "c4-standard-16",
            "disk_size_gb": 100,
        },
        "prod_c4_rgb_dataloader_cpu.yaml": {
            "scenario": "image-rgb",
            "mode": "pipeline",
            "libraries": ["albumentationsx", "torchvision", "kornia", "pillow"],
            "num_items": 10000,
            "device": "none",
            "machine_type": "c4-standard-16",
            "pipeline_scope": "memory_dataloader_augment",
            "disk_size_gb": 100,
        },
        "prod_c4_9ch_micro_cpu.yaml": {
            "scenario": "image-9ch",
            "mode": "micro",
            "libraries": ["albumentationsx", "torchvision", "kornia"],
            "num_items": 2000,
            "device": "none",
            "machine_type": "c4-standard-16",
            "num_channels": 9,
            "disk_size_gb": 100,
        },
        "prod_c4_9ch_dataloader_cpu.yaml": {
            "scenario": "image-9ch",
            "mode": "pipeline",
            "libraries": ["albumentationsx", "torchvision"],
            "num_items": 10000,
            "device": "none",
            "machine_type": "c4-standard-16",
            "num_channels": 9,
            "pipeline_scope": "memory_dataloader_augment",
            "disk_size_gb": 100,
            "batch_size": 128,
        },
        "prod_c4_highmem_9ch_dataloader_cpu_kornia.yaml": {
            "scenario": "image-9ch",
            "mode": "pipeline",
            "libraries": ["kornia"],
            "num_items": 10000,
            "device": "none",
            "machine_type": "c4-highmem-16",
            "num_channels": 9,
            "pipeline_scope": "memory_dataloader_augment",
            "disk_size_gb": 100,
            "batch_size": 128,
        },
        "prod_g2_rgb_micro_gpu.yaml": {
            "scenario": "image-rgb",
            "mode": "micro",
            "libraries": ["torchvision", "kornia"],
            "num_items": 2000,
            "device": "cuda",
            "machine_type": "g2-standard-16",
            "disk_size_gb": 200,
        },
        "prod_g2_rgb_dataloader_gpu.yaml": {
            "scenario": "image-rgb",
            "mode": "pipeline",
            "libraries": ["torchvision", "kornia"],
            "num_items": 10000,
            "device": "cuda",
            "machine_type": "g2-standard-16",
            "pipeline_scope": "memory_dataloader_augment",
            "disk_size_gb": 200,
            "slow_preflight_items": 256,
        },
        "prod_g2_9ch_micro_gpu.yaml": {
            "scenario": "image-9ch",
            "mode": "micro",
            "libraries": ["torchvision", "kornia"],
            "num_items": 1000,
            "device": "cuda",
            "machine_type": "g2-standard-16",
            "num_channels": 9,
            "disk_size_gb": 200,
        },
        "prod_g2_9ch_dataloader_gpu.yaml": {
            "scenario": "image-9ch",
            "mode": "pipeline",
            "libraries": ["torchvision", "kornia"],
            "num_items": 10000,
            "device": "cuda",
            "machine_type": "g2-standard-16",
            "num_channels": 9,
            "pipeline_scope": "decode_dataloader_augment",
            "disk_size_gb": 200,
            "batch_size": 128,
        },
        "prod_c4_video_micro_cpu.yaml": {
            "scenario": "video-16f",
            "mode": "micro",
            "libraries": ["albumentationsx", "torchvision", "kornia"],
            "num_items": 500,
            "device": "none",
            "machine_type": "c4-standard-16",
            "gcs_uri": "gs://imagenet_validation/ucf101/ucf101.tar",
            "clip_length": 16,
            "disk_size_gb": 100,
        },
        "prod_c4_video_dataloader_cpu.yaml": {
            "scenario": "video-16f",
            "mode": "pipeline",
            "libraries": ["albumentationsx", "torchvision", "kornia"],
            "num_items": 625,
            "device": "none",
            "machine_type": "c4-standard-16",
            "gcs_uri": "gs://imagenet_validation/ucf101/ucf101.tar",
            "clip_length": 16,
            "pipeline_scope": "memory_dataloader_augment",
            "disk_size_gb": 100,
            "batch_size": 64,
        },
        "prod_g2_video_micro_gpu.yaml": {
            "scenario": "video-16f",
            "mode": "micro",
            "libraries": ["torchvision", "kornia"],
            "num_items": 500,
            "device": "cuda",
            "machine_type": "g2-standard-16",
            "gcs_uri": "gs://imagenet_validation/ucf101/ucf101.tar",
            "clip_length": 16,
            "disk_size_gb": 200,
        },
        "prod_g2_video_dataloader_gpu.yaml": {
            "scenario": "video-16f",
            "mode": "pipeline",
            "libraries": ["torchvision", "kornia"],
            "num_items": 625,
            "device": "cuda",
            "machine_type": "g2-standard-16",
            "gcs_uri": "gs://imagenet_validation/ucf101/ucf101.tar",
            "clip_length": 16,
            "pipeline_scope": "memory_dataloader_augment",
            "disk_size_gb": 200,
            "batch_size": 16,
        },
    }

    for filename, spec in expected.items():
        config = load_run_config(Path("configs/paper") / filename)

        assert config.selection.scenario == spec["scenario"]
        assert config.selection.mode == spec["mode"]
        assert config.selection.libraries == spec["libraries"]
        assert config.selection.transform_set == "paper"
        assert config.data.gcs_uri == spec.get("gcs_uri", "gs://imagenet_validation/imagenet/val.tar")
        assert config.data.num_items == spec["num_items"]
        assert config.data.num_channels == spec.get("num_channels", 3)
        assert config.data.clip_length == spec.get("clip_length")
        assert config.execution.num_runs == 1
        assert config.execution.device == spec["device"]
        assert config.cloud is not None
        assert config.cloud.zone == "us-central1-a"
        assert config.cloud.machine_type == spec["machine_type"]
        assert config.cloud.disk_size_gb == spec["disk_size_gb"]

        if spec["mode"] == "pipeline":
            assert config.execution.pipeline_scope == spec["pipeline_scope"]
            assert config.execution.batch_size == spec.get("batch_size", 256)
            assert config.execution.workers == 8
            assert config.execution.min_time == 0
            assert config.execution.thread_policy == "pipeline-default"
            assert config.execution.slow_preflight_items == spec.get("slow_preflight_items")


def test_rejects_image_cuda_for_albumentations() -> None:
    with pytest.raises(ValidationError, match="albumentationsx image benchmarks do not support --device cuda"):
        BenchmarkRunConfig(
            selection=SelectionConfig(scenario="image-rgb", mode="micro", libraries=["albumentationsx"]),
            data=DataConfig(data_dir="/data"),
            output=OutputConfig(output_dir="/out"),
            execution=ExecutionConfig(device="cuda"),
        )


def test_rejects_micro_workers() -> None:
    with pytest.raises(ValidationError, match="micro benchmarks do not use DataLoader workers"):
        BenchmarkRunConfig(
            selection=SelectionConfig(scenario="image-rgb", mode="micro", libraries=["torchvision"]),
            data=DataConfig(data_dir="/data"),
            output=OutputConfig(output_dir="/out"),
            execution=ExecutionConfig(workers=1),
        )


def test_rejects_transform_set_without_scenario() -> None:
    with pytest.raises(ValidationError, match=r"selection\.transform_set requires selection\.scenario"):
        BenchmarkRunConfig(
            selection=SelectionConfig(media="image", libraries=["torchvision"], transform_set="paper"),
            data=DataConfig(data_dir="/data"),
            output=OutputConfig(output_dir="/out"),
        )


def test_resolve_transform_set_records_concrete_names() -> None:
    config = load_run_config(Path("configs/examples/local_rgb_micro_cpu.yaml"))

    resolved = resolve_config_transform_set(config, Path.cwd())

    assert resolved.selection.transform_set == "paper"
    assert resolved.selection.transforms
    assert "HorizontalFlip" in resolved.selection.transforms


def test_resolve_video_pipeline_transform_set_records_recipe_names() -> None:
    config = load_run_config(Path("configs/paper/prod_c4_video_dataloader_cpu.yaml"))

    resolved = resolve_config_transform_set(config, Path.cwd())

    assert resolved.selection.transforms
    assert "RandomCrop224+HorizontalFlip+Normalize+ToTensor" in resolved.selection.transforms
    assert "HorizontalFlip" not in resolved.selection.transforms


def test_rejects_detached_cloud_without_gcs_data() -> None:
    with pytest.raises(ValidationError, match=r"data\.gcs_uri is required"):
        BenchmarkRunConfig.model_validate(
            {
                "selection": {"scenario": "image-rgb", "mode": "micro", "libraries": ["torchvision"]},
                "data": {"data_dir": "unused"},
                "output": {
                    "output_dir": "gcp_runs/test",
                    "gcs_results_uri": "gs://bucket/results",
                },
                "cloud": {"provider": "gcp", "project": "p"},
            },
        )


def test_cli_overrides_apply_after_yaml_config() -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_micro_gpu_smoke.yaml"))
    args = argparse.Namespace(
        data_dir="/data/override",
        output="/override",
        libraries=["kornia"],
        transforms=None,
        transform_set="paper",
        spec=None,
        num_items=5,
        num_runs=2,
        device="none",
        workers=0,
        batch_size=32,
        refresh_requirements=False,
        cloud="gcp",
        gcp_project="override-project",
        gcp_zone="us-central1-a",
        gcp_machine_type="g2-standard-16",
        gcp_gpu_type=None,
        gcp_disk_size_gb=300,
        gcp_gcs_data_uri="gs://bucket/data.tar",
        gcp_gcs_results_uri="gs://bucket/results",
        gcp_dry_run=False,
        gcp_attached=False,
        gcp_keep_instance=True,
        gcp_keep_on_failure=True,
        gcp_preemptible=True,
        gcp_timeout_hours=7.5,
        gcp_remote_repo_dir="~/bench-override",
        gcp_venv_cache_uri="gs://bucket/cache",
        gcp_no_venv_cache=False,
        gcp_force_venv_cache_rebuild=True,
        gcp_remote_data_dir=None,
        slow_threshold_sec_per_item=0.2,
        slow_preflight_items=256,
        disable_slow_skip=True,
        dry_run=False,
        _provided_flags={
            "--data-dir",
            "--output",
            "--libraries",
            "--num-items",
            "--num-runs",
            "--slow-threshold-sec-per-item",
            "--slow-preflight-items",
            "--disable-slow-skip",
            "--no-refresh-requirements",
            "--gcp-project",
            "--gcp-zone",
            "--gcp-disk-size-gb",
            "--gcp-gcs-data-uri",
            "--gcp-gcs-results-uri",
            "--gcp-keep-instance",
            "--gcp-keep-on-failure",
            "--gcp-preemptible",
            "--gcp-timeout-hours",
            "--gcp-remote-repo-dir",
            "--gcp-venv-cache-uri",
            "--gcp-force-venv-cache-rebuild",
        },
    )

    resolved = apply_cli_overrides(config, args)

    assert resolved.data.data_dir == "/data/override"
    assert resolved.output.output_dir == "/override"
    assert resolved.selection.libraries == ["kornia"]
    assert resolved.data.num_items == 5
    assert resolved.execution.num_runs == 2
    assert resolved.execution.slow_threshold_sec_per_item == pytest.approx(0.2)
    assert resolved.execution.slow_preflight_items == 256
    assert resolved.execution.disable_slow_skip is True
    assert resolved.execution.refresh_requirements is False
    assert resolved.data.gcs_uri == "gs://bucket/data.tar"
    assert resolved.output.gcs_results_uri == "gs://bucket/results"
    assert resolved.cloud is not None
    assert resolved.cloud.project == "override-project"
    assert resolved.cloud.zone == "us-central1-a"
    assert resolved.cloud.disk_size_gb == 300
    assert resolved.cloud.keep_instance is True
    assert resolved.cloud.keep_on_failure is True
    assert resolved.cloud.preemptible is True
    assert resolved.cloud.timeout_hours == pytest.approx(7.5)
    assert resolved.cloud.remote_repo_dir == "~/bench-override"
    assert resolved.cloud.venv_cache_uri == "gs://bucket/cache"
    assert resolved.cloud.force_venv_cache_rebuild is True


def test_config_exposes_resolved_scenario_fields() -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_dataloader_gpu_smoke.yaml"))

    assert config.resolved_mode() == "pipeline"
    assert config.resolved_media() == "image"
    assert config.resolved_libraries() == ["torchvision", "kornia"]


def test_remote_run_config_payload_uses_vm_paths_and_strips_cloud() -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_dataloader_gpu_smoke.yaml"))

    payload = remote_run_config_payload(
        config,
        data_dir="/root/benchmark-data/val",
        output_dir="/root/benchmark-work/results",
    )

    assert payload["data"]["data_dir"] == "/root/benchmark-data/val"
    assert payload["output"]["output_dir"] == "/root/benchmark-work/results"
    assert "cloud" not in payload


def test_run_config_env_roundtrip() -> None:
    config = load_run_config(Path("configs/examples/local_rgb_micro_cpu.yaml"))

    try:
        install_run_config_env(config)

        assert RUN_CONFIG_ENV_VAR in os.environ
        payload = run_config_payload_from_env()
        assert payload is not None
        assert payload["selection"]["scenario"] == "image-rgb"
    finally:
        os.environ.pop(RUN_CONFIG_ENV_VAR, None)
