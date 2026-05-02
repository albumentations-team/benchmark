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
    config = load_run_config(Path("configs/examples/local_rgb_micro_cpu.yaml"))
    args = argparse.Namespace(
        output="/override",
        num_items=5,
        num_runs=2,
        device="none",
        workers=0,
        batch_size=32,
        gcp_dry_run=False,
        dry_run=False,
        _provided_flags={"--output", "--num-items", "--num-runs"},
    )

    resolved = apply_cli_overrides(config, args)

    assert resolved.output.output_dir == "/override"
    assert resolved.data.num_items == 5
    assert resolved.execution.num_runs == 2


def test_config_exposes_resolved_scenario_fields() -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_gpu_smoke.yaml"))

    assert config.resolved_mode() == "pipeline"
    assert config.resolved_media() == "image"
    assert config.resolved_libraries() == ["torchvision", "kornia"]


def test_remote_run_config_payload_uses_vm_paths_and_strips_cloud() -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_gpu_smoke.yaml"))

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
