from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.config import (
    BenchmarkRunConfig,
    DataConfig,
    ExecutionConfig,
    OutputConfig,
    SelectionConfig,
    apply_cli_overrides,
    config_to_namespace,
    load_run_config,
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


def test_config_to_namespace_keeps_legacy_cli_shape() -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_gpu_smoke.yaml"))

    args = config_to_namespace(config)

    assert args.scenario == "image-rgb"
    assert args.mode == "pipeline"
    assert args.gcp_machine_type == "g2-standard-16"
    assert args.device == "cuda"
