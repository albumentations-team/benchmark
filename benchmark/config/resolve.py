from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from benchmark.config.models import (
    BenchmarkRunConfig,
    CloudConfig,
    DataConfig,
    ExecutionConfig,
    OutputConfig,
    SelectionConfig,
)

if TYPE_CHECKING:
    from pathlib import Path


def _read_config_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"Config file must contain a mapping at top level: {path}"
        raise TypeError(msg)
    return raw


def load_run_config(path: Path) -> BenchmarkRunConfig:
    return BenchmarkRunConfig.model_validate(_read_config_mapping(path))


def write_resolved_config(config: BenchmarkRunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(config.model_dump_json(exclude_none=True))
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def config_to_namespace(config: BenchmarkRunConfig, *, verbose: bool = False) -> argparse.Namespace:
    values = config.to_legacy_args()
    values.update(
        {
            "command": "run",
            "config": None,
            "resolved_config": None,
            "dry_run": False,
            "verbose": verbose,
        },
    )
    return argparse.Namespace(**values)


def run_config_from_args(args: argparse.Namespace) -> BenchmarkRunConfig:
    cloud = None
    if getattr(args, "cloud", None):
        cloud = CloudConfig(
            provider=args.cloud,
            project=args.gcp_project,
            zone=args.gcp_zone,
            machine_type=args.gcp_machine_type,
            gpu_type=args.gcp_gpu_type,
            attached=args.gcp_attached,
            dry_run=args.gcp_dry_run,
            disk_size_gb=args.gcp_disk_size_gb,
            keep_instance=args.gcp_keep_instance,
            keep_on_failure=args.gcp_keep_on_failure,
            preemptible=args.gcp_preemptible,
            remote_repo_dir=args.gcp_remote_repo_dir,
            venv_cache_uri=args.gcp_venv_cache_uri,
            no_venv_cache=args.gcp_no_venv_cache,
            force_venv_cache_rebuild=args.gcp_force_venv_cache_rebuild,
        )

    num_channels = 9 if getattr(args, "multichannel", False) and args.media == "image" else args.num_channels
    return BenchmarkRunConfig(
        selection=SelectionConfig(
            scenario=args.scenario,
            mode=args.mode,
            media=args.media,
            libraries=args.libraries,
            transforms=args.transforms,
            transform_set=args.transform_set,
            spec=args.spec,
            multichannel=args.multichannel,
            decoders=args.decoders,
        ),
        data=DataConfig(
            data_dir=args.data_dir,
            gcs_uri=args.gcp_gcs_data_uri,
            remote_data_dir=args.gcp_remote_data_dir,
            num_items=args.num_items,
            num_channels=num_channels,
            clip_length=args.clip_length,
        ),
        execution=ExecutionConfig(
            num_runs=args.num_runs,
            batch_size=args.batch_size,
            workers=args.workers,
            min_time=args.min_time,
            min_batches=args.min_batches,
            pipeline_scope=args.pipeline_scope,
            device=args.device,
            thread_policy=args.thread_policy,
            refresh_requirements=args.refresh_requirements,
            slow_threshold_sec_per_item=args.slow_threshold_sec_per_item,
            slow_preflight_items=args.slow_preflight_items,
            disable_slow_skip=args.disable_slow_skip,
        ),
        output=OutputConfig(
            output_dir=args.output,
            gcs_results_uri=args.gcp_gcs_results_uri,
        ),
        cloud=cloud,
    )


def _provided(args: argparse.Namespace, flag: str) -> bool:
    return flag in getattr(args, "_provided_flags", set())


def apply_cli_overrides(config: BenchmarkRunConfig, args: argparse.Namespace) -> BenchmarkRunConfig:
    data = config.model_dump()

    if _provided(args, "--output"):
        data["output"]["output_dir"] = args.output
    if _provided(args, "--num-items"):
        data["data"]["num_items"] = args.num_items
    if _provided(args, "--num-runs"):
        data["execution"]["num_runs"] = args.num_runs
    if _provided(args, "--device"):
        data["execution"]["device"] = args.device
    if _provided(args, "--workers"):
        data["execution"]["workers"] = args.workers
    if _provided(args, "--batch-size"):
        data["execution"]["batch_size"] = args.batch_size
    if _provided(args, "--gcp-dry-run"):
        data["cloud"] = data.get("cloud") or {}
        data["cloud"]["dry_run"] = args.gcp_dry_run
    return BenchmarkRunConfig.model_validate(data)
