from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from benchmark.config.models import BenchmarkRunConfig

if TYPE_CHECKING:
    import argparse
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


def run_config_payload(config: BenchmarkRunConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)


def remote_run_config_payload(
    config: BenchmarkRunConfig,
    *,
    data_dir: str,
    output_dir: str,
) -> dict[str, Any]:
    data = run_config_payload(config)
    data["data"]["data_dir"] = data_dir
    data["output"]["output_dir"] = output_dir
    data["cloud"] = None
    return run_config_payload(BenchmarkRunConfig.model_validate(data))


def _provided(args: argparse.Namespace, flag: str) -> bool:
    return flag in getattr(args, "_provided_flags", set())


def _ensure_cloud(data: dict[str, Any]) -> dict[str, Any]:
    cloud = data.get("cloud")
    if not isinstance(cloud, dict):
        cloud = {}
        data["cloud"] = cloud
    return cloud


def apply_cli_overrides(config: BenchmarkRunConfig, args: argparse.Namespace) -> BenchmarkRunConfig:
    data = config.model_dump()

    if _provided(args, "--data-dir"):
        data["data"]["data_dir"] = args.data_dir
    if _provided(args, "--output"):
        data["output"]["output_dir"] = args.output
    if _provided(args, "--libraries"):
        data["selection"]["libraries"] = args.libraries
    if _provided(args, "--transforms"):
        data["selection"]["transforms"] = args.transforms
    if _provided(args, "--transform-set"):
        data["selection"]["transform_set"] = args.transform_set
    if _provided(args, "--spec"):
        data["selection"]["spec"] = args.spec
        data["selection"]["libraries"] = None
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
    if _provided(args, "--slow-threshold-sec-per-item"):
        data["execution"]["slow_threshold_sec_per_item"] = args.slow_threshold_sec_per_item
    if _provided(args, "--slow-preflight-items"):
        data["execution"]["slow_preflight_items"] = args.slow_preflight_items
    if _provided(args, "--disable-slow-skip"):
        data["execution"]["disable_slow_skip"] = args.disable_slow_skip
    if _provided(args, "--no-refresh-requirements"):
        data["execution"]["refresh_requirements"] = args.refresh_requirements
    if _provided(args, "--cloud"):
        _ensure_cloud(data)["provider"] = args.cloud
    if _provided(args, "--gcp-project"):
        _ensure_cloud(data)["project"] = args.gcp_project
    if _provided(args, "--gcp-zone"):
        _ensure_cloud(data)["zone"] = args.gcp_zone
    if _provided(args, "--gcp-machine-type"):
        _ensure_cloud(data)["machine_type"] = args.gcp_machine_type
    if _provided(args, "--gcp-gpu-type"):
        _ensure_cloud(data)["gpu_type"] = args.gcp_gpu_type
    if _provided(args, "--gcp-disk-size-gb"):
        _ensure_cloud(data)["disk_size_gb"] = args.gcp_disk_size_gb
    if _provided(args, "--gcp-gcs-data-uri"):
        data["data"]["gcs_uri"] = args.gcp_gcs_data_uri
    if _provided(args, "--gcp-gcs-results-uri"):
        data["output"]["gcs_results_uri"] = args.gcp_gcs_results_uri
    if _provided(args, "--gcp-dry-run"):
        _ensure_cloud(data)["dry_run"] = args.gcp_dry_run
    if _provided(args, "--gcp-attached"):
        _ensure_cloud(data)["attached"] = args.gcp_attached
    if _provided(args, "--gcp-keep-instance"):
        _ensure_cloud(data)["keep_instance"] = args.gcp_keep_instance
    if _provided(args, "--gcp-keep-on-failure"):
        _ensure_cloud(data)["keep_on_failure"] = args.gcp_keep_on_failure
    if _provided(args, "--gcp-preemptible"):
        _ensure_cloud(data)["preemptible"] = args.gcp_preemptible
    if _provided(args, "--gcp-remote-repo-dir"):
        _ensure_cloud(data)["remote_repo_dir"] = args.gcp_remote_repo_dir
    if _provided(args, "--gcp-venv-cache-uri"):
        _ensure_cloud(data)["venv_cache_uri"] = args.gcp_venv_cache_uri
    if _provided(args, "--gcp-no-venv-cache"):
        _ensure_cloud(data)["no_venv_cache"] = args.gcp_no_venv_cache
    if _provided(args, "--gcp-force-venv-cache-rebuild"):
        _ensure_cloud(data)["force_venv_cache_rebuild"] = args.gcp_force_venv_cache_rebuild
    if _provided(args, "--gcp-remote-data-dir"):
        data["data"]["remote_data_dir"] = args.gcp_remote_data_dir
    return BenchmarkRunConfig.model_validate(data)
