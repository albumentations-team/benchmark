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
    if _provided(args, "--no-refresh-requirements"):
        data["execution"]["refresh_requirements"] = args.refresh_requirements
    if _provided(args, "--gcp-dry-run"):
        data["cloud"] = data.get("cloud") or {}
        data["cloud"]["dry_run"] = args.gcp_dry_run
    if _provided(args, "--gcp-attached"):
        data["cloud"] = data.get("cloud") or {}
        data["cloud"]["attached"] = args.gcp_attached
    if _provided(args, "--gcp-keep-instance"):
        data["cloud"] = data.get("cloud") or {}
        data["cloud"]["keep_instance"] = args.gcp_keep_instance
    if _provided(args, "--gcp-remote-data-dir"):
        data["data"]["remote_data_dir"] = args.gcp_remote_data_dir
    return BenchmarkRunConfig.model_validate(data)
