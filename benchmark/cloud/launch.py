from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.cloud.paths import VM_RESULTS, staged_data_dir_for_gcs_uri
from benchmark.config import (
    BenchmarkRunConfig,
    build_run_cli_argv_from_args,
    build_run_cli_argv_from_config,
    remote_run_config_payload,
    run_config_payload,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


def default_gcp_venv_cache_uri(results_uri: str) -> str:
    base = results_uri.rstrip("/")
    parent = base.rsplit("/", 1)[0] if "/" in base.removeprefix("gs://") else base
    return f"{parent}/augmentation-cache"


@dataclass(frozen=True)
class GcpLaunchOptions:
    project: str | None
    zone: str
    machine_type: str
    gpu_type: str | None
    attached: bool
    disk_size_gb: int
    keep_instance: bool
    keep_on_failure: bool
    preemptible: bool
    remote_repo_dir: str
    venv_cache_uri: str | None
    no_venv_cache: bool
    force_venv_cache_rebuild: bool
    dry_run: bool
    remote_data_dir: str | None
    gcs_data_uri: str | None
    gcs_results_uri: str | None


def gcp_launch_options(args: argparse.Namespace, run_config: BenchmarkRunConfig | None) -> GcpLaunchOptions:
    cloud = run_config.cloud if run_config and run_config.cloud else None
    return GcpLaunchOptions(
        project=cloud.project if cloud else args.gcp_project,
        zone=cloud.zone if cloud else args.gcp_zone,
        machine_type=cloud.machine_type if cloud else args.gcp_machine_type,
        gpu_type=cloud.gpu_type if cloud else args.gcp_gpu_type,
        attached=cloud.attached if cloud else args.gcp_attached,
        disk_size_gb=cloud.disk_size_gb if cloud else args.gcp_disk_size_gb,
        keep_instance=cloud.keep_instance if cloud else args.gcp_keep_instance,
        keep_on_failure=cloud.keep_on_failure if cloud else args.gcp_keep_on_failure,
        preemptible=cloud.preemptible if cloud else args.gcp_preemptible,
        remote_repo_dir=cloud.remote_repo_dir if cloud else args.gcp_remote_repo_dir,
        venv_cache_uri=cloud.venv_cache_uri if cloud else args.gcp_venv_cache_uri,
        no_venv_cache=cloud.no_venv_cache if cloud else args.gcp_no_venv_cache,
        force_venv_cache_rebuild=cloud.force_venv_cache_rebuild if cloud else args.gcp_force_venv_cache_rebuild,
        dry_run=cloud.dry_run if cloud else args.gcp_dry_run,
        remote_data_dir=run_config.data.remote_data_dir if run_config else args.gcp_remote_data_dir,
        gcs_data_uri=run_config.data.gcs_uri if run_config else args.gcp_gcs_data_uri,
        gcs_results_uri=run_config.output.gcs_results_uri if run_config else args.gcp_gcs_results_uri,
    )


def run_gcp(
    args: argparse.Namespace,
    repo_root: Path,
    local_output_dir: Path,
    run_config: BenchmarkRunConfig | None = None,
) -> None:
    """Run benchmarks on a GCP instance (detached by default)."""
    from benchmark.cloud.gcp import GCPRunner, build_gcp_job_dict, new_run_id
    from benchmark.cloud.instance import GCPInstanceConfig, is_gpu_machine_type

    options = gcp_launch_options(args, run_config)

    if not options.project:
        logger.error("--gcp-project is required when using --cloud gcp")
        sys.exit(1)

    gpu_machine = bool(options.gpu_type) or is_gpu_machine_type(options.machine_type)
    image_family = "pytorch-2-9-cu129-ubuntu-2404-nvidia-580" if gpu_machine else "ubuntu-2404-lts-amd64"
    image_project = "deeplearning-platform-release" if gpu_machine else "ubuntu-os-cloud"

    if options.attached:
        if not options.remote_data_dir:
            logger.error("--gcp-remote-data-dir is required with --gcp-attached (path to dataset on the VM)")
            sys.exit(1)
        remote_output = f"{options.remote_repo_dir}/results"
        try:
            bench_argv = (
                build_run_cli_argv_from_config(
                    run_config,
                    data_dir=options.remote_data_dir,
                    output=remote_output,
                    repo_root=repo_root,
                    verbose=getattr(args, "verbose", False),
                )
                if run_config
                else build_run_cli_argv_from_args(
                    args,
                    data_dir=options.remote_data_dir,
                    output=remote_output,
                    repo_root=repo_root,
                )
            )
        except ValueError as e:
            logger.error("%s", e)  # noqa: TRY400
            sys.exit(1)
        config = GCPInstanceConfig(
            project=options.project,
            zone=options.zone,
            machine_type=options.machine_type,
            accelerator_type=options.gpu_type,
            accelerator_count=1 if options.gpu_type else 0,
            image_family=image_family,
            image_project=image_project,
            disk_size_gb=options.disk_size_gb,
            preemptible=options.preemptible,
        )
        runner = GCPRunner(config)
        runner.run_attached(
            repo_root=repo_root,
            remote_cli_args=bench_argv,
            local_output_dir=local_output_dir,
            remote_repo_dir=options.remote_repo_dir,
            keep_instance=options.keep_instance,
        )
        return

    if not options.gcs_data_uri or not options.gcs_results_uri:
        logger.error("Detached GCP runs require --gcp-gcs-data-uri and --gcp-gcs-results-uri")
        sys.exit(1)

    run_id = new_run_id()
    machine_slug = options.machine_type.replace("/", "-").lower()[:24]
    instance_name = f"benchmark-{machine_slug}-{run_id[:12]}".lower().replace("_", "-")

    try:
        staged_data_dir = staged_data_dir_for_gcs_uri(options.gcs_data_uri)
        bench_argv = (
            build_run_cli_argv_from_config(
                run_config,
                data_dir=staged_data_dir,
                output=VM_RESULTS,
                repo_root=repo_root,
                verbose=getattr(args, "verbose", False),
            )
            if run_config
            else build_run_cli_argv_from_args(
                args,
                data_dir=staged_data_dir,
                output=VM_RESULTS,
                repo_root=repo_root,
            )
        )
    except ValueError as e:
        logger.error("%s", e)  # noqa: TRY400
        sys.exit(1)
    remote_config = (
        remote_run_config_payload(
            run_config,
            data_dir=staged_data_dir,
            output_dir=VM_RESULTS,
        )
        if run_config
        else None
    )

    submission = {
        "argv": sys.argv,
        "start_timestamp_unix": time.time(),
    }
    instance_meta = {
        "project": options.project,
        "zone": options.zone,
        "machine_type": options.machine_type,
        "accelerator_type": options.gpu_type,
        "instance_name": instance_name,
    }
    job = build_gcp_job_dict(
        run_id=run_id,
        gcs_data_uri=options.gcs_data_uri,
        benchmark_cli_args=bench_argv,
        run_config=remote_config,
        cloud_config=run_config_payload(run_config).get("cloud") if run_config and run_config.cloud else None,
        terminate_instance=not options.keep_instance,
        keep_instance_on_failure=options.keep_on_failure,
        venv_cache_uri=""
        if options.no_venv_cache
        else options.venv_cache_uri or default_gcp_venv_cache_uri(options.gcs_results_uri),
        force_venv_cache_rebuild=options.force_venv_cache_rebuild,
        submission=submission,
        instance_meta=instance_meta,
    )

    config = GCPInstanceConfig(
        project=options.project,
        zone=options.zone,
        machine_type=options.machine_type,
        accelerator_type=options.gpu_type,
        accelerator_count=1 if options.gpu_type else 0,
        image_family=image_family,
        image_project=image_project,
        disk_size_gb=options.disk_size_gb,
        preemptible=options.preemptible,
        instance_name_override=instance_name,
    )
    runner = GCPRunner(config)

    if options.dry_run:
        prefix = runner.run_detached(
            repo_root=repo_root,
            gcs_data_uri=options.gcs_data_uri,
            gcs_results_base_uri=options.gcs_results_uri,
            job=dict(job),
            dry_run=True,
        )
        logger.info("Dry run complete (no uploads or VM). Run prefix would be: %s", prefix)
        return

    run_prefix = runner.run_detached(
        repo_root=repo_root,
        gcs_data_uri=options.gcs_data_uri,
        gcs_results_base_uri=options.gcs_results_uri,
        job=dict(job),
        dry_run=False,
    )

    meta_path = local_output_dir / "gcp_last_run.json"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_prefix": run_prefix,
        "run_id": run_id,
        "instance_name": instance_name,
        "zone": options.zone,
        "project": options.project,
        "gcs_data_uri": options.gcs_data_uri,
        "gcs_results_base_uri": options.gcs_results_uri.rstrip("/"),
        "terminate_instance": not options.keep_instance,
        "keep_instance_on_failure": options.keep_on_failure,
        "venv_cache_uri": job["venv_cache_uri"],
        "force_venv_cache_rebuild": options.force_venv_cache_rebuild,
        "fetch_results_hint": f"gcloud storage cp -r {run_prefix}/results/* {local_output_dir}/",
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Detached GCP run submitted. Instance %s starting; artifacts under %s\n"
        "Local metadata written to %s\n"
        "When finished, fetch results e.g.:\n  %s",
        instance_name,
        run_prefix,
        meta_path,
        payload["fetch_results_hint"],
    )
