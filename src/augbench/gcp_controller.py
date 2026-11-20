"""GCE workers for a frozen RGB run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from augbench.cloud.gcp.instance_monitor import InstanceCommandRunner, delete_instance
from augbench.cloud.gcp.provisioning import (
    GceInstance,
    GcpProvisioningAttempt,
    GcpVmSpec,
    list_labeled_instances,
    provision_first_available,
    resolve_candidate_zones,
)
from augbench.remote_results import completed_cell_ids, pending_cells

_MAX_POOL_WORKERS = 24

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from augbench.guest_request import GuestRequest
    from augbench.run_config import GcpRunConfig
    from augbench.run_records import CellKey


class GcpControllerStore(Protocol):
    def create_if_absent(self, key: str, payload: bytes) -> bool: ...

    def read(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class GcpStartRequest:
    cloud: GcpRunConfig
    request: GuestRequest
    cells: tuple[CellKey, ...]
    startup_script: Path


@dataclass(frozen=True)
class GcpControllerDependencies:
    remote: GcpControllerStore
    executable: str = "gcloud"
    runner: InstanceCommandRunner | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


class GcpLaunch(BaseModel):
    """The controller's short, actionable state after one invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["complete", "active", "launched"]
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    pending_cell_ids: tuple[str, ...]
    instance_name: str | None = None
    zone: str | None = None
    instances: tuple[GceInstance, ...] = ()
    provisioning_errors: tuple[str, ...] = ()


def start_or_resume(
    *,
    start: GcpStartRequest,
    dependencies: GcpControllerDependencies,
    max_workers: int = 1,
) -> GcpLaunch:
    """Publish immutable inputs and launch workers for validated missing cells."""
    if not 1 <= max_workers <= _MAX_POOL_WORKERS:
        raise ValueError("max_workers must be between 1 and 24")
    cloud = start.cloud
    request = start.request
    cells = start.cells
    startup_script = start.startup_script
    remote = dependencies.remote
    executable = dependencies.executable
    command_runner = dependencies.runner or _run
    if request.gcs_base_uri.rstrip("/") != cloud.gcs_base_uri.rstrip("/"):
        raise ValueError("guest request and cloud config use different GCS roots")
    if not startup_script.is_file():
        raise FileNotFoundError(startup_script)
    matrix = tuple(cells)
    _publish_immutable(remote, f"runs/{request.run.run_id}/run.json", _json_bytes(request.run))
    completed = completed_cell_ids(remote=remote, run_id=request.run.run_id, cells=matrix)
    pending = pending_cells(cells=matrix, completed=completed)
    if not pending:
        return GcpLaunch(status="complete", run_id=request.run.run_id, pending_cell_ids=())

    _delete_reclaimable_augbench_instances(cloud=cloud, executable=executable, runner=command_runner)
    label_value = request.run.run_id[:12]
    instances = list_labeled_instances(
        project=cloud.project,
        label_key="augbench-run",
        label_value=label_value,
        executable=executable,
        runner=command_runner,
    )
    active = tuple(instance for instance in instances if instance.active)
    if active:
        instance = active[0]
        return GcpLaunch(
            status="active",
            run_id=request.run.run_id,
            pending_cell_ids=tuple(cell.cell_id for cell in pending),
            instance_name=instance.name,
            zone=instance.zone,
            instances=active,
        )
    for instance in instances:
        delete_instance(
            project=cloud.project,
            zone=instance.zone,
            instance_name=instance.name,
            executable=executable,
            runner=command_runner,
        )

    pending_request = request.model_copy(update={"pending_cell_ids": tuple(cell.cell_id for cell in pending)})
    if max_workers > 1:
        return _start_pool(start=start, dependencies=dependencies, request=pending_request, max_workers=max_workers)
    request_digest = hashlib.sha256(_json_bytes(pending_request)).hexdigest()
    request_key = f"runs/{request.run.run_id}/requests/{request_digest}.json"
    _publish_immutable(remote, request_key, _json_bytes(pending_request))
    request_uri = f"{cloud.gcs_base_uri.rstrip('/')}/{request_key}"
    family = request.run.family_config["family"]
    instance_name = f"augbench-{family}-{label_value}"
    zones = resolve_candidate_zones(
        configured_zones=cloud.zones,
        project=cloud.project,
        machine_type=cloud.machine_type,
        executable=executable,
        runner=command_runner,
    )
    zone = provision_first_available(
        attempt=GcpProvisioningAttempt(
            zones=zones,
            spec_for_zone=lambda candidate: _vm_spec(
                start=start, request_uri=request_uri, instance_name=instance_name, zone=candidate
            ),
            executable=executable,
            runner=command_runner,
            deadline_seconds=cloud.provisioning_deadline_seconds,
            poll_seconds=30.0,
            clock=dependencies.clock,
            sleep=dependencies.sleep,
        ),
    )
    return GcpLaunch(
        status="launched",
        run_id=request.run.run_id,
        pending_cell_ids=tuple(cell.cell_id for cell in pending),
        instance_name=instance_name,
        zone=zone,
    )


def _start_pool(
    *, start: GcpStartRequest, dependencies: GcpControllerDependencies, request: GuestRequest, max_workers: int
) -> GcpLaunch:
    request = request.model_copy(update={"pool_id": uuid.uuid4().hex})
    key = f"runs/{request.run.run_id}/pools/{request.pool_id}/request.json"
    _publish_immutable(dependencies.remote, key, _json_bytes(request))
    cloud = start.cloud
    runner = dependencies.runner or _run
    zones = resolve_candidate_zones(
        configured_zones=cloud.zones,
        project=cloud.project,
        machine_type=cloud.machine_type,
        executable=dependencies.executable,
        runner=runner,
    )
    pending_ids = set(request.pending_cell_ids)
    jobs = {(cell.implementation, cell.seed) for cell in start.cells if cell.cell_id in pending_ids}
    workers = min(max_workers, len(jobs))

    def provision(index: int) -> GceInstance | str:
        name = f"augbench-{request.run.family_config['family']}-{request.run.run_id[:12]}-{index:02d}"
        ordered_zones = zones[index % len(zones) :] + zones[: index % len(zones)]
        try:
            zone = provision_first_available(
                attempt=GcpProvisioningAttempt(
                    zones=ordered_zones,
                    spec_for_zone=lambda candidate: _vm_spec(
                        start=start,
                        request_uri=f"{cloud.gcs_base_uri.rstrip('/')}/{key}",
                        instance_name=name,
                        zone=candidate,
                    ),
                    executable=dependencies.executable,
                    runner=runner,
                    deadline_seconds=cloud.provisioning_deadline_seconds,
                    poll_seconds=30,
                    clock=dependencies.clock,
                    sleep=dependencies.sleep,
                )
            )
        except (RuntimeError, TimeoutError) as error:
            return f"{name}: {error}"
        return GceInstance(name=name, zone=zone, status="RUNNING")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(provision, range(workers)))
    instances = tuple(result for result in results if isinstance(result, GceInstance))
    errors = tuple(result for result in results if isinstance(result, str))
    if not instances:
        raise RuntimeError("no pool workers launched: " + "; ".join(errors))
    return GcpLaunch(
        status="launched",
        run_id=request.run.run_id,
        pending_cell_ids=request.pending_cell_ids,
        instances=instances,
        provisioning_errors=errors,
    )


def _vm_spec(*, start: GcpStartRequest, request_uri: str, instance_name: str, zone: str) -> GcpVmSpec:
    cloud = start.cloud
    return GcpVmSpec(
        project=cloud.project,
        zone=zone,
        instance_name=instance_name,
        machine_type=cloud.machine_type,
        image=cloud.image,
        boot_disk_size_gib=cloud.boot_disk_size_gib,
        service_account=cloud.service_account,
        startup_script=start.startup_script,
        max_run_duration_seconds=cloud.run_deadline_seconds,
        provisioning_model=cloud.provisioning_model,
        labels={
            "augbench": "1",
            "augbench-run": start.request.run.run_id[:12],
            "augbench-family": start.request.run.family_config["family"],
        },
        metadata={"augbench-request-uri": request_uri},
    )


def _delete_reclaimable_augbench_instances(
    *,
    cloud: GcpRunConfig,
    executable: str,
    runner: InstanceCommandRunner,
) -> None:
    """Free only old terminal augbench disks before requesting a new L4 VM."""
    instances = list_labeled_instances(
        project=cloud.project,
        label_key="augbench",
        label_value="1",
        executable=executable,
        runner=runner,
    )
    for instance in instances:
        if instance.reclaimable:
            delete_instance(
                project=cloud.project,
                zone=instance.zone,
                instance_name=instance.name,
                executable=executable,
                runner=runner,
            )


def _publish_immutable(remote: GcpControllerStore, key: str, payload: bytes) -> None:
    if not remote.create_if_absent(key, payload) and remote.read(key) != payload:
        raise RuntimeError(f"immutable GCS object conflicts at {key}")


def _json_bytes(value: BaseModel) -> bytes:
    return f"{json.dumps(value.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))}\n".encode()


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=False, capture_output=True)  # noqa: S603 - fixed gcloud argument vector.
