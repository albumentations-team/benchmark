"""One-VM GCE controller for a frozen RGB run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from augbench.cloud.gcp.instance_monitor import InstanceCommandRunner, delete_instance
from augbench.cloud.gcp.provisioning import (
    GcpVmSpec,
    list_labeled_instances,
    provision_first_available,
    resolve_candidate_zones,
)
from augbench.remote_results import completed_cell_ids, pending_cells

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from augbench.guest_request import GuestRequest
    from augbench.run_config import GcpRunConfig
    from augbench.run_records import CellKey


class GcpControllerStore(Protocol):
    def create_if_absent(self, key: str, payload: bytes) -> bool: ...

    def read(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> tuple[str, ...]: ...


class GcpLaunch(BaseModel):
    """The controller's short, actionable state after one invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["complete", "active", "launched"]
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    pending_cell_ids: tuple[str, ...]
    instance_name: str | None = None
    zone: str | None = None


def start_or_resume(
    *,
    cloud: GcpRunConfig,
    request: GuestRequest,
    cells: Iterable[CellKey],
    startup_script: Path,
    remote: GcpControllerStore,
    executable: str = "gcloud",
    runner: InstanceCommandRunner | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> GcpLaunch:
    """Publish immutable input, reuse valid cells, then create at most one VM."""
    if request.gcs_base_uri.rstrip("/") != cloud.gcs_base_uri.rstrip("/"):
        raise ValueError("guest request and cloud config use different GCS roots")
    if not startup_script.is_file():
        raise FileNotFoundError(startup_script)
    command_runner = runner or _run
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
    request_digest = hashlib.sha256(_json_bytes(pending_request)).hexdigest()
    request_key = f"runs/{request.run.run_id}/requests/{request_digest}.json"
    _publish_immutable(remote, request_key, _json_bytes(pending_request))
    request_uri = f"{cloud.gcs_base_uri.rstrip('/')}/{request_key}"
    instance_name = f"augbench-rgb-{label_value}"
    zones = resolve_candidate_zones(
        configured_zones=cloud.zones,
        project=cloud.project,
        machine_type=cloud.machine_type,
        executable=executable,
        runner=command_runner,
    )
    zone = provision_first_available(
        zones=zones,
        spec_for_zone=lambda candidate: GcpVmSpec(
            project=cloud.project,
            zone=candidate,
            instance_name=instance_name,
            machine_type=cloud.machine_type,
            image=cloud.image,
            boot_disk_size_gib=cloud.boot_disk_size_gib,
            service_account=cloud.service_account,
            startup_script=startup_script,
            max_run_duration_seconds=cloud.run_deadline_seconds,
            provisioning_model=cloud.provisioning_model,
            labels={"augbench": "1", "augbench-run": label_value, "augbench-family": "rgb"},
            metadata={"augbench-request-uri": request_uri},
        ),
        executable=executable,
        runner=command_runner,
        deadline_seconds=cloud.provisioning_deadline_seconds,
        poll_seconds=30.0,
        clock=clock,
        sleep=sleep,
    )
    return GcpLaunch(
        status="launched",
        run_id=request.run.run_id,
        pending_cell_ids=tuple(cell.cell_id for cell in pending),
        instance_name=instance_name,
        zone=zone,
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
