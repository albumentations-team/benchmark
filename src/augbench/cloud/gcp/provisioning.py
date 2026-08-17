from __future__ import annotations

import re
from pathlib import Path  # noqa: TC003 - Pydantic resolves this annotation at runtime.
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable

_LABEL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_LABEL_VALUE_PATTERN = re.compile(r"^[a-z0-9_-]{0,63}$")


class GcpVmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    instance_name: str = Field(min_length=1, max_length=63)
    machine_type: str = Field(min_length=1)
    image: str = Field(min_length=1)
    boot_disk_size_gib: int = Field(ge=50)
    service_account: str = Field(min_length=1)
    startup_script: Path
    max_run_duration_seconds: int = Field(ge=60)
    provisioning_model: Literal["STANDARD", "SPOT"]
    labels: dict[str, str]
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: dict[str, str]) -> dict[str, str]:
        for key, value in labels.items():
            if not _LABEL_KEY_PATTERN.fullmatch(key) or not _LABEL_VALUE_PATTERN.fullmatch(value):
                raise ValueError(f"invalid GCE label {key}={value}")
        return labels


def build_create_instance_command(spec: GcpVmSpec, *, executable: str = "gcloud") -> list[str]:
    labels = ",".join(f"{key}={value}" for key, value in sorted(spec.labels.items()))
    command = [
        executable,
        "compute",
        "instances",
        "create",
        spec.instance_name,
        f"--project={spec.project}",
        f"--zone={spec.zone}",
        f"--machine-type={spec.machine_type}",
        f"--image={spec.image}",
        f"--boot-disk-size={spec.boot_disk_size_gib}GB",
        "--boot-disk-type=pd-balanced",
        f"--service-account={spec.service_account}",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        "--maintenance-policy=TERMINATE",
        f"--provisioning-model={spec.provisioning_model}",
        f"--max-run-duration={spec.max_run_duration_seconds}s",
        "--instance-termination-action=DELETE",
        "--no-restart-on-failure",
        f"--metadata-from-file=startup-script={spec.startup_script}",
        f"--labels={labels}",
    ]
    if spec.metadata:
        metadata = ",".join(f"{key}={value}" for key, value in sorted(spec.metadata.items()))
        command.append(f"--metadata={metadata}")
    command.append("--quiet")
    return command


def build_delete_instance_command(
    *,
    project: str,
    zone: str,
    instance_name: str,
    executable: str = "gcloud",
) -> list[str]:
    return [
        executable,
        "compute",
        "instances",
        "delete",
        instance_name,
        f"--project={project}",
        f"--zone={zone}",
        "--quiet",
    ]


def discover_g2_l4_zones(
    *,
    project: str,
    machine_type: str,
    executable: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[bytes]],
) -> tuple[str, ...]:
    accelerator_command = [
        executable,
        "compute",
        "accelerator-types",
        "list",
        f"--project={project}",
        "--filter=name=nvidia-l4",
        "--format=value(zone.basename())",
    ]
    accelerator_result = runner(accelerator_command)
    if accelerator_result.returncode != 0:
        message = accelerator_result.stderr.decode(errors="replace")
        raise RuntimeError(f"failed to discover NVIDIA L4 zones: {message.strip()}")
    machine_command = [
        executable,
        "compute",
        "machine-types",
        "list",
        f"--project={project}",
        f"--filter=name={machine_type}",
        "--format=value(zone.basename())",
    ]
    machine_result = runner(machine_command)
    if machine_result.returncode != 0:
        message = machine_result.stderr.decode(errors="replace")
        raise RuntimeError(f"failed to discover {machine_type} zones: {message.strip()}")
    accelerator_zones = set(accelerator_result.stdout.decode(errors="replace").split())
    machine_zones = set(machine_result.stdout.decode(errors="replace").split())
    zones = tuple(sorted(accelerator_zones & machine_zones))
    if not zones:
        raise RuntimeError(f"GCE returned no zones with both NVIDIA L4 and {machine_type}")
    return zones


def resolve_candidate_zones(
    *,
    configured_zones: tuple[str, ...] | Literal["all"],
    project: str,
    machine_type: str,
    executable: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[bytes]],
) -> tuple[str, ...]:
    if configured_zones != "all":
        return configured_zones
    return discover_g2_l4_zones(
        project=project,
        machine_type=machine_type,
        executable=executable,
        runner=runner,
    )


def provision_first_available(
    *,
    zones: tuple[str, ...],
    spec_for_zone: Callable[[str], GcpVmSpec],
    executable: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[bytes]],
    deadline_seconds: float,
    poll_seconds: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> str:
    started = clock()
    errors: list[str] = []
    while True:
        errors.clear()
        for zone in zones:
            completed = runner(build_create_instance_command(spec_for_zone(zone), executable=executable))
            if completed.returncode == 0:
                return zone
            message = completed.stderr.decode(errors="replace")[-4096:]
            if not is_retryable_capacity_error(message):
                raise RuntimeError(f"GCE provisioning failed in {zone}: {message.strip()}")
            errors.append(f"{zone}: {message.strip()}")
        elapsed = clock() - started
        if elapsed >= deadline_seconds:
            raise TimeoutError(f"GCE capacity remained unavailable until deadline: {'; '.join(errors)}")
        sleep(min(poll_seconds, deadline_seconds - elapsed, 60.0))


def is_retryable_capacity_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "zone_resource_pool_exhausted",
            "stockout",
            "does not have enough resources",
            "currently unavailable",
            "resource_availability",
        )
    )


class GceInstance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    status: str = Field(min_length=1)

    @property
    def active(self) -> bool:
        return self.status.upper() not in {"TERMINATED", "STOPPING", "SUSPENDED", "SUSPENDING"}

    @property
    def reclaimable(self) -> bool:
        """A stopped VM can retain its boot disk and consume regional quota."""
        return self.status.upper() in {"TERMINATED", "SUSPENDED"}


def list_labeled_instances(
    *,
    project: str,
    label_key: str,
    label_value: str,
    executable: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[bytes]],
) -> tuple[GceInstance, ...]:
    command = [
        executable,
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter=labels.{label_key}={label_value}",
        "--format=csv[no-heading](name,zone.basename(),status)",
    ]
    completed = runner(command)
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace")
        raise RuntimeError(f"GCE instance listing failed: {message.strip()}")
    instances: list[GceInstance] = []
    for line in completed.stdout.decode(errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not any(fields):
            continue
        instances.append(GceInstance(name=fields[0], zone=fields[1], status=fields[2]))
    return tuple(instances)
