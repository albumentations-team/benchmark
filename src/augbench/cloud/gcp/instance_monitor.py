from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from augbench.cloud.gcp.provisioning import build_delete_instance_command

if TYPE_CHECKING:
    import subprocess


class InstanceCommandRunner(Protocol):
    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[bytes]: ...


TERMINAL_INSTANCE_STATES = frozenset({"STOPPING", "TERMINATED", "SUSPENDING", "SUSPENDED", "NOT_FOUND"})


def query_instance_state(
    *,
    project: str,
    zone: str,
    instance_name: str,
    executable: str,
    runner: InstanceCommandRunner,
) -> str:
    command = [
        executable,
        "compute",
        "instances",
        "describe",
        instance_name,
        f"--project={project}",
        f"--zone={zone}",
        "--format=value(status)",
    ]
    completed = runner(command)
    if completed.returncode == 0:
        return completed.stdout.decode(errors="replace").strip().upper()
    error = completed.stderr.decode(errors="replace").lower()
    if "not found" in error or "was not found" in error:
        return "NOT_FOUND"
    raise RuntimeError(f"GCE instance state query failed: {error.strip()}")


def delete_instance(
    *,
    project: str,
    zone: str,
    instance_name: str,
    executable: str,
    runner: InstanceCommandRunner,
) -> None:
    if (
        query_instance_state(
            project=project,
            zone=zone,
            instance_name=instance_name,
            executable=executable,
            runner=runner,
        )
        == "NOT_FOUND"
    ):
        return
    completed = runner(
        build_delete_instance_command(
            project=project,
            zone=zone,
            instance_name=instance_name,
            executable=executable,
        ),
    )
    error = completed.stderr.decode(errors="replace")
    if completed.returncode != 0 and "not found" not in error.lower() and "was not found" not in error.lower():
        raise RuntimeError(f"GCE instance deletion failed: {error.strip()}")
