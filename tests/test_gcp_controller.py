import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from augbench.gcp_controller import GcpControllerDependencies, GcpStartRequest, start_or_resume
from augbench.guest_request import GuestRequest
from augbench.run_config import GcpRunConfig
from augbench.run_records import CellKey, RunInputs, build_run_record


@dataclass
class _Remote:
    objects: dict[str, bytes] = field(default_factory=dict)

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        if key in self.objects:
            return False
        self.objects[key] = payload
        return True

    def read(self, key: str) -> bytes:
        return self.objects[key]

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(key for key in self.objects if key.startswith(prefix))


def _cloud() -> GcpRunConfig:
    return GcpRunConfig(
        project="albumentations",
        gcs_base_uri="gs://bucket/benchmark",
        zones="all",
        machine_type="g2-standard-16",
        accelerator="nvidia-l4",
        image="projects/example/global/images/example",
        service_account="benchmark@example.iam.gserviceaccount.com",
        python_version="3.13.14",
        provisioning_model="STANDARD",
        boot_disk_size_gib=350,
        provisioning_deadline_seconds=60,
        run_deadline_seconds=3600,
    )


def _request() -> GuestRequest:
    run = build_run_record(
        family_config={"family": "rgb"},
        inputs=RunInputs(
            git_commit="a" * 40,
            code_archive_sha256="b" * 64,
            dataset_archive_sha256="d" * 64,
            recipe_catalog_sha256="e" * 64,
            environment_lock_sha256={"rgb": "f" * 64},
        ),
        hardware={"machine_type": "g2-standard-16"},
    )
    return GuestRequest(
        run=run,
        gcs_base_uri="gs://bucket/benchmark",
        code_archive_uri="gs://bucket/code.tar.gz",
        dataset_archive_uri="gs://bucket/dataset.tar",
        dataset_archive_sha256="d" * 64,
        environment_cache_uri="gs://bucket/environment.tar.gz",
        environment_lock_path="environments/rgb/lock.txt",
        environment_lock_sha256="f" * 64,
        environment_python_version="3.13.14",
        pending_cell_ids=("0" * 64,),
    )


def _cell(request: GuestRequest) -> CellKey:
    return CellKey(
        run_id=request.run.run_id,
        family="rgb",
        implementation="pillow_cpu",
        recipe_id="Resize224+Normalize+ToTensor",
        seed=137,
    )


def test_controller_creates_one_labeled_vm_for_missing_cells(tmp_path: Path) -> None:
    startup = tmp_path / "bootstrap.sh"
    startup.write_text("#!/bin/bash\n", encoding="utf-8")
    request = _request()
    cell = _cell(request)
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        if command[2:4] == ["instances", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[2:4] in (["accelerator-types", "list"], ["machine-types", "list"]):
            return subprocess.CompletedProcess(command, 0, stdout=b"us-central1-a\n", stderr=b"")
        if command[2:4] == ["instances", "create"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise AssertionError(command)

    launch = start_or_resume(
        start=GcpStartRequest(
            cloud=_cloud(),
            request=request,
            cells=(cell,),
            startup_script=startup,
        ),
        dependencies=GcpControllerDependencies(
            remote=_Remote(),
            runner=runner,
            sleep=lambda _seconds: None,
        ),
    )

    assert launch.status == "launched"
    create = next(command for command in commands if command[2:4] == ["instances", "create"])
    assert "--provisioning-model=STANDARD" in create
    assert "--labels=augbench=1,augbench-family=rgb,augbench-run=" + request.run.run_id[:12] in create


def test_controller_reclaims_only_terminal_augbench_instances(tmp_path: Path) -> None:
    startup = tmp_path / "bootstrap.sh"
    startup.write_text("#!/bin/bash\n", encoding="utf-8")
    request = _request()
    cell = _cell(request)
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        if command[2:4] == ["instances", "list"]:
            if "--filter=labels.augbench=1" in command:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=b"old-terminal,us-central1-a,TERMINATED\nold-running,us-central1-b,RUNNING\n",
                    stderr=b"",
                )
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[2:4] == ["instances", "describe"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"TERMINATED\n", stderr=b"")
        if command[2:4] == ["instances", "delete"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[2:4] in (["accelerator-types", "list"], ["machine-types", "list"]):
            return subprocess.CompletedProcess(command, 0, stdout=b"us-central1-a\n", stderr=b"")
        if command[2:4] == ["instances", "create"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise AssertionError(command)

    start_or_resume(
        start=GcpStartRequest(
            cloud=_cloud(),
            request=request,
            cells=(cell,),
            startup_script=startup,
        ),
        dependencies=GcpControllerDependencies(
            remote=_Remote(),
            runner=runner,
            sleep=lambda _seconds: None,
        ),
    )

    deletes = [command for command in commands if command[2:4] == ["instances", "delete"]]
    assert deletes[0][4] == "old-terminal"
    assert all("old-running" not in command for command in deletes)


def test_controller_never_launches_a_duplicate_active_vm(tmp_path: Path) -> None:
    startup = tmp_path / "bootstrap.sh"
    startup.write_text("#!/bin/bash\n", encoding="utf-8")
    request = _request()
    cell = _cell(request)

    def runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        assert command[2:4] == ["instances", "list"]
        return subprocess.CompletedProcess(command, 0, stdout=b"augbench-rgb,us-central1-a,RUNNING\n", stderr=b"")

    launch = start_or_resume(
        start=GcpStartRequest(
            cloud=_cloud(),
            request=request,
            cells=(cell,),
            startup_script=startup,
        ),
        dependencies=GcpControllerDependencies(remote=_Remote(), runner=runner),
    )

    assert launch.status == "active"
    assert launch.instance_name == "augbench-rgb"
