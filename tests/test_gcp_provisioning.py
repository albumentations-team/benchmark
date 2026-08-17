import subprocess

from augbench.cloud.gcp.provisioning import list_labeled_instances


def test_lists_only_the_requested_augbench_instances() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"augbench-rgb,us-central1-a,RUNNING\n", stderr=b"")

    instances = list_labeled_instances(
        project="albumentations",
        label_key="augbench-run",
        label_value="123456789abc",
        executable="gcloud",
        runner=runner,
    )

    assert instances[0].name == "augbench-rgb"
    assert instances[0].active
    assert "--filter=labels.augbench-run=123456789abc" in commands[0]


def test_terminal_instance_is_not_active() -> None:
    def runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=b"augbench-rgb,us-central1-a,TERMINATED\n", stderr=b"")

    instance = list_labeled_instances(
        project="albumentations",
        label_key="augbench-run",
        label_value="123456789abc",
        executable="gcloud",
        runner=runner,
    )[0]

    assert not instance.active
