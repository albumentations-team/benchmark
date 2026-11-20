import subprocess
from pathlib import Path

import pytest

from augbench.cloud.gcp.storage import GcloudObjectStore


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [(0, b"", True), (1, b"412 Precondition Failed", False), (1, b"permission denied", None)],
)
def test_conditional_upload_preserves_payload_and_cleans_up(
    *, returncode: int, stderr: bytes, expected: bool | None
) -> None:
    uploaded_paths: list[Path] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        assert command[:5] == ["gcloud", "storage", "cp", "--quiet", "--if-generation-match=0"]
        assert command[-1] == "gs://bucket/runs/cell.json"
        source = Path(command[-2])
        uploaded_paths.append(source)
        assert source.read_bytes() == b'{"status":"ok"}\n'
        return subprocess.CompletedProcess(command, returncode, stdout=b"", stderr=stderr)

    store = GcloudObjectStore(base_uri="gs://bucket", executable="gcloud", runner=runner)
    if expected is None:
        with pytest.raises(RuntimeError, match="permission denied"):
            store.create_if_absent("runs/cell.json", b'{"status":"ok"}\n')
    else:
        assert store.create_if_absent("runs/cell.json", b'{"status":"ok"}\n') is expected

    assert len(uploaded_paths) == 1
    assert not uploaded_paths[0].exists()


@pytest.mark.parametrize(
    ("owner_matches", "delete_code", "expected"),
    [(True, 0, True), (True, 1, False), (False, 0, False)],
)
def test_lock_release_checks_owner_and_object_generation(
    *, owner_matches: bool, delete_code: int, expected: bool
) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        if command[2:4] == ["objects", "describe"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"37\n", stderr=b"")
        if command[2] == "cat":
            return subprocess.CompletedProcess(command, 0, stdout=b"owner" if owner_matches else b"other", stderr=b"")
        assert command == ["gcloud", "storage", "rm", "--quiet", "--if-generation-match=37", "gs://bucket/lock"]
        return subprocess.CompletedProcess(command, delete_code, stdout=b"", stderr=b"412 Precondition Failed")

    store = GcloudObjectStore(base_uri="gs://bucket", executable="gcloud", runner=runner)
    assert store.remove_if_matches("lock", b"owner") is expected
    assert sum(command[2] == "rm" for command in commands) == int(owner_matches)
