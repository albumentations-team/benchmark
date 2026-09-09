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
