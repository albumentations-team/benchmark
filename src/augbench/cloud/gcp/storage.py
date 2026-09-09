from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import Protocol


class CommandRunner(Protocol):
    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[bytes]: ...


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=False, capture_output=True)  # noqa: S603 - fixed gcloud argument vector.


class GcloudObjectStore:
    def __init__(
        self,
        *,
        base_uri: str,
        executable: str | None = None,
        runner: CommandRunner = _run,
    ) -> None:
        self._base_uri = base_uri.rstrip("/")
        if not self._base_uri.startswith("gs://"):
            raise ValueError("GCS base URI must start with gs://")
        resolved_executable = executable or which("gcloud")
        if resolved_executable is None:
            raise RuntimeError("gcloud executable was not found")
        self._executable = resolved_executable
        self._runner = runner

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        with tempfile.NamedTemporaryFile(prefix="augbench-gcs-", delete_on_close=False) as temporary:
            temporary.write(payload)
            temporary.flush()
            return self.create_file_if_absent(key, Path(temporary.name))

    def create_file_if_absent(self, key: str, source: Path) -> bool:
        uri = self._uri(key)
        completed = self._runner(
            [
                self._executable,
                "storage",
                "cp",
                "--quiet",
                "--if-generation-match=0",
                str(source),
                uri,
            ],
        )
        if completed.returncode == 0:
            return True
        error_text = completed.stderr.decode(errors="replace")
        if "412" in error_text or "Precondition" in error_text:
            return False
        raise RuntimeError(f"gcloud conditional upload failed for {uri}: {error_text.strip()}")

    def read(self, key: str) -> bytes:
        uri = self._uri(key)
        completed = self._runner([self._executable, "storage", "cat", "--quiet", uri])
        if completed.returncode != 0:
            error_text = completed.stderr.decode(errors="replace")
            raise RuntimeError(f"gcloud read failed for {uri}: {error_text.strip()}")
        return completed.stdout

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        uri = self._uri(prefix.rstrip("/"))
        completed = self._runner([self._executable, "storage", "ls", "--recursive", uri])
        if completed.returncode != 0:
            error_text = completed.stderr.decode(errors="replace")
            lowered = error_text.lower()
            if "not found" in lowered or "no urls matched" in lowered or "matched no objects" in lowered:
                return ()
            raise RuntimeError(f"gcloud list failed for {uri}: {error_text.strip()}")

        base_prefix = f"{self._base_uri}/"
        key_prefix = prefix.rstrip("/") + "/"
        keys: set[str] = set()
        for raw_line in completed.stdout.decode(errors="replace").splitlines():
            object_uri = raw_line.strip()
            if not object_uri.startswith(base_prefix) or object_uri.endswith("/:"):
                continue
            key = object_uri.removeprefix(base_prefix)
            if key.startswith(key_prefix):
                keys.add(key)
        return tuple(sorted(keys))

    def _uri(self, key: str) -> str:
        parts = key.split("/")
        if not key or key.startswith(("/", "gs://")) or ".." in parts:
            raise ValueError(f"invalid relative GCS object key: {key!r}")
        return f"{self._base_uri}/{key}"
