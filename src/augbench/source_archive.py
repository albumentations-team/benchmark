"""Create an immutable code archive from a clean Git commit."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - archive paths are used at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class CodeArchive:
    commit: str
    path: Path
    sha256: str


def create_clean_code_archive(
    *,
    repository_root: Path,
    output: Path,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> CodeArchive:
    """Archive HEAD only when it exactly matches the working tree."""
    status = runner(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if status.returncode != 0:
        raise RuntimeError(_error("could not inspect the Git worktree", status))
    if status.stdout.strip():
        raise ValueError("production requires a clean Git worktree; commit the benchmark inputs first")
    revision = runner(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if revision.returncode != 0:
        raise RuntimeError(_error("could not resolve HEAD", revision))
    commit = revision.stdout.decode().strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Git returned an invalid commit ID")
    output.parent.mkdir(parents=True, exist_ok=True)
    archived = runner(
        ["git", "archive", "--format=tar.gz", f"--output={output}", commit],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if archived.returncode != 0:
        raise RuntimeError(_error("could not archive the clean commit", archived))
    return CodeArchive(commit=commit, path=output, sha256=_sha256_file(output))


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _error(prefix: str, completed: subprocess.CompletedProcess[bytes]) -> str:
    return f"{prefix}: {completed.stderr.decode(errors='replace').strip()}"
