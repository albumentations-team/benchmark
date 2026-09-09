import subprocess
from pathlib import Path

import pytest

from augbench.source_archive import create_clean_code_archive


def test_refuses_to_archive_a_dirty_worktree(tmp_path: Path) -> None:
    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert command == ["git", "status", "--porcelain"]
        return subprocess.CompletedProcess(command, 0, stdout=b" M src/augbench/example.py\n", stderr=b"")

    with pytest.raises(ValueError, match="clean Git worktree"):
        create_clean_code_archive(repository_root=tmp_path, output=tmp_path / "source.tar.gz", runner=runner)


def test_refuses_an_invalid_head_identifier(tmp_path: Path) -> None:
    responses = iter(
        (
            subprocess.CompletedProcess(["git"], 0, stdout=b"", stderr=b""),
            subprocess.CompletedProcess(["git"], 0, stdout=b"not-a-commit\n", stderr=b""),
        ),
    )

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del command
        return next(responses)

    with pytest.raises(ValueError, match="invalid commit"):
        create_clean_code_archive(repository_root=tmp_path, output=tmp_path / "source.tar.gz", runner=runner)
