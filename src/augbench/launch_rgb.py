"""Prepare and submit the one supported production benchmark: RGB on L4."""

from __future__ import annotations

import tempfile
from pathlib import Path

from augbench.cloud.gcp.storage import GcloudObjectStore
from augbench.frozen_rgb_run import FrozenRgbRun, build_frozen_rgb_run
from augbench.gcp_controller import GcpLaunch, start_or_resume
from augbench.run_config import load_gcp_config
from augbench.source_archive import create_clean_code_archive


def launch_rgb(*, repository_root: Path) -> tuple[FrozenRgbRun, GcpLaunch]:
    """Upload a clean code artifact and either resume or launch the one L4 VM."""
    cloud = load_gcp_config(repository_root / "configs/cloud/gcp-l4.yaml")
    with tempfile.TemporaryDirectory(prefix="augbench-source-") as directory:
        archive = create_clean_code_archive(
            repository_root=repository_root,
            output=Path(directory) / "source.tar.gz",
        )
        frozen = build_frozen_rgb_run(
            repository_root=repository_root,
            git_commit=archive.commit,
            code_archive_sha256=archive.sha256,
        )
        remote = GcloudObjectStore(base_uri=cloud.gcs_base_uri)
        key = f"code/{archive.sha256}/source.tar.gz"
        if not remote.create_file_if_absent(key, archive.path) and remote.read(key) != archive.path.read_bytes():
            raise RuntimeError("existing GCS code archive conflicts with its SHA-256 identity")
    launch = start_or_resume(
        cloud=cloud,
        request=frozen.request,
        cells=frozen.cells,
        startup_script=repository_root / "infra/gcp/bootstrap.sh",
        remote=remote,
    )
    return frozen, launch
