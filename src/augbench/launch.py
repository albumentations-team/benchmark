"""Prepare and submit a disk-to-CUDA image benchmark on L4."""

from __future__ import annotations

import tempfile
from pathlib import Path

from augbench.cloud.gcp.storage import GcloudObjectStore
from augbench.frozen_run import FrozenRun, build_frozen_run
from augbench.gcp_controller import GcpControllerDependencies, GcpLaunch, GcpStartRequest, start_or_resume
from augbench.run_config import FamilyRunConfig, load_family_config, load_gcp_config
from augbench.source_archive import create_clean_code_archive


def launch(*, repository_root: Path, config: FamilyRunConfig | None = None) -> tuple[FrozenRun, GcpLaunch]:
    """Upload a clean code artifact, then resume the run or top up its L4 fleet."""
    if config is None:
        config = load_family_config(repository_root / "configs/families/rgb.yaml")
    cloud = load_gcp_config(repository_root / config.cloud)
    with tempfile.TemporaryDirectory(prefix="augbench-source-") as directory:
        archive = create_clean_code_archive(
            repository_root=repository_root,
            output=Path(directory) / "source.tar.gz",
        )
        frozen = build_frozen_run(
            repository_root=repository_root,
            git_commit=archive.commit,
            code_archive_sha256=archive.sha256,
            config=config,
        )
        remote = GcloudObjectStore(base_uri=cloud.gcs_base_uri)
        key = f"code/{archive.sha256}/source.tar.gz"
        if not remote.create_file_if_absent(key, archive.path) and remote.read(key) != archive.path.read_bytes():
            raise RuntimeError("existing GCS code archive conflicts with its SHA-256 identity")
    launch = start_or_resume(
        start=GcpStartRequest(
            cloud=cloud,
            request=frozen.request,
            cells=frozen.cells,
            startup_script=repository_root / "infra/gcp/bootstrap.sh",
        ),
        dependencies=GcpControllerDependencies(remote=remote),
    )
    return frozen, launch
