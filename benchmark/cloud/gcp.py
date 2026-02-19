r"""GCP cloud execution module.

Automates the full lifecycle of a remote benchmark run:

    1. Create GCP Compute Engine instance
    2. Wait for it to be SSH-ready
    3. Upload the repo
    4. Run `benchmark.cli run` remotely (same CLI, no cloud code in the runner)
    5. Download results back to local output dir
    6. Delete the instance

Usage from the CLI (benchmark.cli run --cloud gcp ...):

    python -m benchmark.cli run \\
        --cloud gcp \\
        --gcp-project my-project \\
        --gcp-zone us-central1-a \\
        --gcp-machine-type n1-standard-8 \\
        --data-dir /data/images \\
        --output ./results

Or programmatically:

    from benchmark.cloud.gcp import GCPRunner
    from benchmark.cloud.instance import GCPInstanceConfig

    config = GCPInstanceConfig.gpu_t4(project="my-project")
    runner = GCPRunner(config)
    runner.run(
        repo_root=Path("."),
        remote_cli_args=["--media", "image", "--libraries", "albumentationsx"],
        local_output_dir=Path("./results"),
        remote_data_dir="/data/images",
    )
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from .instance import GCPInstanceConfig

logger = logging.getLogger(__name__)

_GCLOUD = "gcloud"


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    logger.debug("$ %s", shlex.join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def _run_stream(cmd: list[str], **kwargs: Any) -> int:
    """Run command streaming stdout/stderr to the terminal."""
    logger.debug("$ %s", shlex.join(cmd))
    result = subprocess.run(cmd, text=True, check=False, **kwargs)
    return result.returncode


class GCPRunner:
    """Manages the lifecycle of a GCP benchmark instance."""

    def __init__(self, config: GCPInstanceConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    def create_instance(self) -> None:
        """Create the GCP Compute Engine instance."""
        cfg = self.config
        cmd = [
            _GCLOUD,
            "compute",
            "instances",
            "create",
            cfg.instance_name,
            "--project",
            cfg.project,
            "--zone",
            cfg.zone,
            "--machine-type",
            cfg.machine_type,
            "--image-family",
            cfg.image_family,
            "--image-project",
            cfg.image_project,
            f"--boot-disk-size={cfg.disk_size_gb}GB",
            "--scopes",
            ",".join(cfg.scopes),
        ]

        if cfg.tags:
            cmd += ["--tags", ",".join(cfg.tags)]

        if cfg.accelerator_type and cfg.accelerator_count > 0:
            cmd += [
                "--accelerator",
                f"type={cfg.accelerator_type},count={cfg.accelerator_count}",
                "--maintenance-policy",
                "TERMINATE",
            ]

        if cfg.preemptible:
            cmd.append("--preemptible")

        logger.info("Creating instance %s in %s...", cfg.instance_name, cfg.zone)
        _run(cmd)
        logger.info("Instance created.")

    def delete_instance(self) -> None:
        """Delete the GCP instance (non-interactive)."""
        cfg = self.config
        logger.info("Deleting instance %s...", cfg.instance_name)
        _run(
            [
                _GCLOUD,
                "compute",
                "instances",
                "delete",
                cfg.instance_name,
                "--project",
                cfg.project,
                "--zone",
                cfg.zone,
                "--quiet",
            ],
        )
        logger.info("Instance deleted.")

    def wait_for_ssh(self, timeout: int = 300, poll_interval: int = 10) -> None:
        """Poll until SSH is available on the instance."""
        cfg = self.config
        deadline = time.time() + timeout
        logger.info("Waiting for SSH on %s...", cfg.instance_name)
        while time.time() < deadline:
            result = subprocess.run(
                [
                    _GCLOUD,
                    "compute",
                    "ssh",
                    cfg.instance_name,
                    "--project",
                    cfg.project,
                    "--zone",
                    cfg.zone,
                    "--command",
                    "echo ready",
                    "--strict-host-key-checking=no",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and "ready" in result.stdout:
                logger.info("SSH ready.")
                return
            logger.debug("SSH not ready yet, retrying in %ds...", poll_interval)
            time.sleep(poll_interval)
        raise TimeoutError(f"SSH not available on {cfg.instance_name} after {timeout}s")

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    def upload_repo(self, repo_root: Path, remote_dir: str = "~/benchmark") -> None:
        """Upload the repo to the remote instance via a tar archive, excluding venvs and outputs."""
        cfg = self.config
        archive_path = repo_root / ".upload_repo.tar.gz"

        exclude_patterns = [
            ".venv*",
            "venv*",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "outputs",
            "output",
            "results",
            ".git",
        ]
        exclude_flags = [flag for pattern in exclude_patterns for flag in ("--exclude", pattern)]

        try:
            archive_path.unlink(missing_ok=True)
            logger.info("Creating repo archive for upload (excluding venvs and outputs)...")
            subprocess.run(
                ["tar", "czf", str(archive_path), *exclude_flags, "-C", str(repo_root), "."],
                check=True,
            )

            logger.info("Uploading repo archive to %s:%s...", cfg.instance_name, remote_dir)
            _run_stream(
                [
                    _GCLOUD,
                    "compute",
                    "ssh",
                    cfg.instance_name,
                    "--project",
                    cfg.project,
                    "--zone",
                    cfg.zone,
                    "--command",
                    f"mkdir -p {shlex.quote(remote_dir)}",
                ],
            )
            remote_archive = f"{cfg.instance_name}:{remote_dir}/repo.tar.gz"
            _run_stream(
                [
                    _GCLOUD,
                    "compute",
                    "scp",
                    "--project",
                    cfg.project,
                    "--zone",
                    cfg.zone,
                    str(archive_path),
                    remote_archive,
                ],
            )
            _run_stream(
                [
                    _GCLOUD,
                    "compute",
                    "ssh",
                    cfg.instance_name,
                    "--project",
                    cfg.project,
                    "--zone",
                    cfg.zone,
                    "--command",
                    f"cd {shlex.quote(remote_dir)} && tar xzf repo.tar.gz && rm repo.tar.gz",
                ],
            )
            logger.info("Repo uploaded.")
        finally:
            archive_path.unlink(missing_ok=True)

    def download_results(self, local_output_dir: Path, remote_output_dir: str = "~/benchmark/results") -> None:
        """Download result JSON files from the instance."""
        cfg = self.config
        local_output_dir.mkdir(parents=True, exist_ok=True)
        dest = str(local_output_dir)

        logger.info("Downloading results from %s:%s...", cfg.instance_name, remote_output_dir)
        _run_stream(
            [
                _GCLOUD,
                "compute",
                "scp",
                "--project",
                cfg.project,
                "--zone",
                cfg.zone,
                "--recurse",
                f"{cfg.instance_name}:{remote_output_dir}/",
                dest,
            ],
        )
        logger.info("Results downloaded to %s.", dest)

    # ------------------------------------------------------------------
    # Remote execution
    # ------------------------------------------------------------------

    def run_remote_benchmark(
        self,
        remote_cli_args: list[str],
        remote_data_dir: str,
        remote_repo_dir: str = "~/benchmark",
    ) -> None:
        """Run benchmark.cli run on the remote instance."""
        cfg = self.config
        cli_args = shlex.join(remote_cli_args)
        cmd_str = (
            f"cd {shlex.quote(remote_repo_dir)} && "
            f"python -m pip install -e . -q && "
            f"python -m benchmark.cli run "
            f"--data-dir {shlex.quote(remote_data_dir)} "
            f"--output {shlex.quote(remote_repo_dir + '/results')} "
            f"{cli_args}"
        )

        logger.info("Running benchmark on %s...", cfg.instance_name)
        rc = _run_stream(
            [
                _GCLOUD,
                "compute",
                "ssh",
                cfg.instance_name,
                "--project",
                cfg.project,
                "--zone",
                cfg.zone,
                "--command",
                cmd_str,
            ],
        )
        if rc != 0:
            raise RuntimeError(f"Remote benchmark failed with exit code {rc}")
        logger.info("Remote benchmark complete.")

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        repo_root: Path,
        remote_cli_args: list[str],
        local_output_dir: Path,
        remote_data_dir: str,
        remote_repo_dir: str = "~/benchmark",
        keep_instance: bool = False,
    ) -> None:
        """Full lifecycle: create → upload → run → download → delete.

        Args:
            repo_root:          Local repo root to upload.
            remote_cli_args:    Extra args passed to `benchmark.cli run` remotely
                                (e.g. ["--media", "video", "--libraries", "kornia"]).
            local_output_dir:   Where to write downloaded results.
            remote_data_dir:    Path to data directory on the remote instance.
            remote_repo_dir:    Where the repo will be placed on the instance.
            keep_instance:      If True, do not delete the instance after the run
                                (useful for debugging).
        """
        self.create_instance()
        try:
            self.wait_for_ssh()
            self.upload_repo(repo_root, remote_repo_dir)
            self.run_remote_benchmark(remote_cli_args, remote_data_dir, remote_repo_dir)
            self.download_results(local_output_dir, f"{remote_repo_dir}/results")
        finally:
            if not keep_instance:
                self.delete_instance()
            else:
                logger.info("Keeping instance alive (--keep-instance). Remember to delete manually.")
