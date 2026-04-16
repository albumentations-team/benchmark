r"""GCP cloud execution: synchronous (SSH) and detached (startup-script + GCS) benchmark runs.

Detached flow (default for ``--cloud gcp``):

    1. Build a repo tarball and upload it with ``job.json`` + ``bootstrap.sh`` to GCS under a unique run prefix.
    2. Create a Compute Engine VM whose startup script downloads ``bootstrap.sh`` and runs it.
    3. On the VM: rsync dataset from GCS to local disk, ``pip install -e .``, run ``benchmark.cli run``
       (no cloud flags).
    4. Upload results and logs to GCS, then delete the VM via the Compute API (unless
       ``terminate_instance`` is false).

The local process exits after the instance is created; the laptop can go offline while the VM finishes.

Synchronous flow (``--gcp-attached``): create VM → SSH → upload repo → run → download results →
delete VM (client ``finally`` still deletes the instance if the client stays up).

Usage (detached):

    python -m benchmark.cli run \\
        --cloud gcp \\
        --gcp-project my-project \\
        --gcp-gcs-data-uri gs://my-bucket/datasets/video-50 \\
        --gcp-gcs-results-uri gs://my-bucket/benchmark-runs \\
        --data-dir /ignored-locally \\
        --output ./ignored-locally \\
        --media video

Artifacts for run ``<run_id>`` live under ``<gcp-gcs-results-uri>/<run_id>/``:

    - ``job.json`` — run definition and benchmark CLI argv
    - ``repo.tar.gz`` — repository snapshot
    - ``bootstrap.sh`` — VM-side driver script
    - ``results/`` — JSON outputs from ``benchmark.cli run``
    - ``vm.log`` — captured bootstrap + benchmark stdout/stderr
    - ``exit_code.txt`` — integer exit code of ``benchmark.cli run``
    - ``run_meta.json`` — exit code + instance metadata + end timestamp
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .instance import GCPInstanceConfig

logger = logging.getLogger(__name__)

_GCLOUD = "gcloud"

# VM paths (must match bootstrap.sh)
_VM_WORKDIR = "/root/benchmark-work"
_VM_REPODIR = f"{_VM_WORKDIR}/repo"
_VM_DATADIR = "/root/benchmark-data"
_VM_RESULTS = f"{_VM_WORKDIR}/results"

_REPO_EXCLUDE_PATTERNS = [
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

# Bootstrap: fetch job.json + repo, stage data from GCS, run benchmark, upload artifacts, optional self-delete.
_BOOTSTRAP_SH = (
    r"""#!/bin/bash
set -euo pipefail
RUN_LOG=/var/log/benchmark-gcp-run.log
exec > >(tee -a "$RUN_LOG") 2>&1
echo "=== benchmark bootstrap start $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="

if ! command -v gsutil >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq google-cloud-cli ca-certificates curl
fi

PREFIX=$(curl -s -f -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/benchmark-run-prefix")

WORKDIR="""
    + _VM_WORKDIR
    + r"""
REPODIR="""
    + _VM_REPODIR
    + r"""
DATADIR="""
    + _VM_DATADIR
    + r"""
mkdir -p "$WORKDIR" "$DATADIR"
cd "$WORKDIR"

gsutil cp "${PREFIX}/job.json" ./job.json
gsutil cp "${PREFIX}/repo.tar.gz" ./repo.tar.gz

mkdir -p "$REPODIR"
tar xzf repo.tar.gz -C "$REPODIR"
rm -f repo.tar.gz

DATA_URI=$(python3 -c 'import json; print(json.load(open("job.json"))["gcs_data_uri"])')
gsutil -m rsync -r "$DATA_URI" "$DATADIR"

cd "$REPODIR"
python3 -c 'import sys; assert sys.version_info >= (3, 13), "Need Python 3.13+ on VM: " + sys.version'
python3 -m pip install -q pip setuptools wheel uv
python3 -m pip install -q -e .
cd "$WORKDIR"

set +e
python3 << 'PY'
import json
import subprocess
import sys
from pathlib import Path

work = Path("""
    + repr(_VM_WORKDIR)
    + r""")
repo = Path("""
    + repr(_VM_REPODIR)
    + r""")
j = json.loads((work / "job.json").read_text())
rc = subprocess.call(
    [sys.executable, "-m", "benchmark.cli", "run", *j["benchmark_cli_args"]],
    cwd=repo,
)
(work / "benchmark_exit_code.txt").write_text(str(rc))
sys.exit(0)
PY
set -e

RUN_PREFIX=$(python3 -c 'import json; print(json.load(open("job.json"))["gcs_results_prefix"].rstrip("/"))')
mkdir -p """
    + _VM_RESULTS
    + r"""

gsutil -m rsync -r """
    + _VM_RESULTS
    + r""" "${RUN_PREFIX}/results/" || true
gsutil cp "$RUN_LOG" "${RUN_PREFIX}/vm.log" || true

python3 << 'PY'
import json
import subprocess
import time
from pathlib import Path

work = Path("""
    + repr(_VM_WORKDIR)
    + r""")
j = json.loads((work / "job.json").read_text())
prefix = j["gcs_results_prefix"].rstrip("/")
rc = int((work / "benchmark_exit_code.txt").read_text().strip())
(work / "exit_code.txt").write_text(str(rc))
subprocess.run(["gsutil", "cp", str(work / "exit_code.txt"), f"{prefix}/exit_code.txt"], check=False)
meta = {
    "exit_code": rc,
    "end_timestamp_unix": time.time(),
    **j.get("instance", {}),
    "submission": j.get("submission", {}),
}
(work / "run_meta.json").write_text(json.dumps(meta, indent=2))
subprocess.run(["gsutil", "cp", str(work / "run_meta.json"), f"{prefix}/run_meta.json"], check=False)
PY

terminate=$(python3 -c 'import json; print(json.load(open("job.json"))["terminate_instance"])')
if [[ "$terminate" == "True" ]]; then
  MD=http://metadata.google.internal/computeMetadata/v1
  PROJECT=$(curl -s -f -H "Metadata-Flavor: Google" "$MD/project/project-id")
  ZONE_RAW=$(curl -s -f -H "Metadata-Flavor: Google" "$MD/instance/zone")
  ZONE=${ZONE_RAW##*/}
  NAME=$(curl -s -f -H "Metadata-Flavor: Google" "$MD/instance/name")
  TOKEN_JSON=$(curl -s -f -H "Metadata-Flavor: Google" \
    "$MD/instance/service-accounts/default/token?scopes=https://www.googleapis.com/auth/cloud-platform")
  TOKEN=$(python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])" <<<"$TOKEN_JSON")
  curl -s -S -f -X DELETE \
    -H "Authorization: Bearer $TOKEN" \
    "https://compute.googleapis.com/compute/v1/projects/${PROJECT}/zones/${ZONE}/instances/${NAME}" \
    || echo "WARN: instance self-delete failed; delete manually or fix IAM (compute.instances.delete)."
fi
echo "=== benchmark bootstrap end $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
"""
)

_STARTUP_INLINE = r"""#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || true
apt-get install -y -qq ca-certificates curl python3 google-cloud-cli || true
PREFIX=$(curl -s -f -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/benchmark-run-prefix")
TMP=/tmp/benchmark-gcp-bootstrap.sh
gsutil cp "${PREFIX}/bootstrap.sh" "$TMP"
chmod +x "$TMP"
exec "$TMP"
"""


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    logger.debug("$ %s", shlex.join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def _run_stream(cmd: list[str], **kwargs: Any) -> int:
    """Run command streaming stdout/stderr to the terminal."""
    logger.debug("$ %s", shlex.join(cmd))
    result = subprocess.run(cmd, text=True, check=False, **kwargs)
    return result.returncode


def _validate_gs_uri(uri: str, *, kind: str) -> str:
    u = uri.strip()
    if not u.startswith("gs://"):
        msg = f"{kind} must be a gs:// URI, got {uri!r}"
        raise ValueError(msg)
    return u.rstrip("/")


def _gcs_cp(local_path: Path, dest_uri: str) -> None:
    """Upload a file to GCS using ``gcloud storage cp``."""
    cmd = [_GCLOUD, "storage", "cp", str(local_path), dest_uri]
    _run(cmd)


def _make_repo_tarball(repo_root: Path) -> Path:
    """Create a gzipped tar of the repo in a temp file; caller must unlink when done."""
    archive_path = Path(tempfile.mkstemp(suffix=".tar.gz", prefix="benchmark-repo-")[1])
    exclude_flags = [flag for pattern in _REPO_EXCLUDE_PATTERNS for flag in ("--exclude", pattern)]
    subprocess.run(
        ["tar", "czf", str(archive_path), *exclude_flags, "-C", str(repo_root), "."],
        check=True,
    )
    return archive_path


class GCPRunner:
    """Manages the lifecycle of a GCP benchmark instance."""

    def __init__(self, config: GCPInstanceConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    def create_instance(
        self,
        *,
        metadata: dict[str, str] | None = None,
        startup_script_path: Path | None = None,
    ) -> None:
        """Create the GCP Compute Engine instance."""
        cfg = self.config
        cmd: list[str] = [
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

        if startup_script_path is not None:
            cmd += ["--metadata-from-file", f"startup-script={startup_script_path}"]

        if metadata:
            meta_arg = ",".join(f"{key}={value}" for key, value in metadata.items())
            cmd += ["--metadata", meta_arg]

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
        msg = f"SSH not available on {cfg.instance_name} after {timeout}s"
        raise TimeoutError(msg)

    # ------------------------------------------------------------------
    # File transfer (attached mode)
    # ------------------------------------------------------------------

    def upload_repo(self, repo_root: Path, remote_dir: str = "~/benchmark") -> None:
        """Upload the repo to the remote instance via a tar archive, excluding venvs and outputs."""
        cfg = self.config
        archive_path = repo_root / ".upload_repo.tar.gz"

        exclude_flags = [flag for pattern in _REPO_EXCLUDE_PATTERNS for flag in ("--exclude", pattern)]

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
    # Remote execution (attached mode)
    # ------------------------------------------------------------------

    def run_remote_benchmark(
        self,
        remote_cli_args: list[str],
        remote_repo_dir: str = "~/benchmark",
    ) -> None:
        """Run ``benchmark.cli run`` on the remote instance; argv is everything after ``run``."""
        cfg = self.config
        cli_args = shlex.join(remote_cli_args)
        cmd_str = (
            f"cd {shlex.quote(remote_repo_dir)} && "
            f"python3 -m pip install -q pip setuptools wheel uv && "
            f"python3 -m pip install -q -e . && "
            f"python3 -m benchmark.cli run {cli_args}"
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
            msg = f"Remote benchmark failed with exit code {rc}"
            raise RuntimeError(msg)
        logger.info("Remote benchmark complete.")

    # ------------------------------------------------------------------
    # Detached run
    # ------------------------------------------------------------------

    def run_detached(
        self,
        *,
        repo_root: Path,
        gcs_data_uri: str,
        gcs_results_base_uri: str,
        job: dict[str, Any],
        dry_run: bool = False,
    ) -> str:
        """Upload artifacts to GCS, create VM with startup script, return the run prefix (gs://.../run_id)."""
        _validate_gs_uri(gcs_data_uri, kind="--gcp-gcs-data-uri")
        run_id = job["run_id"]
        run_prefix = f"{_validate_gs_uri(gcs_results_base_uri, kind='--gcp-gcs-results-uri')}/{run_id}"
        job["gcs_results_prefix"] = run_prefix

        job_uri = f"{run_prefix}/job.json"
        repo_uri = f"{run_prefix}/repo.tar.gz"
        bootstrap_uri = f"{run_prefix}/bootstrap.sh"

        if dry_run:
            logger.info("Dry run: would upload to %s", run_prefix)
            logger.info("job.json:\n%s", json.dumps(job, indent=2))
            return run_prefix

        tar_path = _make_repo_tarball(repo_root)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix="-bootstrap.sh", delete=False, encoding="utf-8") as f:
                f.write(_BOOTSTRAP_SH)
                bootstrap_path = Path(f.name)

            with tempfile.NamedTemporaryFile(mode="w", suffix="-job.json", delete=False, encoding="utf-8") as f:
                json.dump(job, f, indent=2)
                job_path = Path(f.name)

            try:
                _gcs_cp(job_path, job_uri)
                _gcs_cp(tar_path, repo_uri)
                subprocess.run(["chmod", "+x", str(bootstrap_path)], check=False)
                _gcs_cp(bootstrap_path, bootstrap_uri)
            finally:
                job_path.unlink(missing_ok=True)
                bootstrap_path.unlink(missing_ok=True)
        finally:
            tar_path.unlink(missing_ok=True)

        metadata = {"benchmark-run-prefix": run_prefix}
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix="-startup-inline.sh",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(_STARTUP_INLINE)
                startup_inline_path = Path(f.name)
            try:
                self.create_instance(metadata=metadata, startup_script_path=startup_inline_path)
            finally:
                startup_inline_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to create instance; artifacts are under %s", run_prefix)
            raise

        return run_prefix

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def run_attached(
        self,
        repo_root: Path,
        remote_cli_args: list[str],
        local_output_dir: Path,
        remote_repo_dir: str = "~/benchmark",
        keep_instance: bool = False,
    ) -> None:
        """Create VM → upload repo → run benchmark over SSH → download results → delete VM."""
        self.create_instance()
        try:
            self.wait_for_ssh()
            self.upload_repo(repo_root, remote_repo_dir)
            self.run_remote_benchmark(remote_cli_args, remote_repo_dir)
            self.download_results(local_output_dir, f"{remote_repo_dir}/results")
        finally:
            if not keep_instance:
                self.delete_instance()
            else:
                logger.info("Keeping instance alive (--gcp-keep-instance). Remember to delete manually.")

    def run(
        self,
        repo_root: Path,
        remote_cli_args: list[str],
        local_output_dir: Path,
        remote_data_dir: str,
        remote_repo_dir: str = "~/benchmark",
        keep_instance: bool = False,
    ) -> None:
        """Legacy attached entrypoint: injects ``--data-dir`` and ``--output`` into the remote command."""
        out = f"{remote_repo_dir}/results"
        full_args = [
            "--data-dir",
            remote_data_dir,
            "--output",
            out,
            *remote_cli_args,
        ]
        self.run_attached(
            repo_root=repo_root,
            remote_cli_args=full_args,
            local_output_dir=local_output_dir,
            remote_repo_dir=remote_repo_dir,
            keep_instance=keep_instance,
        )


def build_gcp_job_dict(
    *,
    run_id: str,
    gcs_data_uri: str,
    benchmark_cli_args: list[str],
    terminate_instance: bool,
    submission: dict[str, Any],
    instance_meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the ``job.json`` payload uploaded to GCS for detached runs."""
    return {
        "run_id": run_id,
        "gcs_data_uri": _validate_gs_uri(gcs_data_uri, kind="--gcp-gcs-data-uri"),
        "gcs_results_prefix": "",
        "terminate_instance": terminate_instance,
        "benchmark_cli_args": benchmark_cli_args,
        "submission": submission,
        "instance": instance_meta,
    }


def new_run_id() -> str:
    """Return a new unique run id (hex, no hyphens)."""
    return uuid.uuid4().hex
