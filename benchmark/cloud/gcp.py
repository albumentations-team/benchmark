r"""GCP cloud execution: synchronous (SSH) and detached (startup-script + GCS) benchmark runs.

Detached flow (default for ``--cloud gcp``):

    1. Build a repo tarball and upload it with ``job.json`` + ``bootstrap.sh`` to GCS under a unique run prefix.
    2. Create a Compute Engine VM whose startup script downloads ``bootstrap.sh`` and runs it.
    3. On the VM: download the dataset archive from GCS to local disk, extract it, install Python 3.13 with ``uv``,
       run ``benchmark.cli run``
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
import os
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
export HOME="${HOME:-/root}"
export PATH="/snap/bin:${HOME}/.local/bin:${PATH}"
RUN_LOG=/var/log/benchmark-gcp-run.log
touch "$RUN_LOG"
echo "benchmark bootstrap logging to ${RUN_LOG}"
exec >>"$RUN_LOG" 2>&1
echo "=== benchmark bootstrap start $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="

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

self_delete_vm() {
  MD=http://metadata.google.internal/computeMetadata/v1
  local project zone_raw zone name token_json token
  project=$(curl -s -f -H "Metadata-Flavor: Google" "$MD/project/project-id") || return 1
  zone_raw=$(curl -s -f -H "Metadata-Flavor: Google" "$MD/instance/zone") || return 1
  zone=${zone_raw##*/}
  name=$(curl -s -f -H "Metadata-Flavor: Google" "$MD/instance/name") || return 1
  token_json=$(curl -s -f -H "Metadata-Flavor: Google" \
    "$MD/instance/service-accounts/default/token?scopes=https://www.googleapis.com/auth/cloud-platform") || return 1
  token=$(python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])" <<<"$token_json") || return 1
  curl -s -S -f -X DELETE \
    -H "Authorization: Bearer $token" \
    "https://compute.googleapis.com/compute/v1/projects/${project}/zones/${zone}/instances/${name}" \
    || echo "WARN: instance self-delete failed; delete manually or fix IAM (compute.instances.delete)."
}

should_terminate() {
  if [[ ! -f "$WORKDIR/job.json" ]]; then
    return 0
  fi
  [[ "$(python3 -c 'import json; print(json.load(open("'"$WORKDIR"'/job.json"))["terminate_instance"])')" == "True" ]]
}

upload_terminal_artifacts() {
  local rc="$1"
  local status="$2"
  local run_prefix="$PREFIX"
  local terminal_ok=1
  if [[ -f "$WORKDIR/job.json" ]]; then
    run_prefix=$(
      python3 -c 'import json; print(json.load(open("'"$WORKDIR"'/job.json"))["gcs_results_prefix"].rstrip("/"))'
    )
  fi
  mkdir -p "$WORKDIR"
  printf '%s\n' "$rc" > "$WORKDIR/exit_code.txt"

  if [[ -n "${RESULT_SYNC_PID:-}" ]]; then
    kill "$RESULT_SYNC_PID" 2>/dev/null || true
    RESULT_SYNC_PID=""
  fi

  terminal_log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] terminal-upload: $*" >&2
  }

  gcs_cp_retry() {
    local label="$1"
    local timeout_secs="$2"
    local src="$3"
    local dest="$4"
    local attempt rc
    for attempt in 1 2 3 4 5; do
      terminal_log "upload ${label} attempt ${attempt}/5: timeout ${timeout_secs}s to ${dest}"
      set +e
      timeout "${timeout_secs}s" gcloud --quiet storage cp "$src" "$dest"
      rc=$?
      set -e
      if [[ "$rc" == "0" ]]; then
        terminal_log "upload ${label} succeeded"
        return 0
      fi
      terminal_log "WARN: upload ${label} failed with rc=${rc}"
      sleep 3
    done
    terminal_log "ERROR: upload ${label} failed after retries: timeout ${timeout_secs}s to ${dest}"
    return 1
  }

  gcs_describe_retry() {
    local label="$1"
    local uri="$2"
    local attempt rc
    for attempt in 1 2 3 4 5; do
      terminal_log "confirm ${label} attempt ${attempt}/5: timeout 30s gcloud --quiet storage objects describe ${uri}"
      set +e
      timeout 30s gcloud --quiet storage objects describe "$uri" >/dev/null 2>&1
      rc=$?
      set -e
      if [[ "$rc" == "0" ]]; then
        terminal_log "confirm ${label} succeeded"
        return 0
      fi
      terminal_log "WARN: confirm ${label} failed with rc=${rc}"
      sleep 3
    done
    terminal_log "ERROR: confirm ${label} failed after retries: timeout 30s for ${uri}"
    return 1
  }

  BENCHMARK_TERMINAL_STATUS="$status" BENCHMARK_TERMINAL_RC="$rc" python3 << 'PY' || true
import json
import os
import time
from pathlib import Path

work = Path("/root/benchmark-work")
job_path = work / "job.json"
meta = {
    "exit_code": int(os.environ["BENCHMARK_TERMINAL_RC"]),
    "status": os.environ["BENCHMARK_TERMINAL_STATUS"],
    "end_timestamp_unix": time.time(),
}
if job_path.exists():
    job = json.loads(job_path.read_text(encoding="utf-8"))
    meta.update(job.get("instance", {}))
    meta["submission"] = job.get("submission", {})
(work / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
PY

  gcs_cp_retry "exit_code.txt" 60 "$WORKDIR/exit_code.txt" "${run_prefix}/exit_code.txt" || terminal_ok=0
  gcs_cp_retry "run_meta.json" 60 "$WORKDIR/run_meta.json" "${run_prefix}/run_meta.json" || terminal_ok=0

  local marker_name marker_path
  if [[ "$status" == "success" ]]; then
    marker_name="DONE"
    marker_path="$WORKDIR/DONE"
    date -u > "$marker_path"
  else
    marker_name="FAILED"
    marker_path="$WORKDIR/FAILED"
    printf 'failed exit %s\n%s\n' "$rc" "$(date -u)" > "$marker_path"
  fi

  gcs_cp_retry "$marker_name" 60 "$marker_path" "${run_prefix}/${marker_name}" || terminal_ok=0
  gcs_describe_retry "$marker_name" "${run_prefix}/${marker_name}" || terminal_ok=0
  if [[ "$terminal_ok" != "1" ]]; then
    terminal_log "ERROR: terminal marker upload was not confirmed for ${run_prefix}; keeping VM for triage."
    return 1
  fi

  if [[ -d "$WORKDIR/results" ]]; then
    terminal_log "sync results to ${run_prefix}/results/"
    timeout 300s gcloud --quiet storage rsync --recursive "$WORKDIR/results" "${run_prefix}/results/" || \
      terminal_log "WARN: result rsync failed after terminal marker confirmation"
  fi
  gcs_cp_retry "vm.log" 60 "$RUN_LOG" "${run_prefix}/vm.log" || true
  return 0
}

on_error() {
  local rc=$?
  local line="$1"
  echo "ERROR: benchmark bootstrap failed at line ${line} (exit ${rc})" >&2
  terminal_uploaded=0
  upload_terminal_artifacts "$rc" "failed" && terminal_uploaded=1
  keep_on_failure=$(
    python3 -c 'import json; print(json.load(open("'"$WORKDIR"'/job.json")).get("keep_instance_on_failure", False))' \
      2>/dev/null || echo "False"
  )
  if should_terminate && [[ "$keep_on_failure" != "True" && "$terminal_uploaded" == "1" ]]; then
    self_delete_vm || true
  fi
  exit "$rc"
}
trap 'on_error $LINENO' ERR

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud is required on the VM image; Ubuntu GCE images should provide google-cloud-cli snap." >&2
  exit 1
fi

(
  while true; do
    sleep 15
    timeout 60s gcloud --quiet storage cp "$RUN_LOG" "${PREFIX}/vm.log" >/dev/null 2>&1 || true
  done
) &
LOG_SYNC_PID=$!
RESULT_SYNC_PID=""
cleanup_background_syncs() {
  kill "$LOG_SYNC_PID" "${RESULT_SYNC_PID:-}" 2>/dev/null || true
}
trap cleanup_background_syncs EXIT

mkdir -p "$WORKDIR" "$DATADIR"
cd "$WORKDIR"

gcloud --quiet storage cp "${PREFIX}/job.json" ./job.json
gcloud --quiet storage cp "${PREFIX}/repo.tar.gz" ./repo.tar.gz
RUN_PREFIX=$(python3 -c 'import json; print(json.load(open("job.json"))["gcs_results_prefix"].rstrip("/"))')
mkdir -p "$WORKDIR/results"
(
  while true; do
    sleep 30
    if [[ -d "$WORKDIR/results" ]]; then
      timeout 300s gcloud --quiet storage rsync --recursive \
        "$WORKDIR/results" "${RUN_PREFIX}/results/" >/dev/null 2>&1 || true
    fi
  done
) &
RESULT_SYNC_PID=$!

mkdir -p "$REPODIR"
tar xzf repo.tar.gz -C "$REPODIR"
rm -f repo.tar.gz

compute_venv_cache_path() {
  python3 << 'PY'
import hashlib
import json
import platform
from pathlib import Path

work = Path("/root/benchmark-work")
repo = work / "repo"
job = json.loads((work / "job.json").read_text(encoding="utf-8"))
cache_uri = str(job.get("venv_cache_uri", "")).rstrip("/")
if not cache_uri:
    raise SystemExit(0)

digest = hashlib.sha256()
digest.update(b"grouping-schema=v1\n")
digest.update(f"os={platform.system()}\narch={platform.machine()}\npython=3.13\n".encode())
for path in sorted((repo / "requirements").glob("*.txt")):
    digest.update(f"path={path.name}\n".encode())
    digest.update(path.read_bytes())
    digest.update(b"\n")
print(f"{cache_uri}/venvs-{platform.system()}-{platform.machine()}-{digest.hexdigest()[:16]}.tar.gz")
PY
}

VENV_CACHE_PATH="$(compute_venv_cache_path || true)"
FORCE_VENV_CACHE_REBUILD=$(
  python3 -c 'import json; print(json.load(open("job.json")).get("force_venv_cache_rebuild", False))'
)
VENV_CACHE_HIT=0
if [[ -n "$VENV_CACHE_PATH" ]]; then
  echo "Venv cache path: ${VENV_CACHE_PATH}"
  if [[ "$FORCE_VENV_CACHE_REBUILD" == "True" ]]; then
    echo "Venv cache lookup skipped: force rebuild requested."
  elif timeout 30s gcloud --quiet storage objects describe "$VENV_CACHE_PATH" >/dev/null 2>&1; then
    echo "Venv cache hit; restoring..."
    if timeout 300s gcloud --quiet storage cp "$VENV_CACHE_PATH" "$WORKDIR/venvs.tar.gz"; then
      tar xzf "$WORKDIR/venvs.tar.gz" -C "$WORKDIR"
      rm -f "$WORKDIR/venvs.tar.gz"
      VENV_CACHE_HIT=1
      echo "Venv cache restored."
    else
      echo "WARN: failed to download venv cache; rebuilding locally." >&2
    fi
  else
    echo "Venv cache miss; will populate after successful run."
  fi
else
  echo "Venv cache disabled."
fi

STAGE_LIMIT=$(python3 << 'PY'
import json

j = json.load(open("job.json"))
args = j["benchmark_cli_args"]
gcs_data_uri: str = j["gcs_data_uri"]


def value(flag: str) -> str:
    try:
        return str(args[args.index(flag) + 1])
    except (ValueError, IndexError):
        return ""


mode = value("--mode")
num_items = value("--num-items")
is_tar = gcs_data_uri.lower().endswith((".tar", ".tar.gz", ".tgz"))
if mode == "micro" and not is_tar:
    raise SystemExit(
        "For --mode micro on GCP, --gcp-gcs-data-uri must point to a tarball (e.g. gs://.../imagenet/val.tar). "
        f"Got: {gcs_data_uri!r}"
    )
if mode == "micro":
    print(num_items or "2000")
else:
    print("0")
PY
)
TAR_PATH="$WORKDIR/imagenet-val.tar"
TAR_URI=$(python3 -c 'import json; print(json.load(open("job.json"))["gcs_data_uri"])')
mkdir -p "$DATADIR"
if [[ "$TAR_URI" == *.tar || "$TAR_URI" == *.tar.gz || "$TAR_URI" == *.tgz ]]; then
  echo "Staging dataset tarball: ${TAR_URI}"
  gcloud storage cp "$TAR_URI" "$TAR_PATH"
  export BENCHMARK_TAR_PATH="$TAR_PATH"
  export BENCHMARK_STAGING_DIR="$DATADIR"
  export BENCHMARK_TAR_EXTRACT_LIMIT="$STAGE_LIMIT"
  python3 << 'PY'
import os
import tarfile
from pathlib import Path

data_dir = Path(os.environ["BENCHMARK_STAGING_DIR"])
limit_raw = os.environ.get("BENCHMARK_TAR_EXTRACT_LIMIT", "0")
limit = int(limit_raw) if limit_raw.isdigit() else 0
tar_path = Path(os.environ["BENCHMARK_TAR_PATH"])

# ImageNet val tarball layout (standard): val/<name>.JPEG


def is_image_member(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("val/") and (lower.endswith(".jpeg") or lower.endswith(".jpg") or lower.endswith(".png"))


with tarfile.open(tar_path, mode="r:*") as tf:
    if limit == 0:
        tf.extractall(path=data_dir, filter="data")
    else:
        members = [m for m in tf.getmembers() if m.isfile() and is_image_member(m.name)]
        members.sort(key=lambda m: m.name)
        members = members[:limit]
        tf.extractall(path=data_dir, members=members, filter="data")
PY
  rm -f "$TAR_PATH"
else
  echo "Staging dataset directory: ${TAR_URI}"
  gcloud --quiet storage rsync --recursive "$TAR_URI" "$DATADIR"
fi

cd "$REPODIR"
echo "Installing uv..."
curl -Ls https://astral.sh/uv/install.sh | sh
uv --version
uv python install 3.13
CONTROL_VENV="$WORKDIR/control-venv"
CONTROL_PYTHON="$CONTROL_VENV/bin/python"
if [[ ! -x "$CONTROL_PYTHON" ]]; then
  uv venv "$CONTROL_VENV" --python 3.13 --seed
fi
UV_LINK_MODE=copy uv pip install --python "$CONTROL_PYTHON" -q -r requirements/requirements.txt -e .
export CONTROL_PYTHON
cd "$WORKDIR"

set +e
python3 << 'PY'
import json
import os
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
    [
        os.environ["CONTROL_PYTHON"],
        "-m",
        "benchmark.cli",
        "run",
        *j["benchmark_cli_args"],
    ],
    cwd=repo,
)
(work / "benchmark_exit_code.txt").write_text(str(rc))
sys.exit(0)
PY
set -e

RC=$(cat "$WORKDIR/benchmark_exit_code.txt")
terminal_uploaded=0
if [[ "$RC" == "0" ]]; then
  upload_terminal_artifacts "$RC" "success" && terminal_uploaded=1
  if [[ -n "${VENV_CACHE_PATH:-}" && "$VENV_CACHE_HIT" != "1" ]]; then
    echo "Populating venv cache at ${VENV_CACHE_PATH}..."
    rm -f "$WORKDIR/venvs.tar.gz"
    cache_inputs=()
    [[ -d "$WORKDIR/control-venv" ]] && cache_inputs+=("control-venv")
    while IFS= read -r path; do
      cache_inputs+=("${path#"$WORKDIR"/}")
    done < <(find "$REPODIR" -maxdepth 1 -type d -name '.venv_*' | sort)
    if (( ${#cache_inputs[@]} > 0 )); then
      (
        cd "$WORKDIR"
        GZIP=-1 timeout 300s tar czf "$WORKDIR/venvs.tar.gz" "${cache_inputs[@]}"
      ) && timeout 600s gcloud --quiet storage cp "$WORKDIR/venvs.tar.gz" "$VENV_CACHE_PATH" || \
        echo "WARN: failed to populate venv cache at ${VENV_CACHE_PATH}" >&2
      rm -f "$WORKDIR/venvs.tar.gz"
    else
      echo "WARN: no venvs found to cache." >&2
    fi
  fi
else
  upload_terminal_artifacts "$RC" "failed" && terminal_uploaded=1
fi

terminate=$(python3 -c 'import json; print(json.load(open("job.json"))["terminate_instance"])')
keep_on_failure=$(python3 -c 'import json; print(json.load(open("job.json")).get("keep_instance_on_failure", False))')
if [[ "$terminate" == "True" && "$terminal_uploaded" == "1" && ( "$RC" == "0" || "$keep_on_failure" != "True" ) ]]; then
  self_delete_vm || true
fi
echo "=== benchmark bootstrap end $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
exit "$RC"
"""
)

_STARTUP_INLINE = r"""#!/bin/bash
set -euo pipefail
export PATH="/snap/bin:${PATH}"
PREFIX=$(curl -s -f -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/benchmark-run-prefix")
TMP=/tmp/benchmark-gcp-bootstrap.sh
if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud is required to fetch bootstrap.sh; expected it from the GCE image google-cloud-cli snap."
  exit 1
fi
gcloud --quiet storage cp "${PREFIX}/bootstrap.sh" "$TMP"
chmod +x "$TMP"
exec "$TMP"
"""


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    logger.debug("$ %s", shlex.join(cmd))
    kwargs.setdefault("timeout", int(os.environ.get("BENCHMARK_GCLOUD_TIMEOUT_SECS", "600")))
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)
    except subprocess.CalledProcessError as e:
        logger.exception(
            "Command failed: %s\nstdout:\n%s\nstderr:\n%s",
            shlex.join(cmd),
            e.stdout or "",
            e.stderr or "",
        )
        raise
    except subprocess.TimeoutExpired:
        logger.exception("Command timed out after %ss: %s", kwargs["timeout"], shlex.join(cmd))
        raise


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
    _run([_GCLOUD, "storage", "cp", str(local_path), dest_uri], timeout=300)


def _make_repo_tarball(repo_root: Path) -> Path:
    """Create a gzipped tar of the repo in a temp file; caller must unlink when done."""
    fd, tmp = tempfile.mkstemp(suffix=".tar.gz", prefix="benchmark-repo-")
    os.close(fd)
    archive_path = Path(tmp)
    exclude_flags = [flag for pattern in _REPO_EXCLUDE_PATTERNS for flag in ("--exclude", pattern)]
    env = {**os.environ, "COPYFILE_DISABLE": "1"}
    subprocess.run(
        ["tar", "--no-xattrs", "-czf", str(archive_path), *exclude_flags, "-C", str(repo_root), "."],
        check=True,
        env=env,
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
        _run(cmd, timeout=900)
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
                timeout=30,
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
            env = {**os.environ, "COPYFILE_DISABLE": "1"}
            subprocess.run(
                ["tar", "--no-xattrs", "-czf", str(archive_path), *exclude_flags, "-C", str(repo_root), "."],
                check=True,
                env=env,
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
    keep_instance_on_failure: bool,
    venv_cache_uri: str,
    force_venv_cache_rebuild: bool,
    submission: dict[str, Any],
    instance_meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the ``job.json`` payload uploaded to GCS for detached runs."""
    return {
        "run_id": run_id,
        "gcs_data_uri": _validate_gs_uri(gcs_data_uri, kind="--gcp-gcs-data-uri"),
        "gcs_results_prefix": "",
        "terminate_instance": terminate_instance,
        "keep_instance_on_failure": keep_instance_on_failure,
        "venv_cache_uri": _validate_gs_uri(venv_cache_uri, kind="--gcp-venv-cache-uri") if venv_cache_uri else "",
        "force_venv_cache_rebuild": force_venv_cache_rebuild,
        "benchmark_cli_args": benchmark_cli_args,
        "submission": submission,
        "instance": instance_meta,
    }


def new_run_id() -> str:
    """Return a new unique run id (hex, no hyphens)."""
    return uuid.uuid4().hex
