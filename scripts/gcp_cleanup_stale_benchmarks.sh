#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-albumentations}"
GCS_RESULTS_URI="${GCS_RESULTS_URI:-gs://imagenet_validation/augmentation-results}"
DELETE=0
YES=0

usage() {
  cat <<'EOF'
Usage: scripts/gcp_cleanup_stale_benchmarks.sh [--project PROJECT] [--gcs-results-uri URI] [--delete] [--yes]

Lists benchmark VMs whose benchmark-expires label is older than now. With
--delete, asks for confirmation and deletes expired VMs whose run directory has
no DONE or FAILED marker. Use --yes for non-interactive deletion. Dry-run is the
default.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --project)
    PROJECT="$2"
    shift 2
    ;;
  --gcs-results-uri)
    GCS_RESULTS_URI="$2"
    shift 2
    ;;
  --delete)
    DELETE=1
    shift
    ;;
  --yes)
    YES=1
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

json="$(gcloud compute instances list \
  --project "$PROJECT" \
  --filter='labels.benchmark=augmentation AND labels.benchmark-expires:*' \
  --format=json)"

BENCHMARK_INSTANCES_JSON="$json" python3 - "$DELETE" "$PROJECT" "$YES" "$GCS_RESULTS_URI" <<'PY'
import json
import os
import subprocess
import sys
import time

delete = sys.argv[1] == "1"
project = sys.argv[2]
yes = sys.argv[3] == "1"
gcs_results_uri = sys.argv[4].rstrip("/")
now = int(time.time())
instances = json.loads(os.environ["BENCHMARK_INSTANCES_JSON"])
stale = []

for instance in instances:
    labels = instance.get("labels", {})
    expires = labels.get("benchmark-expires")
    run_id = labels.get("benchmark-run-id", "")
    try:
        expires_unix = int(expires)
    except (TypeError, ValueError):
        continue
    if expires_unix <= now:
        zone = instance["zone"].rsplit("/", 1)[-1]
        stale.append((instance["name"], zone, expires_unix, run_id))

if not stale:
    print("No stale benchmark VMs found.")
    raise SystemExit(0)

def _gcs_exists(uri: str) -> bool | None:
    result = subprocess.run(
        ["gcloud", "storage", "ls", "--project", project, uri],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    stderr = result.stderr.lower()
    if "matched no objects" in stderr or "no urls matched" in stderr or "not found" in stderr:
        return False
    return None


def _marker_state(run_id: str) -> str:
    if not run_id:
        return "unknown:no-run-id-label"
    run_prefix = f"{gcs_results_uri}/{run_id}"
    done = _gcs_exists(f"{run_prefix}/DONE")
    failed = _gcs_exists(f"{run_prefix}/FAILED")
    if done is None or failed is None:
        return "unknown:gcs-check-failed"
    if done:
        return "DONE"
    if failed:
        return "FAILED"
    return "missing-terminal-marker"


print("name\tzone\trun_id\tmarker_state\texpired")
delete_targets = []
for name, zone, expires_unix, run_id in stale:
    age_hours = (now - expires_unix) / 3600
    marker_state = _marker_state(run_id)
    print(f"{name}\t{zone}\t{run_id or '-'}\t{marker_state}\t{age_hours:.2f}h ago")
    if marker_state == "missing-terminal-marker":
        delete_targets.append((name, zone))

if not delete:
    print("\nDry run only. Re-run with --delete to remove expired VMs that have no DONE/FAILED marker.")
    raise SystemExit(0)

if not delete_targets:
    print("\nNo expired benchmark VMs without DONE/FAILED markers found.")
    raise SystemExit(0)

if not yes:
    response = input("\nDelete expired benchmark VMs without DONE/FAILED markers? Type 'delete' to continue: ")
    if response.strip() != "delete":
        print("Cancelled.")
        raise SystemExit(0)

for name, zone in delete_targets:
    subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            name,
            "--project",
            project,
            "--zone",
            zone,
            "--quiet",
        ],
        check=True,
    )
PY
