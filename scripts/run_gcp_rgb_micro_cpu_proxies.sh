#!/usr/bin/env bash
set -euo pipefail

cd /Users/vladimiriglovikov/workspace/benchmark

PROJECT=albumentations
ZONE="${ZONE:-us-central1-b}"
# ImageNet val tarball in GCS (no per-image prefix needed).
IMAGENET_VAL_TAR_GCS="${IMAGENET_VAL_TAR_GCS:-gs://imagenet_validation/imagenet/val.tar}"
RESULTS_GCS=gs://imagenet_validation/augmentation-results
MACHINE_TYPES="${MACHINE_TYPES:-n2-standard-16}"
NUM_ITEMS="${NUM_ITEMS:-2000}"
NUM_RUNS="${NUM_RUNS:-5}"
TRANSFORMS="${TRANSFORMS:-}"
KEEP_ON_FAILURE="${KEEP_ON_FAILURE:-0}"
SLOW_THRESHOLD_SEC_PER_ITEM="${SLOW_THRESHOLD_SEC_PER_ITEM:-}"
SLOW_PREFLIGHT_ITEMS="${SLOW_PREFLIGHT_ITEMS:-}"
DISABLE_SLOW_SKIP="${DISABLE_SLOW_SKIP:-0}"
GCP_VENV_CACHE_URI="${GCP_VENV_CACHE_URI:-}"
GCP_NO_VENV_CACHE="${GCP_NO_VENV_CACHE:-0}"
GCP_FORCE_VENV_CACHE_REBUILD="${GCP_FORCE_VENV_CACHE_REBUILD:-0}"
# Default: do not launch overlapping benchmark VMs (important on small global CPU quotas).
WAIT_FOR_GCP_COMPLETION="${WAIT_FOR_GCP_COMPLETION:-1}"
POLL_SECS="${POLL_SECS:-30}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-$((24 * 3600))}"

run_with_timeout() {
  local timeout_secs="$1"
  shift
  python - "$timeout_secs" "$@" <<'PY'
import subprocess
import sys

timeout = int(sys.argv[1])
cmd = sys.argv[2:]
try:
    raise SystemExit(subprocess.run(cmd, check=False, text=True, timeout=timeout).returncode)
except subprocess.TimeoutExpired:
    raise SystemExit(124)
PY
}

gcs_terminal_marker_exists() {
  local prefix="$1"
  run_with_timeout 10 gcloud storage objects describe "$prefix/DONE" >/dev/null 2>&1 && return 0
  run_with_timeout 10 gcloud storage objects describe "$prefix/FAILED" >/dev/null 2>&1 && return 0
  run_with_timeout 10 gcloud storage objects describe "$prefix/exit_code.txt" >/dev/null 2>&1 && return 0
  return 1
}

probe_vm_status() {
  local instance="$1"
  local project="$2"
  local zone="$3"
  local vm_status vm_rc
  set +e
  vm_status="$(run_with_timeout 20 gcloud compute instances describe "$instance" --project "$project" --zone "$zone" --format='value(status)' 2>/dev/null)"
  vm_rc=$?
  set -e
  if [[ "$vm_rc" == "0" && -n "$vm_status" ]]; then
    printf '%s\n' "$vm_status"
    return 0
  fi
  printf 'UNKNOWN\n'
  return 0
}

vm_absence_confirmed() {
  local instance="$1"
  local project="$2"
  local zone="$3"
  local prefix="$4"
  local attempt vm_status vm_rc listed list_rc

  for attempt in 1 2 3; do
    if gcs_terminal_marker_exists "$prefix"; then
      return 1
    fi

    set +e
    vm_status="$(run_with_timeout 20 gcloud compute instances describe "$instance" --project "$project" --zone "$zone" --format='value(status)' 2>/dev/null)"
    vm_rc=$?
    listed="$(run_with_timeout 20 gcloud compute instances list --project "$project" --filter="name=($instance) AND zone:($zone)" --format='value(name)' 2>/dev/null)"
    list_rc=$?
    set -e

    if [[ "$vm_rc" == "0" && -n "$vm_status" ]]; then
      return 1
    fi
    if [[ "$list_rc" != "0" ]]; then
      echo "WARN: VM status list probe failed for $instance (attempt $attempt/3); treating as UNKNOWN." >&2
      return 1
    fi
    if [[ -n "$listed" ]]; then
      return 1
    fi
    sleep 3
  done

  return 0
}

wait_for_vm_delete_after_terminal_marker() {
  local instance="$1"
  local project="$2"
  local zone="$3"
  local deadline=$(($(date +%s) + 1200))
  local vm_status
  while (($(date +%s) < deadline)); do
    vm_status="$(probe_vm_status "$instance" "$project" "$zone")"
    if [[ "$vm_status" == "UNKNOWN" ]] && vm_absence_confirmed "$instance" "$project" "$zone" "gs://unused"; then
      echo "VM $instance is gone."
      return 0
    fi
    echo "  ...terminal marker found; waiting for VM cleanup/cache upload ($instance status=$vm_status)"
    sleep 15
  done
  echo "WARN: VM $instance still exists after terminal marker; continuing. Delete manually if it remains after cache upload." >&2
}

wait_for_gcp_run() {
  local out_dir="$1"
  local meta="$out_dir/gcp_last_run.json"
  if [[ ! -f "$meta" ]]; then
    echo "ERROR: missing $meta (cannot wait for completion)" >&2
    return 1
  fi
  local prefix instance zone project
  prefix="$(
    python - "$meta" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["run_prefix"].rstrip("/"))
PY
  )"
  instance="$(
    python - "$meta" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["instance_name"])
PY
  )"
  zone="$(
    python - "$meta" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["zone"])
PY
  )"
  project="$(
    python - "$meta" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["project"])
PY
  )"
  echo "Waiting for run to finish: $prefix"
  local start
  start="$(date +%s)"
  while true; do
    if run_with_timeout 20 gcloud storage objects describe "$prefix/DONE" >/dev/null 2>&1; then
      echo "DONE sentinel found."
      mkdir -p "$out_dir"
      run_with_timeout 300 gcloud storage rsync --recursive "$prefix/results/" "$out_dir/" || {
        echo "ERROR: failed to fetch results from $prefix/results/" >&2
        return 1
      }
      wait_for_vm_delete_after_terminal_marker "$instance" "$project" "$zone"
      return 0
    fi
    if run_with_timeout 20 gcloud storage objects describe "$prefix/FAILED" >/dev/null 2>&1; then
      echo "ERROR: FAILED sentinel found." >&2
      run_with_timeout 60 gcloud storage cat "$prefix/FAILED" 2>/dev/null || true
      echo "Logs: gcloud storage cat $prefix/vm.log" >&2
      mkdir -p "$out_dir"
      run_with_timeout 300 gcloud storage rsync --recursive "$prefix/results/" "$out_dir/" >/dev/null 2>&1 || true
      return 1
    fi
    if run_with_timeout 20 gcloud storage objects describe "$prefix/exit_code.txt" >/dev/null 2>&1; then
      local rc
      rc="$(run_with_timeout 60 gcloud storage cat "$prefix/exit_code.txt" | tr -d ' \t\r\n')"
      echo "Run finished with exit_code=$rc"
      if [[ "$rc" != "0" ]]; then
        echo "ERROR: remote benchmark failed (see $prefix/vm.log)" >&2
        mkdir -p "$out_dir"
        run_with_timeout 300 gcloud storage rsync --recursive "$prefix/results/" "$out_dir/" >/dev/null 2>&1 || true
        return 1
      fi
      mkdir -p "$out_dir"
      run_with_timeout 300 gcloud storage rsync --recursive "$prefix/results/" "$out_dir/" || {
        echo "ERROR: failed to fetch results from $prefix/results/" >&2
        return 1
      }
      wait_for_vm_delete_after_terminal_marker "$instance" "$project" "$zone"
      return 0
    fi
    local vm_status
    vm_status="$(probe_vm_status "$instance" "$project" "$zone")"
    if [[ "$vm_status" == "UNKNOWN" ]]; then
      echo "WARN: could not confirm VM status for $instance; treating as UNKNOWN, not gone." >&2
      echo "  rescue log: gcloud storage cat $prefix/vm.log" >&2
      echo "  rescue ssh: gcloud compute ssh $instance --project $project --zone $zone --command 'sudo tail -n 120 /var/log/benchmark-gcp-run.log'" >&2
    fi
    if [[ "$vm_status" == "UNKNOWN" ]] && vm_absence_confirmed "$instance" "$project" "$zone" "$prefix"; then
      echo "ERROR: VM $instance is gone but no DONE/FAILED/exit_code.txt was written." >&2
      echo "Logs: gcloud storage cat $prefix/vm.log" >&2
      mkdir -p "$out_dir"
      run_with_timeout 300 gcloud storage rsync --recursive "$prefix/results/" "$out_dir/" >/dev/null 2>&1 || true
      return 1
    fi
    local now
    now="$(date +%s)"
    if ((now - start > MAX_WAIT_SECS)); then
      echo "ERROR: timed out waiting for $prefix/exit_code.txt" >&2
      return 1
    fi
    echo "  ...still running ($prefix) - sleep ${POLL_SECS}s"
    sleep "$POLL_SECS"
  done
}

for MACHINE_TYPE in $MACHINE_TYPES; do
  LOCAL_OUT=/Users/vladimiriglovikov/workspace/benchmark/gcp_runs/${MACHINE_TYPE}-rgb-micro-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$LOCAL_OUT"

  echo "=== submitting RGB micro on ${MACHINE_TYPE} ==="

  cmd=(
    python -m benchmark.cli run
    --cloud gcp
    --gcp-project "$PROJECT"
    --gcp-zone "$ZONE"
    --gcp-machine-type "$MACHINE_TYPE"
    --gcp-gcs-data-uri "$IMAGENET_VAL_TAR_GCS"
    --gcp-gcs-results-uri "$RESULTS_GCS"
    --gcp-disk-size-gb 200
    --data-dir /tmp/unused
    --output "$LOCAL_OUT"
    --scenario image-rgb
    --mode micro
    --libraries albumentationsx torchvision kornia pillow
    --num-items "$NUM_ITEMS"
    --num-runs "$NUM_RUNS"
    --timer pyperf
  )
  if [[ -n "$TRANSFORMS" ]]; then
    cmd+=(--transforms "$TRANSFORMS")
  fi
  if [[ -n "$SLOW_THRESHOLD_SEC_PER_ITEM" ]]; then
    cmd+=(--slow-threshold-sec-per-item "$SLOW_THRESHOLD_SEC_PER_ITEM")
  fi
  if [[ -n "$SLOW_PREFLIGHT_ITEMS" ]]; then
    cmd+=(--slow-preflight-items "$SLOW_PREFLIGHT_ITEMS")
  fi
  if [[ "$DISABLE_SLOW_SKIP" == "1" ]]; then
    cmd+=(--disable-slow-skip)
  fi
  if [[ -n "$GCP_VENV_CACHE_URI" ]]; then
    cmd+=(--gcp-venv-cache-uri "$GCP_VENV_CACHE_URI")
  fi
  if [[ "$GCP_NO_VENV_CACHE" == "1" ]]; then
    cmd+=(--gcp-no-venv-cache)
  fi
  if [[ "$GCP_FORCE_VENV_CACHE_REBUILD" == "1" ]]; then
    cmd+=(--gcp-force-venv-cache-rebuild)
  fi
  if [[ "$KEEP_ON_FAILURE" == "1" ]]; then
    cmd+=(--gcp-keep-on-failure)
  fi
  "${cmd[@]}"

  if [[ "$WAIT_FOR_GCP_COMPLETION" == "1" ]]; then
    wait_for_gcp_run "$LOCAL_OUT"
  fi
done
