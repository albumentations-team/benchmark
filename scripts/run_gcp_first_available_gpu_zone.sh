#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-${1:-configs/paper/prod_g2_video_micro_gpu.yaml}}"
if [[ "${1:-}" == "$CONFIG" ]]; then
  shift
fi

ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-west1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west4-a us-west4-c northamerica-northeast1-b northamerica-northeast1-c northamerica-northeast2-a northamerica-northeast2-b europe-west1-b europe-west1-c europe-west4-a europe-west4-b europe-west4-c}"
LOG_DIR="${LOG_DIR:-gcp_runs/zone_attempt_logs}"
mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d-%H%M%S)"

is_capacity_failure() {
  local log_file="$1"
  grep -Eq \
    "ZONE_RESOURCE_POOL_EXHAUSTED|ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS|resource_availability|STOCKOUT|currently unavailable|does not have enough resources" \
    "$log_file"
}

echo "Config: $CONFIG"
echo "Zones: $ZONES"
echo "Extra args: $*"
echo

for zone in $ZONES; do
  log_file="$LOG_DIR/${timestamp}-${zone}.log"
  echo "=== trying $zone ==="

  set +e
  python -m benchmark.cli run --config "$CONFIG" --gcp-zone "$zone" "$@" 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" == "0" ]]; then
    echo
    echo "Launched successfully in $zone"
    echo "Log: $log_file"
    exit 0
  fi

  if is_capacity_failure "$log_file"; then
    echo "Capacity unavailable in $zone; trying next zone."
    echo
    continue
  fi

  echo "Non-capacity failure in $zone; stopping. See $log_file" >&2
  exit "$rc"
done

echo "No zone accepted the launch for $CONFIG." >&2
echo "Attempt logs are in $LOG_DIR" >&2
exit 1
