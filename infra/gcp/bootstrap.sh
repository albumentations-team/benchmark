#!/usr/bin/env bash
# Stage one frozen RGB run, execute it, then leave terminal cleanup to the controller.
set -Eeuo pipefail

readonly STATE_ROOT=/var/lib/augbench
readonly REQUEST_PATH="$STATE_ROOT/request.json"
readonly LOG_PATH="$STATE_ROOT/bootstrap.log"
readonly METADATA_URL=http://metadata.google.internal/computeMetadata/v1/instance/attributes/augbench-request-uri
readonly INSTANCE_ID_URL=http://metadata.google.internal/computeMetadata/v1/instance/id

request_field() {
  python3 -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
' "$REQUEST_PATH" "$1"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh | sh -s -- --quiet
  export PATH="/root/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

download_verified() {
  local uri="$1"
  local expected="$2"
  local target="$3"
  if [[ -f "$target" ]] && [[ "$(sha256sum "$target" | awk "{print \$1}")" == "$expected" ]]; then
    return
  fi
  local partial="${target}.partial"
  rm -f "$partial"
  gcloud storage cp --quiet "$uri" "$partial"
  printf '%s  %s\n' "$expected" "$partial" | sha256sum --check --status
  mv "$partial" "$target"
}

repair_python_links() {
  local environment_root="$1"
  local python_version="$2"
  local managed_python
  managed_python="$(uv python find --managed-python --resolve-links "$python_version")"
  rm -f "$environment_root/bin/python" "$environment_root/bin/python3" \
    "$environment_root/bin/python${python_version%.*}"
  ln -s "$managed_python" "$environment_root/bin/python"
  ln -s python "$environment_root/bin/python3"
  ln -s python "$environment_root/bin/python${python_version%.*}"
}

stage_environment() {
  local cache_uri="$1"
  local lock_path="$2"
  local lock_sha256="$3"
  local python_version="$4"
  local environment_root="$STATE_ROOT/environment"
  local cache_archive="$STATE_ROOT/downloads/environment.tar.gz"
  local cache_missing=false
  printf '%s  %s\n' "$lock_sha256" "$lock_path" | sha256sum --check --status
  if ! gcloud storage cp --quiet "$cache_uri" "$cache_archive"; then
    cache_missing=true
    rm -rf "$environment_root"
    uv python install --no-bin "$python_version"
    uv venv --relocatable --managed-python --link-mode copy --python "$python_version" "$environment_root"
  else
    rm -rf "$environment_root"
    mkdir -p "$environment_root"
    tar -xzf "$cache_archive" -C "$environment_root"
    uv python install --no-bin "$python_version"
  fi
  repair_python_links "$environment_root" "$python_version"
  uv pip sync --python "$environment_root/bin/python" --require-hashes --link-mode copy --torch-backend cu130 "$lock_path"
  if [[ "$cache_missing" == true ]]; then
    tar --exclude='bin/python*' -czf "$cache_archive" -C "$environment_root" .
    gcloud storage cp --quiet --if-generation-match=0 "$cache_archive" "$cache_uri" || true
  fi
}

publish_log() {
  [[ -s "$LOG_PATH" && -f "$REQUEST_PATH" ]] || return 0
  local gcs_base_uri run_id instance_id
  gcs_base_uri="$(request_field gcs_base_uri)" || return 0
  run_id="$(request_field run.run_id)" || return 0
  instance_id="$(curl --fail --silent --show-error --header 'Metadata-Flavor: Google' "$INSTANCE_ID_URL" || hostname)"
  gcloud storage cp --quiet --if-generation-match=0 "$LOG_PATH" \
    "${gcs_base_uri%/}/runs/${run_id}/logs/bootstrap-${instance_id}.log" || true
}

finish() {
  local status=$?
  trap - EXIT
  set +e
  publish_log
  sync
  shutdown -h now || true
  exit "$status"
}
trap finish EXIT

mkdir -p "$STATE_ROOT/downloads"
exec > >(tee -a "$LOG_PATH") 2>&1

ensure_uv
request_uri="$(curl --fail --silent --show-error --header 'Metadata-Flavor: Google' "$METADATA_URL")"
gcloud storage cp --quiet "$request_uri" "$REQUEST_PATH"

code_uri="$(request_field code_archive_uri)"
code_sha256="$(request_field run.inputs.code_archive_sha256)"
dataset_uri="$(request_field dataset_archive_uri)"
dataset_sha256="$(request_field dataset_archive_sha256)"
download_verified "$code_uri" "$code_sha256" "$STATE_ROOT/downloads/code.tar.gz"
download_verified "$dataset_uri" "$dataset_sha256" "$STATE_ROOT/downloads/dataset.tar"

readonly CODE_ROOT="$STATE_ROOT/code"
rm -rf "$CODE_ROOT"
mkdir -p "$CODE_ROOT"
tar -xzf "$STATE_ROOT/downloads/code.tar.gz" -C "$CODE_ROOT"

stage_environment \
  "$(request_field environment_cache_uri)" \
  "$CODE_ROOT/$(request_field environment_lock_path)" \
  "$(request_field environment_lock_sha256)" \
  "$(request_field environment_python_version)"

cd "$CODE_ROOT"
PYTHONPATH="$CODE_ROOT/src" "$STATE_ROOT/environment/bin/python" -m augbench.guest_worker \
  --request "$REQUEST_PATH" \
  --family-config "$CODE_ROOT/configs/families/rgb.yaml" \
  --dataset-archive "$STATE_ROOT/downloads/dataset.tar" \
  --dataset-cache "$STATE_ROOT/datasets"
