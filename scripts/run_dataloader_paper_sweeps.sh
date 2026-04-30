#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/dataloader-paper-sweeps}"
NUM_RUNS="${NUM_RUNS:-5}"
MIN_TIME="${MIN_TIME:-0}"
MIN_BATCHES="${MIN_BATCHES:-20}"
WORKERS="${WORKERS:-0 1 2 4 8 16}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-64}"
VIDEO_BATCH_SIZE="${VIDEO_BATCH_SIZE:-8}"
THREAD_POLICY="${THREAD_POLICY:-pipeline-default}"
DEVICE="${DEVICE:-none}"

run_sweep() {
  local scenario="$1"
  local data_dir="$2"
  local scope="$3"
  local batch_size="$4"
  local workers

  for workers in ${WORKERS}; do
    "${PYTHON_BIN}" -m benchmark.cli run \
      --scenario "${scenario}" \
      --mode pipeline \
      --pipeline-scope "${scope}" \
      --thread-policy "${THREAD_POLICY}" \
      --device "${DEVICE}" \
      --data-dir "${data_dir}" \
      --output "${OUTPUT_DIR}/${scenario}/${scope}/workers-${workers}" \
      --batch-size "${batch_size}" \
      --workers "${workers}" \
      --num-runs "${NUM_RUNS}" \
      --min-time "${MIN_TIME}" \
      --min-batches "${MIN_BATCHES}"
  done
}

if [[ -n "${RGB_DATA_DIR:-}" ]]; then
  run_sweep "image-rgb" "${RGB_DATA_DIR}" "memory_dataloader_augment" "${IMAGE_BATCH_SIZE}"
  run_sweep "image-rgb" "${RGB_DATA_DIR}" "decode_dataloader_augment" "${IMAGE_BATCH_SIZE}"
fi

if [[ -n "${MULTICHANNEL_DATA_DIR:-}" ]]; then
  run_sweep "image-9ch" "${MULTICHANNEL_DATA_DIR}" "memory_dataloader_augment" "${IMAGE_BATCH_SIZE}"
  run_sweep "image-9ch" "${MULTICHANNEL_DATA_DIR}" "decode_dataloader_augment" "${IMAGE_BATCH_SIZE}"
fi

if [[ -n "${VIDEO_DATA_DIR:-}" ]]; then
  run_sweep "video-16f" "${VIDEO_DATA_DIR}" "memory_dataloader_augment" "${VIDEO_BATCH_SIZE}"
  run_sweep "video-16f" "${VIDEO_DATA_DIR}" "decode_dataloader_augment" "${VIDEO_BATCH_SIZE}"
fi

if [[ "${DEVICE}" != "none" ]]; then
  if [[ -n "${RGB_DATA_DIR:-}" ]]; then
    run_sweep "image-rgb" "${RGB_DATA_DIR}" "decode_dataloader_augment_batch_copy" "${IMAGE_BATCH_SIZE}"
  fi
  if [[ -n "${VIDEO_DATA_DIR:-}" ]]; then
    run_sweep "video-16f" "${VIDEO_DATA_DIR}" "decode_dataloader_augment_batch_copy" "${VIDEO_BATCH_SIZE}"
  fi
fi
