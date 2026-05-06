#!/usr/bin/env bash
# Update README with benchmark tables from result JSONs.
#
# Usage:
#   ./tools/update_docs.sh
#   ./tools/update_docs.sh --image-results output/rgb_micro/image-rgb/micro
#   ./tools/update_docs.sh --dataloader-results output/rgb_dataloader/image-rgb/pipeline
#   ./tools/update_docs.sh --multichannel-results output/image_9ch/multichannel  # accepted, not inserted into README
#   ./tools/update_docs.sh --video-results output/video_micro                  # accepted, not inserted into README

set -e

# Default to the current tracked RGB paper-focused CPU snapshots.
IMAGE_RESULTS="${IMAGE_RESULTS:-results/published/paper-rgb-micro-c4-standard-16-2026-05-04}"
DATALOADER_RESULTS="${DATALOADER_RESULTS:-results/published/paper-rgb-dataloader-memory-c4-standard-16-2026-05-04}"
MULTICHANNEL_RESULTS="${MULTICHANNEL_RESULTS:-}"
VIDEO_RESULTS="${VIDEO_RESULTS:-}"

while [[ $# -gt 0 ]]; do
  case $1 in
  --image-results)
    IMAGE_RESULTS="$2"
    shift 2
    ;;
  --dataloader-results)
    DATALOADER_RESULTS="$2"
    shift 2
    ;;
  --multichannel-results)
    MULTICHANNEL_RESULTS="$2"
    shift 2
    ;;
  --video-results)
    VIDEO_RESULTS="$2"
    shift 2
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

echo "Updating docs from image results: $IMAGE_RESULTS"
echo "Updating docs from DataLoader results: $DATALOADER_RESULTS"
if [[ -n "$MULTICHANNEL_RESULTS" ]]; then
  echo "Accepted multichannel results path for compatibility: $MULTICHANNEL_RESULTS"
fi
if [[ -n "$VIDEO_RESULTS" ]]; then
  echo "Accepted video results path for compatibility: $VIDEO_RESULTS"
fi

# Print comparison tables
if [[ -d "$IMAGE_RESULTS" ]] && ls "$IMAGE_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "RGB micro comparison table:"
  python -m tools.compare --results-dir "$IMAGE_RESULTS"
fi

if [[ -d "$DATALOADER_RESULTS" ]] && ls "$DATALOADER_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "RGB DataLoader comparison table:"
  python -m tools.compare --results-dir "$DATALOADER_RESULTS"
fi

if [[ -n "$MULTICHANNEL_RESULTS" && -d "$MULTICHANNEL_RESULTS" ]] && ls "$MULTICHANNEL_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "9-channel comparison table (not inserted into README):"
  python -m tools.compare --results-dir "$MULTICHANNEL_RESULTS"
fi

if [[ -n "$VIDEO_RESULTS" && -d "$VIDEO_RESULTS" ]] && ls "$VIDEO_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "Video comparison table (not inserted into README):"
  python -m tools.compare --results-dir "$VIDEO_RESULTS"
fi

# Patch README with full benchmark tables
echo "Updating README..."
UPDATE_README_ARGS=(
  --image-results "$IMAGE_RESULTS"
  --dataloader-results "$DATALOADER_RESULTS"
)
python -m tools.update_readme "${UPDATE_README_ARGS[@]}"

echo "Done. Check README.md"
