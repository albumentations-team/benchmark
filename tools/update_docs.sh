#!/usr/bin/env bash
# Update README with benchmark tables from result JSONs.
#
# Usage:
#   ./tools/update_docs.sh
#   ./tools/update_docs.sh --image-results output/rgb_micro_macos_m4max/image-rgb/micro
#   ./tools/update_docs.sh --multichannel-results output/multichannel_micro_paper_core/multichannel
#   ./tools/update_docs.sh --video-results output_videos/

set -e

# Default to the MacBook M4 RGB micro run while the remaining paper benchmarks are still in progress.
IMAGE_RESULTS="${IMAGE_RESULTS:-output/rgb_micro_macos_m4max/image-rgb/micro}"
MULTICHANNEL_RESULTS="${MULTICHANNEL_RESULTS:-}"
VIDEO_RESULTS="${VIDEO_RESULTS:-}"

while [[ $# -gt 0 ]]; do
  case $1 in
  --image-results)
    IMAGE_RESULTS="$2"
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
if [[ -n "$MULTICHANNEL_RESULTS" ]]; then
  echo "Updating docs from multichannel results: $MULTICHANNEL_RESULTS"
fi
if [[ -n "$VIDEO_RESULTS" ]]; then
  echo "Updating docs from video results: $VIDEO_RESULTS"
fi

# Print comparison tables
if [[ -d "$IMAGE_RESULTS" ]] && ls "$IMAGE_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "Image comparison table:"
  python -m tools.compare --results-dir "$IMAGE_RESULTS"
fi

if [[ -n "$VIDEO_RESULTS" && -d "$VIDEO_RESULTS" ]] && ls "$VIDEO_RESULTS"/*_video_results.json 1>/dev/null 2>&1; then
  echo "Video comparison table:"
  python -m tools.compare --results-dir "$VIDEO_RESULTS"
fi

if [[ -z "$MULTICHANNEL_RESULTS" ]]; then
  MULTICHANNEL_RESULTS="${IMAGE_RESULTS}/multichannel"
fi
if [[ -d "$MULTICHANNEL_RESULTS" ]] && ls "$MULTICHANNEL_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "Multichannel comparison table:"
  python -m tools.compare --results-dir "$MULTICHANNEL_RESULTS"
fi

# Patch README with full benchmark tables
echo "Updating README..."
UPDATE_README_ARGS=(
  --image-results "$IMAGE_RESULTS"
  --multichannel-results "$MULTICHANNEL_RESULTS"
)
if [[ -n "$VIDEO_RESULTS" ]]; then
  UPDATE_README_ARGS+=(--video-results "$VIDEO_RESULTS")
else
  UPDATE_README_ARGS+=(--video-results /tmp/benchmark-no-video-results)
fi
python -m tools.update_readme "${UPDATE_README_ARGS[@]}"

echo "Done. Check README.md"
