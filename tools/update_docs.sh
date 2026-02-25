#!/usr/bin/env bash
# Update README with benchmark tables from result JSONs.
#
# Usage:
#   ./tools/update_docs.sh
#   ./tools/update_docs.sh --image-results output/ --video-results output_videos/

set -e

IMAGE_RESULTS="${IMAGE_RESULTS:-output/}"
VIDEO_RESULTS="${VIDEO_RESULTS:-output_videos/}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --image-results) IMAGE_RESULTS="$2"; shift 2 ;;
    --video-results) VIDEO_RESULTS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "Updating docs from image results: $IMAGE_RESULTS"
echo "Updating docs from video results: $VIDEO_RESULTS"

# Print comparison tables
if [[ -d "$IMAGE_RESULTS" ]] && ls "$IMAGE_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "Image comparison table:"
  python -m tools.compare --results-dir "$IMAGE_RESULTS"
fi

if [[ -d "$VIDEO_RESULTS" ]] && ls "$VIDEO_RESULTS"/*_video_results.json 1>/dev/null 2>&1; then
  echo "Video comparison table:"
  python -m tools.compare --results-dir "$VIDEO_RESULTS"
fi

MULTICHANNEL_RESULTS="${IMAGE_RESULTS}/multichannel"
if [[ -d "$MULTICHANNEL_RESULTS" ]] && ls "$MULTICHANNEL_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "Multichannel comparison table:"
  python -m tools.compare --results-dir "$MULTICHANNEL_RESULTS"
fi

# Patch README with full benchmark tables
echo "Updating README..."
python -m tools.update_readme \
  --image-results "$IMAGE_RESULTS" \
  --video-results "$VIDEO_RESULTS" \
  --multichannel-results "$MULTICHANNEL_RESULTS"

echo "Done. Check README.md"
