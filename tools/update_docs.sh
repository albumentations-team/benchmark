#!/usr/bin/env bash
# Update benchmark documentation with latest results.
#
# Usage:
#   ./tools/update_docs.sh
#   ./tools/update_docs.sh --image-results output/ --video-results output_videos/

set -e

IMAGE_RESULTS="${IMAGE_RESULTS:-output/}"
VIDEO_RESULTS="${VIDEO_RESULTS:-output_videos/}"
DOCS_DIR="${DOCS_DIR:-docs}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --image-results) IMAGE_RESULTS="$2"; shift 2 ;;
    --video-results) VIDEO_RESULTS="$2"; shift 2 ;;
    --docs-dir) DOCS_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "Updating docs from image results: $IMAGE_RESULTS"
echo "Updating docs from video results: $VIDEO_RESULTS"

# Generate image comparison table and plot
if [[ -d "$IMAGE_RESULTS" ]] && ls "$IMAGE_RESULTS"/*_results.json 1>/dev/null 2>&1; then
  echo "Generating image speedup plot..."
  python -m tools.generate_speedup_plots \
    --results-dir "$IMAGE_RESULTS" \
    --output-dir "$DOCS_DIR/images" \
    --type images

  echo "Image comparison table:"
  python -m tools.compare --results-dir "$IMAGE_RESULTS"
fi

# Generate video comparison (if results exist)
if [[ -d "$VIDEO_RESULTS" ]] && ls "$VIDEO_RESULTS"/*_video_results.json 1>/dev/null 2>&1; then
  echo "Generating video speedup plot..."
  python -m tools.generate_speedup_plots \
    --results-dir "$VIDEO_RESULTS" \
    --output-dir "$DOCS_DIR/videos" \
    --type videos

  echo "Video comparison table:"
  python -m tools.compare --results-dir "$VIDEO_RESULTS"
fi

# Patch README with full benchmark tables
echo "Updating README..."
python -m tools.update_readme \
  --image-results "$IMAGE_RESULTS" \
  --video-results "$VIDEO_RESULTS"

echo "Done. Check README.md, $DOCS_DIR/images/, and $DOCS_DIR/videos/"
