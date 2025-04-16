#!/bin/bash
set -e

# Default paths
IMAGE_RESULTS_DIR="output"
VIDEO_RESULTS_DIR="output_videos"
DOCS_DIR="docs"
MAIN_README="README.md"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --image-results)
      IMAGE_RESULTS_DIR="$2"
      shift 2
      ;;
    --video-results)
      VIDEO_RESULTS_DIR="$2"
      shift 2
      ;;
    --docs-dir)
      DOCS_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --image-results DIR   Directory containing image benchmark results (default: output)"
      echo "  --video-results DIR   Directory containing video benchmark results (default: output_videos)"
      echo "  --docs-dir DIR        Directory for documentation (default: docs)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "$DOCS_DIR/images"
mkdir -p "$DOCS_DIR/videos"

echo "Updating documentation..."

# Generate image benchmark results and plots
if [ -d "$IMAGE_RESULTS_DIR" ]; then
  echo "Processing image benchmark results..."

  # Generate markdown table and update image README
  python tools/compare_results.py \
    --results-dir "$IMAGE_RESULTS_DIR" \
    --update-readme "$DOCS_DIR/images/README.md"

  # Generate speedup analysis plot
  python tools/generate_speedup_plots.py \
    --results-dir "$IMAGE_RESULTS_DIR" \
    --output-dir "$DOCS_DIR/images" \
    --type images \
    --reference-library albumentations

  echo "Image documentation updated."
else
  echo "Warning: Image results directory not found at $IMAGE_RESULTS_DIR"
fi

# Generate video benchmark results and plots
if [ -d "$VIDEO_RESULTS_DIR" ]; then
  echo "Processing video benchmark results..."

  # Generate markdown table and update video README
  python tools/compare_video_results.py \
    --results-dir "$VIDEO_RESULTS_DIR" \
    --update-readme "$DOCS_DIR/videos/README.md"

  # Generate speedup analysis plot
  python tools/generate_speedup_plots.py \
    --results-dir "$VIDEO_RESULTS_DIR" \
    --output-dir "$DOCS_DIR/videos" \
    --type videos \
    --reference-library albumentations

  echo "Video documentation updated."
else
  echo "Warning: Video results directory not found at $VIDEO_RESULTS_DIR"
fi

# Update speedup summaries in main README
if [ -f "$DOCS_DIR/images/images_speedups.csv" ]; then
  # Extract median speedup from CSV
  MEDIAN_SPEEDUP=$(python -c "import pandas as pd; df = pd.read_csv('$DOCS_DIR/images/images_speedups.csv', index_col=0); print(f'{df[\"albumentations\"].median():.2f}')")
  MAX_SPEEDUP=$(python -c "import pandas as pd; df = pd.read_csv('$DOCS_DIR/images/images_speedups.csv', index_col=0); max_val = df[\"albumentations\"].max(); max_transform = df[\"albumentations\"].idxmax(); print(f'{max_val:.2f}× ({max_transform})')")

  # Update image speedup summary in main README
  IMAGE_SUMMARY="Albumentations is generally the fastest library for image augmentation, with a median speedup of ${MEDIAN_SPEEDUP}× compared to other libraries. For some transforms, the speedup can be as high as ${MAX_SPEEDUP}."

  # Use sed to replace the content between markers
  sed -i.bak "s|<!-- IMAGE_SPEEDUP_SUMMARY_START -->.*<!-- IMAGE_SPEEDUP_SUMMARY_END -->|<!-- IMAGE_SPEEDUP_SUMMARY_START -->\n${IMAGE_SUMMARY}\n<!-- IMAGE_SPEEDUP_SUMMARY_END -->|" "$MAIN_README"
  rm "${MAIN_README}.bak"
fi

# Update video speedup summary in main README
if [ -f "$DOCS_DIR/videos/videos_speedups.csv" ]; then
  # Extract statistics from CSV
  FASTER_CPU_COUNT=$(python -c "import pandas as pd; df = pd.read_csv('$DOCS_DIR/videos/videos_speedups.csv', index_col=0); print(len(df[df['albumentations'] > 1]))")
  TOTAL_TRANSFORMS=$(python -c "import pandas as pd; df = pd.read_csv('$DOCS_DIR/videos/videos_speedups.csv', index_col=0); print(len(df))")
  FASTEST_CPU_TRANSFORM=$(python -c "import pandas as pd; df = pd.read_csv('$DOCS_DIR/videos/videos_speedups.csv', index_col=0); max_idx = df['albumentations'].idxmax(); print(f'{max_idx} ({df.loc[max_idx, \"albumentations\"]:.2f}×)')")
  FASTEST_GPU_TRANSFORM=$(python -c "import pandas as pd; df = pd.read_csv('$DOCS_DIR/videos/videos_speedups.csv', index_col=0); min_idx = df['albumentations'].idxmin(); print(f'{min_idx} ({1/df.loc[min_idx, \"albumentations\"]:.2f}×)')")

  # Update video speedup summary in main README
  VIDEO_SUMMARY="For video processing, the performance comparison between CPU (Albumentations) and GPU (Kornia) shows interesting trade-offs. CPU processing is faster for ${FASTER_CPU_COUNT} out of ${TOTAL_TRANSFORMS} transforms, with ${FASTEST_CPU_TRANSFORM} showing the highest CPU advantage. GPU excels at complex operations like ${FASTEST_GPU_TRANSFORM}."

  # Use sed to replace the content between markers
  sed -i.bak "s|<!-- VIDEO_SPEEDUP_SUMMARY_START -->.*<!-- VIDEO_SPEEDUP_SUMMARY_END -->|<!-- VIDEO_SPEEDUP_SUMMARY_START -->\n${VIDEO_SUMMARY}\n<!-- VIDEO_SPEEDUP_SUMMARY_END -->|" "$MAIN_README"
  rm "${MAIN_README}.bak"
fi

echo "Documentation update complete."
