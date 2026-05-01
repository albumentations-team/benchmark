#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-}"
OUTPUT="${OUTPUT:-output/multichannel_micro_paper_core}"
NUM_ITEMS="${NUM_ITEMS:-2000}"
NUM_RUNS="${NUM_RUNS:-5}"
NO_REFRESH_REQUIREMENTS="${NO_REFRESH_REQUIREMENTS:-0}"
PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"
PUBLISH_FORCE="${PUBLISH_FORCE:-1}"
PUBLISH_PURPOSE="${PUBLISH_PURPOSE:-website-multichannel}"
PUBLISH_MACHINE="${PUBLISH_MACHINE:-macos}"
PUBLISH_NOTES="${PUBLISH_NOTES:-9-channel multichannel micro benchmark snapshot for website/repo tables; summary JSONs only.}"
PUBLISHED_RESULTS_DIR="${PUBLISHED_RESULTS_DIR:-results/published/website-multichannel-micro-macos-$(date +%F)}"
UPDATE_README="${UPDATE_README:-1}"
DOCS_IMAGE_RESULTS="${DOCS_IMAGE_RESULTS:-output/rgb_micro_macos_m4max/image-rgb/micro}"
DOCS_VIDEO_RESULTS="${DOCS_VIDEO_RESULTS:-}"

if [[ -z "$DATA_DIR" ]]; then
  cat >&2 <<'EOF'
ERROR: DATA_DIR is required.

Usage:
  DATA_DIR=/path/to/imagenet/val scripts/run_multichannel_micro_paper_core.sh

Smoke:
  DATA_DIR=/path/to/imagenet/val NUM_ITEMS=50 NUM_RUNS=1 \
    OUTPUT=output/multichannel_micro_paper_core_smoke PUBLISH_RESULTS=0 UPDATE_README=0 \
    scripts/run_multichannel_micro_paper_core.sh

Publishing:
  PUBLISH_RESULTS=1 writes summary JSONs to results/published/website-multichannel-micro-macos-YYYY-MM-DD.
  UPDATE_README=1 refreshes README.md from DOCS_IMAGE_RESULTS and the published multichannel snapshot.

Extra benchmark.cli args can be appended after the script name.
EOF
  exit 2
fi

PAPER_CORE_TRANSFORMS=(
  Resize
  RandomCrop224
  RandomResizedCrop
  CenterCrop224
  HorizontalFlip
  VerticalFlip
  Pad
  Rotate
  Affine
  Perspective
  Elastic
  ChannelShuffle
  Grayscale
  GaussianBlur
  GaussianNoise
  Invert
  Posterize
  Solarize
  Sharpen
  AutoContrast
  Normalize
  Erasing
  JpegCompression
  RandomGamma
  MedianBlur
  MotionBlur
  Brightness
  Contrast
  Blur
  ChannelDropout
  LinearIllumination
  CornerIllumination
  GaussianIllumination
  PlasmaBrightness
  PlasmaContrast
  PlasmaShadow
  OpticalDistortion
  Shear
  ThinPlateSpline
  LongestMaxSize
  SmallestMaxSize
  RandomJigsaw
  RandomRotate90
)

cmd=(
  python -m benchmark.cli run
  --media image
  --mode micro
  --data-dir "$DATA_DIR"
  --output "$OUTPUT"
  --libraries albumentationsx torchvision kornia
  --multichannel
  --transforms "${PAPER_CORE_TRANSFORMS[@]}"
  --num-items "$NUM_ITEMS"
  --num-runs "$NUM_RUNS"
)

if [[ "$NO_REFRESH_REQUIREMENTS" == "1" ]]; then
  cmd+=(--no-refresh-requirements)
fi

cmd+=("$@")

printf 'Running 9-channel micro paper-core benchmark (%d transforms)\n' "${#PAPER_CORE_TRANSFORMS[@]}"
printf 'Output root: %s\n' "$OUTPUT"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

MULTICHANNEL_RESULTS_DIR="$OUTPUT/multichannel"
MULTICHANNEL_DOCS_RESULTS="$MULTICHANNEL_RESULTS_DIR"

if [[ "$PUBLISH_RESULTS" == "1" ]]; then
  publish_cmd=(
    python -m tools.publish_results
    --from "$MULTICHANNEL_RESULTS_DIR"
    --to "$PUBLISHED_RESULTS_DIR"
    --purpose "$PUBLISH_PURPOSE"
    --machine "$PUBLISH_MACHINE"
    --notes "$PUBLISH_NOTES"
  )
  if [[ "$PUBLISH_FORCE" == "1" ]]; then
    publish_cmd+=(--force)
  fi

  printf 'Publishing curated multichannel results:'
  printf ' %q' "${publish_cmd[@]}"
  printf '\n'
  "${publish_cmd[@]}"
  MULTICHANNEL_DOCS_RESULTS="$PUBLISHED_RESULTS_DIR"
fi

if [[ "$UPDATE_README" == "1" ]]; then
  update_docs_cmd=(
    ./tools/update_docs.sh
    --image-results "$DOCS_IMAGE_RESULTS"
    --multichannel-results "$MULTICHANNEL_DOCS_RESULTS"
  )
  if [[ -n "$DOCS_VIDEO_RESULTS" ]]; then
    update_docs_cmd+=(--video-results "$DOCS_VIDEO_RESULTS")
  fi

  printf 'Updating README/docs:'
  printf ' %q' "${update_docs_cmd[@]}"
  printf '\n'
  "${update_docs_cmd[@]}"
fi
