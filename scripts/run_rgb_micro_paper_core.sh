#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-}"
OUTPUT="${OUTPUT:-output/rgb_micro_paper_core}"
NUM_ITEMS="${NUM_ITEMS:-2000}"
NUM_RUNS="${NUM_RUNS:-5}"
TIMER="${TIMER:-pyperf}"
NO_REFRESH_REQUIREMENTS="${NO_REFRESH_REQUIREMENTS:-1}"

if [[ -z "$DATA_DIR" ]]; then
  cat >&2 <<'EOF'
ERROR: DATA_DIR is required.

Usage:
  DATA_DIR=/path/to/imagenet/val scripts/run_rgb_micro_paper_core.sh

Smoke:
  DATA_DIR=/path/to/imagenet/val NUM_ITEMS=50 NUM_RUNS=1 TIMER=simple \
    OUTPUT=output/rgb_micro_paper_core_smoke scripts/run_rgb_micro_paper_core.sh

Extra benchmark.cli args can be appended after the script name.
EOF
  exit 2
fi

PAPER_CORE_TRANSFORMS=(
  Resize
  RandomCrop128
  RandomResizedCrop
  CenterCrop128
  HorizontalFlip
  VerticalFlip
  Pad
  Rotate
  Affine
  Perspective
  Elastic
  ColorJitter
  ColorJiggle
  ChannelShuffle
  Grayscale
  RGBShift
  GaussianBlur
  GaussianNoise
  Invert
  Posterize
  Solarize
  Sharpen
  AutoContrast
  Equalize
  Normalize
  Erasing
  JpegCompression
  RandomGamma
  MedianBlur
  MotionBlur
  CLAHE
  Brightness
  Contrast
  Blur
  ChannelDropout
  LinearIllumination
  CornerIllumination
  GaussianIllumination
  Hue
  PlankianJitter
  PlasmaBrightness
  PlasmaContrast
  PlasmaShadow
  Rain
  SaltAndPepper
  Saturation
  Snow
  OpticalDistortion
  Shear
  ThinPlateSpline
  LongestMaxSize
  SmallestMaxSize
  PhotoMetricDistort
  Colorize
  # ModeFilter # very slow, paper can live without it
  RandomJigsaw
  Transpose
  RandomRotate90
  Dithering
  UnsharpMask
)

cmd=(
  python -m benchmark.cli run
  --scenario image-rgb
  --mode micro
  --data-dir "$DATA_DIR"
  --output "$OUTPUT"
  --libraries albumentationsx torchvision kornia pillow
  --transforms "${PAPER_CORE_TRANSFORMS[@]}"
  --num-items "$NUM_ITEMS"
  --num-runs "$NUM_RUNS"
  --timer "$TIMER"
)

if [[ "$NO_REFRESH_REQUIREMENTS" == "1" ]]; then
  cmd+=(--no-refresh-requirements)
fi

cmd+=("$@")

printf 'Running RGB micro paper-core benchmark (%d transforms)\n' "${#PAPER_CORE_TRANSFORMS[@]}"
printf 'Output: %s\n' "$OUTPUT"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
