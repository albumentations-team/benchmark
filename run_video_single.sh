#!/bin/bash
# benchmark/run_video_single.sh
#
# Script to run video augmentation benchmarks.
# Requires a Python file defining LIBRARY and CUSTOM_TRANSFORMS.
#
# Examples:
#   # Run with custom transforms and custom output filename
#   ./run_video_single.sh -d /path/to/videos -o output/custom_transforms_albu_2.0.8.json -s my_transforms.py
#
#   # Run with example transforms
#   ./run_video_single.sh -d /path/to/videos -o output/example_results.json -s example_direct_transforms.py

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Help message
show_help() {
    echo "Usage: $0 -d DATA_DIR -o OUTPUT_FILE -s SPECS_FILE [-n NUM_VIDEOS] [-r NUM_RUNS] [--max-warmup MAX_WARMUP] [--warmup-window WINDOW] [--warmup-threshold THRESHOLD] [--min-warmup-windows MIN_WINDOWS]"
    echo
    echo "Required arguments:"
    echo "  -d DATA_DIR     Directory containing videos"
    echo "  -o OUTPUT_FILE  Path to output JSON file"
    echo "  -s SPECS_FILE   Python file with transforms"
    echo "                  Must define LIBRARY, __call__, and TRANSFORMS"
    echo "                  See examples/*.py for examples"
    echo
    echo "Optional arguments:"
    echo "  -n NUM_VIDEOS   Number of videos to process (default: 50)"
    echo "  -r NUM_RUNS     Number of benchmark runs (default: 5)"
    echo "  --max-warmup    Maximum warmup iterations (default: 100)"
    echo "  --warmup-window Window size for variance check (default: 5)"
    echo "  --warmup-threshold Variance stability threshold (default: 0.05)"
    echo "  --min-warmup-windows Minimum windows to check (default: 3)"
    echo "  -h             Show this help message"
    echo
    echo "Examples:"
    echo "  # Run with custom output filename:"
    echo "  $0 -d /path/to/videos -o output/my_results.json -s my_transforms.py"
    echo
    echo "  # Run with example transforms:"
    echo "  $0 -d /path/to/videos -o output/albu_2.0.8_results.json -s example_direct_transforms.py"
}

# Parse command line arguments
while getopts "d:o:n:r:s:h-:" opt; do
    case "${opt}" in
        -)
            case "${OPTARG}" in
                max-warmup)
                    MAX_WARMUP="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                warmup-window)
                    WARMUP_WINDOW="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                warmup-threshold)
                    WARMUP_THRESHOLD="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                min-warmup-windows)
                    MIN_WARMUP_WINDOWS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                *)
                    echo "Invalid option: --${OPTARG}" >&2
                    show_help
                    exit 1
                    ;;
            esac;;
        d) DATA_DIR="$OPTARG";;
        o) OUTPUT_FILE="$OPTARG";;
        n) NUM_VIDEOS="$OPTARG";;
        r) NUM_RUNS="$OPTARG";;
        s) SPECS_FILE="$OPTARG";;
        h) show_help; exit 0;;
        ?) show_help; exit 1;;
    esac
done

# Validate required arguments
if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_FILE" ] || [ -z "$SPECS_FILE" ]; then
    echo "Error: Missing required arguments"
    show_help
    exit 1
fi

# Validate specs file
if [ ! -f "$SPECS_FILE" ]; then
    echo "Error: Specs file not found: $SPECS_FILE"
    exit 1
fi
echo "Using transforms from: $SPECS_FILE"

# Validate output file has .json extension
if [[ ! "$OUTPUT_FILE" =~ \.json$ ]]; then
    echo "Error: Output file must have .json extension"
    exit 1
fi

# Set default values for optional arguments
NUM_VIDEOS=${NUM_VIDEOS:-1000}
NUM_RUNS=${NUM_RUNS:-5}
MAX_WARMUP=${MAX_WARMUP:-100}
WARMUP_WINDOW=${WARMUP_WINDOW:-5}
WARMUP_THRESHOLD=${WARMUP_THRESHOLD:-0.05}
MIN_WARMUP_WINDOWS=${MIN_WARMUP_WINDOWS:-3}

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Extract library name from specs file
LIBRARY=$(python -c "
import sys
sys.path.insert(0, '$(dirname $SPECS_FILE)')
spec = __import__('$(basename $SPECS_FILE .py)')
print(spec.LIBRARY)
" 2>/dev/null)

if [ -z "$LIBRARY" ]; then
    echo "Error: Could not read LIBRARY from $SPECS_FILE"
    exit 1
fi

echo "Library: $LIBRARY"

# Create and activate virtual environment
echo "Creating virtual environment for ${LIBRARY}..."
python -m venv "${SCRIPT_DIR}/.venv_${LIBRARY}_video"

# Handle different platforms
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source "${SCRIPT_DIR}/.venv_${LIBRARY}_video/Scripts/activate"
else
    source "${SCRIPT_DIR}/.venv_${LIBRARY}_video/bin/activate"
fi

# Install requirements
echo "Installing requirements..."
pip install -U uv
uv pip install setuptools
uv pip install -U -r "${SCRIPT_DIR}/requirements/requirements.txt"

# Install library-specific requirements
if [ "$LIBRARY" == "albumentations" ]; then
    # For albumentations, use the default requirements file
    uv pip install -U --force-reinstall -r "${SCRIPT_DIR}/requirements/${LIBRARY}.txt"
else
    # For video libraries (torchvision, kornia), use the video-specific requirements files
    uv pip install -U --force-reinstall -r "${SCRIPT_DIR}/requirements/${LIBRARY}-video.txt"
fi

# Run benchmark
echo "Running video benchmark..."

# Build command
CMD="python -m benchmark.video_runner"
CMD="$CMD -d $DATA_DIR"
CMD="$CMD -o $OUTPUT_FILE"
CMD="$CMD -s $SPECS_FILE"
CMD="$CMD -n $NUM_VIDEOS"
CMD="$CMD -r $NUM_RUNS"
CMD="$CMD --max-warmup $MAX_WARMUP"
CMD="$CMD --warmup-window $WARMUP_WINDOW"
CMD="$CMD --warmup-threshold $WARMUP_THRESHOLD"
CMD="$CMD --min-warmup-windows $MIN_WARMUP_WINDOWS"

# Execute the command
eval $CMD

# Deactivate virtual environment
deactivate

echo "Video benchmark complete. Results saved to: $OUTPUT_FILE"
echo "Transforms used from: $SPECS_FILE"
echo "To analyze parametric results, run:"
echo "  python tools/analyze_parametric_results.py $OUTPUT_FILE"
