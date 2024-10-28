#!/bin/bash
# benchmark/run_single.sh

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Help message
show_help() {
    echo "Usage: $0 -l LIBRARY -d DATA_DIR -o OUTPUT_DIR [-n NUM_IMAGES] [-r NUM_RUNS] [--max-warmup MAX_WARMUP] [--warmup-window WINDOW] [--warmup-threshold THRESHOLD] [--min-warmup-windows MIN_WINDOWS]"
    echo
    echo "Required arguments:"
    echo "  -l LIBRARY      Library to benchmark (albumentations, imgaug, torchvision, kornia, augly)"
    echo "  -d DATA_DIR     Directory containing images"
    echo "  -o OUTPUT_DIR   Directory for output files"
    echo
    echo "Optional arguments:"
    echo "  -n NUM_IMAGES   Number of images to process (default: 1000)"
    echo "  -r NUM_RUNS     Number of benchmark runs (default: 5)"
    echo "  --max-warmup    Maximum warmup iterations (default: 5000)"
    echo "  --warmup-window Window size for variance check (default: 5)"
    echo "  --warmup-threshold Variance stability threshold (default: 0.05)"
    echo "  --min-warmup-windows Minimum windows to check (default: 3)"
    echo "  -h             Show this help message"
}

# Parse command line arguments
while getopts "l:d:o:n:r:h-:" opt; do
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
        l) LIBRARY="$OPTARG";;
        d) DATA_DIR="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        n) NUM_IMAGES="$OPTARG";;
        r) NUM_RUNS="$OPTARG";;
        h) show_help; exit 0;;
        ?) show_help; exit 1;;
    esac
done

# Validate required arguments
if [ -z "$LIBRARY" ] || [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    show_help
    exit 1
fi

# Set default values for optional arguments
NUM_IMAGES=${NUM_IMAGES:-1000}
NUM_RUNS=${NUM_RUNS:-5}
MAX_WARMUP=${MAX_WARMUP:-1000}
WARMUP_WINDOW=${WARMUP_WINDOW:-20}
WARMUP_THRESHOLD=${WARMUP_THRESHOLD:-0.05}
MIN_WARMUP_WINDOWS=${MIN_WARMUP_WINDOWS:-3}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create and activate virtual environment
echo "Creating virtual environment for ${LIBRARY}..."
python -m venv "${SCRIPT_DIR}/.venv_${LIBRARY}"

# Handle different platforms
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source "${SCRIPT_DIR}/.venv_${LIBRARY}/Scripts/activate"
else
    source "${SCRIPT_DIR}/.venv_${LIBRARY}/bin/activate"
fi

# Install requirements
echo "Installing requirements..."
pip install -U uv
uv pip install setuptools
uv pip install -U -r "${SCRIPT_DIR}/requirements/requirements.txt"
uv pip install -U -r "${SCRIPT_DIR}/requirements/${LIBRARY}.txt"

# Run benchmark
echo "Running benchmark..."
python -m benchmark.runner \
    -l "$LIBRARY" \
    -d "$DATA_DIR" \
    -o "${OUTPUT_DIR}/${LIBRARY}_results.json" \
    -n "$NUM_IMAGES" \
    -r "$NUM_RUNS" \
    --max-warmup "$MAX_WARMUP" \
    --warmup-window "$WARMUP_WINDOW" \
    --warmup-threshold "$WARMUP_THRESHOLD" \
    --min-warmup-windows "$MIN_WARMUP_WINDOWS"

# Deactivate virtual environment
deactivate

echo "Benchmark complete. Results saved to: ${OUTPUT_DIR}/${LIBRARY}_results.json"
