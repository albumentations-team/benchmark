#!/bin/bash
# benchmark/run_single.sh
#
# Script to run image augmentation benchmarks.
# Requires a Python file defining LIBRARY and TRANSFORMS.
#
# Examples:
#   # Run with custom transforms and custom output filename
#   ./run_single.sh -d /path/to/images -o output/custom_transforms.json -s my_transforms.py
#
#   # Run with built-in transforms
#   ./run_single.sh -d /path/to/images -o output/albumentationsx_results.json -s benchmark/transforms/albumentationsx_impl.py

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Help message
show_help() {
    echo "Usage: $0 -d DATA_DIR -o OUTPUT_FILE -s SPECS_FILE [-n NUM_IMAGES] [-r NUM_RUNS] [--max-warmup MAX_WARMUP] [--warmup-window WINDOW] [--warmup-threshold THRESHOLD] [--min-warmup-windows MIN_WINDOWS]"
    echo
    echo "Required arguments:"
    echo "  -d DATA_DIR     Directory containing images"
    echo "  -o OUTPUT_FILE  Path to output JSON file"
    echo "  -s SPECS_FILE   Python file with transforms"
    echo "                  Must define LIBRARY, __call__, and TRANSFORMS"
    echo "                  See benchmark/transforms/*_impl.py for examples"
    echo
    echo "Optional arguments:"
    echo "  -n NUM_IMAGES   Number of images to process (default: 1000)"
    echo "  -r NUM_RUNS     Number of benchmark runs (default: 5)"
    echo "  --max-warmup    Maximum warmup iterations (default: 1000)"
    echo "  --warmup-window Window size for variance check (default: 20)"
    echo "  --warmup-threshold Variance stability threshold (default: 0.05)"
    echo "  --min-warmup-windows Minimum windows to check (default: 3)"
    echo "  -h             Show this help message"
    echo
    echo "Examples:"
    echo "  # Run with built-in transforms:"
    echo "  $0 -d /path/to/images -o output/albumentationsx_results.json -s benchmark/transforms/albumentationsx_impl.py"
    echo
    echo "  # Run with custom transforms:"
    echo "  $0 -d /path/to/images -o output/my_results.json -s my_transforms.py"
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
        n) NUM_IMAGES="$OPTARG";;
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
NUM_IMAGES=${NUM_IMAGES:-1000}
NUM_RUNS=${NUM_RUNS:-5}
MAX_WARMUP=${MAX_WARMUP:-1000}
WARMUP_WINDOW=${WARMUP_WINDOW:-20}
WARMUP_THRESHOLD=${WARMUP_THRESHOLD:-0.05}
MIN_WARMUP_WINDOWS=${MIN_WARMUP_WINDOWS:-3}

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Extract library name from specs file using AST parsing to avoid import errors
LIBRARY=$(python "${SCRIPT_DIR}/extract_library_name.py" "$SPECS_FILE" 2>/dev/null)

if [ -z "$LIBRARY" ]; then
    echo "Error: Could not read LIBRARY from $SPECS_FILE"
    echo "Make sure the file defines LIBRARY = \"library_name\""
    exit 1
fi

echo "Library: $LIBRARY"

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
uv pip install -U --force-reinstall -r "${SCRIPT_DIR}/requirements/${LIBRARY}.txt"

# Run benchmark
echo "Running benchmark..."

# Build command
CMD="python -m benchmark.runner"
CMD="$CMD -d $DATA_DIR"
CMD="$CMD -o $OUTPUT_FILE"
CMD="$CMD -s $SPECS_FILE"
CMD="$CMD -n $NUM_IMAGES"
CMD="$CMD -r $NUM_RUNS"
CMD="$CMD --max-warmup $MAX_WARMUP"
CMD="$CMD --warmup-window $WARMUP_WINDOW"
CMD="$CMD --warmup-threshold $WARMUP_THRESHOLD"
CMD="$CMD --min-warmup-windows $MIN_WARMUP_WINDOWS"

# Execute the command
eval $CMD

# Deactivate virtual environment
deactivate

echo "Benchmark complete. Results saved to: $OUTPUT_FILE"
echo "Transforms used from: $SPECS_FILE"
echo "To analyze results, run:"
echo "  python tools/compare_results.py $OUTPUT_FILE"
