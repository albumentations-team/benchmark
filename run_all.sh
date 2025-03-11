#!/bin/bash

# Help message
show_help() {
    echo "Usage: $0 -d DATA_DIR -o OUTPUT_DIR [-n NUM_IMAGES] [-r NUM_RUNS] [--max-warmup MAX_WARMUP] [--warmup-window WINDOW] [--warmup-threshold THRESHOLD] [--min-warmup-windows MIN_WINDOWS] [--update-docs]"
    echo
    echo "Required arguments:"
    echo "  -d DATA_DIR     Directory containing images"
    echo "  -o OUTPUT_DIR   Directory for output files"
    echo
    echo "Optional arguments:"
    echo "  -n NUM_IMAGES   Number of images to process (default: 2000)"
    echo "  -r NUM_RUNS     Number of benchmark runs (default: 5)"
    echo "  --max-warmup    Maximum warmup iterations (default: 1000)"
    echo "  --warmup-window Window size for variance check (default: 5)"
    echo "  --warmup-threshold Variance stability threshold (default: 0.05)"
    echo "  --min-warmup-windows Minimum windows to check (default: 10)"
    echo "  --update-docs   Update documentation with results (default: false)"
    echo "  -h             Show this help message"
}

# Default values
NUM_IMAGES=2000
NUM_RUNS=5
MAX_WARMUP=1000
WARMUP_WINDOW=5
WARMUP_THRESHOLD=0.05
MIN_WARMUP_WINDOWS=10
UPDATE_DOCS=false

# Parse command line arguments
while getopts "d:o:n:r:h-:" opt; do
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
                update-docs)
                    UPDATE_DOCS=true
                    ;;
                *)
                    echo "Unknown option --${OPTARG}"
                    show_help
                    exit 1
                    ;;
            esac
            ;;
        d)
            DATA_DIR="${OPTARG}"
            ;;
        o)
            OUTPUT_DIR="${OPTARG}"
            ;;
        n)
            NUM_IMAGES="${OPTARG}"
            ;;
        r)
            NUM_RUNS="${OPTARG}"
            ;;
        h)
            show_help
            exit 0
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments."
    show_help
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run benchmarks for each library
LIBRARIES=("albumentations" "imgaug" "torchvision" "kornia" "augly")

for lib in "${LIBRARIES[@]}"; do
    echo "Running benchmark for $lib..."
    ./run_single.sh \
        -l "$lib" \
        -d "$DATA_DIR" \
        -o "$OUTPUT_DIR" \
        -n "$NUM_IMAGES" \
        -r "$NUM_RUNS" \
        --max-warmup "$MAX_WARMUP" \
        --warmup-window "$WARMUP_WINDOW" \
        --warmup-threshold "$WARMUP_THRESHOLD" \
        --min-warmup-windows "$MIN_WARMUP_WINDOWS"
done

# Generate comparison table
echo "Generating comparison table..."
python -m tools.compare_results -r "$OUTPUT_DIR" -o"${OUTPUT_DIR}/comparison.md"

echo "All benchmarks complete."
echo "Individual results saved in: $OUTPUT_DIR"
echo "Comparison table saved as: ${OUTPUT_DIR}/comparison.md"

# Update documentation if requested
if [ "$UPDATE_DOCS" = true ]; then
    echo "Updating documentation..."
    ./tools/update_docs.sh --image-results "$OUTPUT_DIR"
fi
