#!/bin/bash
# benchmark/run_all.sh

# Help message
show_help() {
    echo "Usage: $0 -d DATA_DIR -o OUTPUT_DIR [-n NUM_IMAGES] [-r NUM_RUNS] [-w WARMUP_RUNS]"
    echo
    echo "Required arguments:"
    echo "  -d DATA_DIR     Directory containing images"
    echo "  -o OUTPUT_DIR   Directory for output files"
    echo
    echo "Optional arguments:"
    echo "  -n NUM_IMAGES   Number of images to process (default: 1000)"
    echo "  -r NUM_RUNS     Number of benchmark runs (default: 5)"
    echo "  -w WARMUP_RUNS  Number of warmup runs (default: 10)"
    echo "  -h             Show this help message"
}

# Parse command line arguments
while getopts "d:o:n:r:w:h" opt; do
    case $opt in
        d) DATA_DIR="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        n) NUM_IMAGES="$OPTARG";;
        r) NUM_RUNS="$OPTARG";;
        w) WARMUP_RUNS="$OPTARG";;
        h) show_help; exit 0;;
        ?) show_help; exit 1;;
    esac
done

# Validate required arguments
if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    show_help
    exit 1
fi

# Set default values for optional arguments
NUM_IMAGES=${NUM_IMAGES:-1000}
NUM_RUNS=${NUM_RUNS:-5}
WARMUP_RUNS=${WARMUP_RUNS:-10}

# Libraries to benchmark
LIBRARIES=("albumentations" "imgaug" "torchvision" "kornia" "augly")

# Run benchmarks for each library
for library in "${LIBRARIES[@]}"; do
    echo "Running benchmark for ${library}..."
    ./run_single.sh \
        -l "$library" \
        -d "$DATA_DIR" \
        -o "$OUTPUT_DIR" \
        -n "$NUM_IMAGES" \
        -r "$NUM_RUNS" \
        -w "$WARMUP_RUNS"
done

echo "All benchmarks complete. Results saved in: $OUTPUT_DIR"
