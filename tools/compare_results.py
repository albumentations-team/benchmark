#!/usr/bin/env python3
import json
import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_results(file_path: Path) -> tuple[str, dict[str, float], dict[str, float], str]:
    """Load results from a JSON file and extract mean and std throughputs and version"""
    with file_path.open() as f:
        data = json.load(f)

    library = file_path.stem.replace("_results", "")
    medians = {}
    stds = {}
    version = data["metadata"]["library_versions"].get(library, "N/A")

    for transform_name, results in data["results"].items():
        # Strip parameters from transform name
        transform_name_stripped = transform_name.split("(")[0].strip()
        if results["supported"]:
            medians[transform_name_stripped] = results["median_throughput"]
            stds[transform_name_stripped] = results["std_throughput"]

    return library, medians, stds, version


def create_comparison_table(results_dir: Path) -> pd.DataFrame:
    """Create a comparison table from all result files in the directory"""
    result_files = list(results_dir.glob("*_results.json"))

    # First create a DataFrame with just the median values for comparison
    medians_data: dict[str, dict[str, float]] = {}
    versions: dict[str, str] = {}
    stds_data: dict[str, dict[str, float]] = {}

    for file_path in result_files:
        library, medians, stds, version = load_results(file_path)
        medians_data[library] = medians
        stds_data[library] = stds
        versions[library] = version

    # Convert to DataFrames for easier manipulation
    df_medians = pd.DataFrame(medians_data)
    df_stds = pd.DataFrame(stds_data)

    # Sort DataFrame index (transform names) alphabetically
    df_medians = df_medians.sort_index()
    df_stds = df_stds.loc[df_medians.index]  # Ensure stds are sorted the same way

    # Filter out transforms that are only supported by one library
    support_count = df_medians.notna().sum(axis=1)
    df_medians = df_medians[support_count > 1]
    df_stds = df_stds[support_count > 1]

    # Find maximum values in each row
    max_values = df_medians.max(axis=1)

    # Calculate speedup: Albumentationsx / fastest among other libraries
    speedups = []
    for idx in df_medians.index:
        if "albumentationsx" in df_medians.columns and not pd.isna(df_medians.loc[idx, "albumentationsx"]):
            # Get all libraries except albumentationsx
            other_libs = [col for col in df_medians.columns if col != "albumentationsx"]
            # Filter out NaN values
            other_values = [df_medians.loc[idx, lib] for lib in other_libs if not pd.isna(df_medians.loc[idx, lib])]

            if other_values:  # If there are other libraries with this transform
                fastest_other = max(other_values)
                speedup = df_medians.loc[idx, "albumentationsx"] / fastest_other
                speedups.append(f"{speedup:.2f}x")
            else:
                speedups.append("N/A")
        else:
            speedups.append("N/A")

    # Create the final formatted DataFrame
    formatted_data = {"Transform": df_medians.index}

    for library in sorted(df_medians.columns):
        column_values = []
        for idx in df_medians.index:
            if pd.isna(df_medians.loc[idx, library]):
                value = "-"
            else:
                median = df_medians.loc[idx, library]
                std = df_stds.loc[idx, library]
                # Bold if it's the maximum value
                value = f"**{median:.0f} ± {std:.0f}**" if median == max_values[idx] else f"{median:.0f} ± {std:.0f}"
            column_values.append(value)

        formatted_data[f"{library}<br>{versions[library]}"] = column_values

    # Add speedup column
    formatted_data["Speedup<br>(Albx/fastest other)"] = speedups

    return pd.DataFrame(formatted_data)


def get_system_summary(results_dir: Path) -> str:
    """Extract and format system information from any result file"""
    result_files = list(results_dir.glob("*_results.json"))
    if not result_files:
        return "No result files found"

    # Use first file for system info (should be same across all)
    with result_files[0].open() as f:
        data = json.load(f)

    metadata = data["metadata"]
    sys_info = metadata["system_info"]
    bench_params = metadata["benchmark_params"]

    summary = [
        "### System Information",
        "",
        f"- Platform: {sys_info['platform']}",
        f"- Processor: {sys_info['processor']}",
        f"- CPU Count: {sys_info['cpu_count']}",
        f"- Python Version: {sys_info['python_version'].split()[0]}",
        "",
    ]

    summary.extend(
        [
            "### Benchmark Parameters",
            "",
            f"- Number of images: {bench_params['num_images']}",
            f"- Runs per transform: {bench_params['num_runs']}",
            f"- Max warmup iterations: {bench_params['max_warmup_iterations']}",
            "",
        ],
    )

    # Add library versions
    summary.extend(
        [
            "",
            "### Library Versions",
            "",
        ],
    )

    # Collect versions from all result files
    versions = {}
    for file_path in result_files:
        with file_path.open() as f:
            data = json.load(f)
        library = file_path.stem.replace("_results", "")
        version = data["metadata"]["library_versions"].get(library, "N/A")
        versions[library] = version

    for library, version in sorted(versions.items()):
        summary.append(f"- {library}: {version}")

    return "\n".join(summary)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparison table from benchmark results")
    parser.add_argument("-r", "--results-dir", type=Path, help="Directory containing benchmark result JSON files")
    parser.add_argument("--update-readme", type=Path, help="Path to README file to update with results")

    args = parser.parse_args()

    # Create comparison table
    df = create_comparison_table(args.results_dir)
    markdown_table = df.to_markdown(index=False)

    # Combine summary and table with auto-generated comment
    full_report = f"""<!-- This file is auto-generated. Do not edit directly. -->

# Image Augmentation Benchmarks

This directory contains benchmark results for image augmentation libraries.

## Overview

The image benchmarks measure the performance of various image augmentation libraries on standard image
transformations. The benchmarks are run on a single CPU thread to ensure consistent and comparable results.

## Methodology

1. **Image Loading**: Images are loaded using library-specific loaders to ensure optimal format compatibility:
   - OpenCV (BGR → RGB) for Albumentationsx and imgaug
   - torchvision for PyTorch-based operations
   - PIL for augly
   - Normalized tensors for Kornia

2. **Warmup Phase**:
   - Performs adaptive warmup until performance variance stabilizes
   - Uses configurable parameters for stability detection
   - Implements early stopping for slow transforms
   - Maximum time limits prevent hanging on problematic transforms

3. **Measurement Phase**:
   - Multiple runs of each transform
   - Measures throughput (images/second)
   - Calculates statistical metrics (median, standard deviation)

4. **Environment Control**:
   - Forces single-threaded execution across libraries
   - Captures detailed system information and library versions
   - Monitors thread settings for various numerical libraries

## Running the Benchmarks

To run the image benchmarks:

```bash
./run_single.sh -l albumentationsx -d /path/to/images -o /path/to/output
```

Number shows how many uint8 images per second can be processed on one CPU thread. Larger is better.
The Speedup column shows how many times faster Albumentationsx is compared to the fastest other
library for each transform.

{markdown_table}

## Analysis

The benchmark results show that Albumentationsx is generally the fastest library for most image
transformations. This is due to its optimized implementation and use of OpenCV for many operations.

Some key observations:
- Albumentationsx is particularly fast for geometric transformations like resize, rotate, and affine
- For some specialized transformations, other libraries may be faster
- The performance gap is most significant for complex transformations

## Recommendations

Based on the benchmark results, we recommend:

1. Use Albumentationsx for production workloads where performance is critical
2. Consider the specific transformations you need and check their relative performance
3. For GPU acceleration, consider Kornia, especially for batch processing
"""

    # Overwrite README if requested
    if args.update_readme:
        args.update_readme.write_text(full_report)
        logger.info(f"Overwrote {args.update_readme} with new content.")

    # Log preview
    logger.info("\nBenchmark Report Preview:")
    logger.info(full_report)


if __name__ == "__main__":
    main()
