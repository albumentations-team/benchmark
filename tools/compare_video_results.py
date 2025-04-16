#!/usr/bin/env python3
"""Compare video benchmark results and generate a markdown table."""

import argparse
import ast  # Added import
import json
import logging
from pathlib import Path
from typing import Any

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Helper function to safely parse string literals
def convert_literal(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (ValueError, SyntaxError):
            pass  # Keep original string if not a valid literal dict/list
    return value


# Helper function specifically for thread_settings dictionary values
def convert_thread_settings(settings: dict[str, Any]) -> dict[str, Any]:
    updated: dict[str, Any] = {}
    for key, subvalue in settings.items():
        updated[key] = convert_literal(subvalue)
    return updated


# New helper function to process a metadata dictionary
def process_metadata(metadata_dict: dict[str, Any]) -> dict[str, Any]:
    """Processes a metadata dictionary, converting string literals."""
    processed = metadata_dict.copy()
    for key, value in processed.items():
        if key == "thread_settings" and isinstance(value, dict):
            processed[key] = convert_thread_settings(value)
        else:
            processed[key] = convert_literal(value)
    return processed


# New helper function to save individual metadata YAML files
def save_metadata_files(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Save individual metadata YAML files to the given output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
    for library, result in results.items():
        if "metadata" in result:
            try:
                processed_metadata = process_metadata(result["metadata"])
                yaml_filename = f"{library}_video_metadata.yaml"
                yaml_output_path = output_dir / yaml_filename
                with yaml_output_path.open("w") as f:
                    yaml.dump(
                        processed_metadata,
                        f,
                        default_flow_style=False,
                        indent=2,
                        sort_keys=False,
                    )
                logger.info(f"Saved metadata for {library} to {yaml_output_path}")
            except Exception:
                logger.exception(f"Failed to save metadata YAML for {library}")


def load_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all video benchmark results from the directory."""
    results = {}
    for file_path in results_dir.glob("*_video_results.json"):
        library = file_path.stem.replace("_video_results", "")
        with file_path.open() as f:
            results[library] = json.load(f)
    return results


def format_throughput(throughput: float, std: float | None = None, is_max: bool = False) -> str:
    """Format throughput value with optional standard deviation and bold if it's the max value."""
    formatted = f"{throughput:.2f}"
    if std is not None:
        formatted = f"{formatted} ± {std:.2f}"

    return f"**{formatted}**" if is_max else formatted


def format_time(time_ms: float, std: float | None = None) -> str:
    """Format time value with optional standard deviation."""
    return f"{time_ms:.2f} ± {std:.2f}" if std is not None else f"{time_ms:.2f}"


def extract_gpu_info(thread_settings: dict[str, Any]) -> str:
    """Extract GPU information from thread settings."""
    if "pytorch" not in thread_settings:
        return "Unknown hardware"

    pytorch_settings = thread_settings["pytorch"]

    # Handle the case where pytorch settings are stored as a string
    if isinstance(pytorch_settings, str):
        # Try to extract GPU info from the string
        if "gpu_available': True" in pytorch_settings:
            # Extract GPU name if available
            if "gpu_name':" in pytorch_settings:
                import re

                gpu_name_match = re.search(r"gpu_name': '([^']+)'", pytorch_settings)
                if gpu_name_match:
                    return gpu_name_match.group(1)
                # Extract GPU device if name not available
                gpu_device_match = re.search(r"gpu_device': ([^,}]+)", pytorch_settings)
                if gpu_device_match:
                    return f"GPU {gpu_device_match.group(1)}"
                return "GPU (details unknown)"
            return "GPU (details unknown)"
        return "CPU (GPU not available)"
    # Handle the case where pytorch settings are a dictionary
    if pytorch_settings.get("gpu_available", False):
        gpu_name = pytorch_settings.get("gpu_name", None)
        if gpu_name:
            return gpu_name
        gpu_device = pytorch_settings.get("gpu_device", "Unknown")
        return f"GPU {gpu_device}"
    return "CPU (GPU not available)"


def get_hardware_info(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Extract hardware information from metadata."""
    hardware_info = {}
    for library, result in results.items():
        if "metadata" in result:
            # Try to extract hardware info from thread_settings
            if "thread_settings" in result["metadata"]:
                thread_settings = result["metadata"]["thread_settings"]

                # For CPU (Albumentations)
                if library.lower() == "albumentations":
                    cpu_info = result["metadata"]["system_info"].get("processor", "CPU")
                    # Always use "1 core" for albumentations as we fix CPU thread in actual benchmark
                    hardware_info[library] = f"{cpu_info} (1 core)"

                # For GPU-based libraries (Kornia, TorchVision, etc.)
                elif "pytorch" in thread_settings:
                    hardware_info[library] = extract_gpu_info(thread_settings)
                else:
                    hardware_info[library] = "Unknown hardware"
            # Default hardware info if thread_settings not found
            elif library.lower() == "albumentations":
                # Always use "CPU (1 core)" for albumentations
                cpu_info = result["metadata"]["system_info"].get("processor", "CPU")
                hardware_info[library] = f"{cpu_info} (1 core)"
            elif library.lower() in ["kornia", "torchvision"]:
                hardware_info[library] = "GPU (details unknown)"
            else:
                hardware_info[library] = "Unknown"
        # Default hardware info if metadata not found
        elif library.lower() == "albumentations":
            hardware_info[library] = "CPU (1 core)"
        elif library.lower() in ["kornia", "torchvision"]:
            hardware_info[library] = "GPU (details unknown)"
        else:
            hardware_info[library] = "Unknown"
    return hardware_info


def generate_comparison_table(results: dict[str, dict[str, Any]]) -> str:
    """Generate a markdown table comparing the results."""
    if not results:
        return "No results found."

    # Extract all transform names from all libraries
    all_transforms = set()
    for library_results in results.values():
        all_transforms.update(library_results["results"].keys())

    # Clean transform names - remove parameters in parentheses
    clean_transforms = {}
    for transform in all_transforms:
        clean_name = transform.split("(")[0].strip()
        clean_transforms[transform] = clean_name

    # Sort transforms for consistent ordering
    sorted_transforms = sorted(all_transforms, key=lambda x: clean_transforms[x])

    # Get list of libraries
    libraries = sorted(results.keys())

    # Get hardware info
    hardware_info = get_hardware_info(results)

    # Create table header with hardware info
    header = (
        "| Transform | "
        + " | ".join(f"{lib} (videos per second)<br>{hardware_info.get(lib, 'Unknown')}" for lib in libraries)
        + " | Speedup<br>(Alb/fastest other) |\n"
    )

    # Create table separator
    separator = "|" + "|".join("---" for _ in range(len(libraries) + 2)) + "|\n"

    # Create table rows
    rows = []
    for transform in sorted_transforms:
        # Use clean transform name for display
        clean_name = clean_transforms[transform]
        row = f"| {clean_name} |"

        # Get throughput values for each library
        throughputs = []
        for lib in libraries:
            lib_results = results[lib]["results"].get(transform, {})
            if lib_results.get("supported", False) and not lib_results.get("early_stopped", False):
                throughput = lib_results.get("median_throughput", 0)
                std = lib_results.get("std_throughput", 0)
                throughputs.append((lib, throughput, std))
            else:
                throughputs.append((lib, 0, 0))

        # Find the max throughput for this transform
        max_throughput = max((t for _, t, _ in throughputs if t > 0), default=0)

        # Add formatted throughput values to the row
        for _, throughput, std in throughputs:
            if throughput > 0:
                is_max = throughput == max_throughput
                row += f" {format_throughput(throughput, std, is_max)} |"
            else:
                row += " N/A |"

        # Calculate speedup: Albumentations / fastest among other libraries
        alb_throughput = next((t for lib, t, _ in throughputs if lib == "albumentations" and t > 0), 0)
        other_throughputs = [t for lib, t, _ in throughputs if lib != "albumentations" and t > 0]

        if alb_throughput > 0 and other_throughputs:
            fastest_other = max(other_throughputs)
            speedup = alb_throughput / fastest_other
            row += f" {speedup:.2f}x |"
        else:
            row += " N/A |"

        rows.append(row)

    # Combine all parts of the table
    return header + separator + "\n".join(rows)


def get_metadata_summary(results: dict[str, dict[str, Any]]) -> str:
    """Generate a summary of metadata for each library in YAML format."""
    metadata_summary: list[str] = []
    for library, result in results.items():
        if "metadata" in result:
            metadata_summary.extend((f"## {library.capitalize()} Metadata\n", "```yaml"))

            # Process the metadata first
            processed_metadata = process_metadata(result["metadata"])

            # Dump the processed metadata as YAML for the summary
            try:
                yaml_str = yaml.dump(processed_metadata, default_flow_style=False, indent=2, sort_keys=False)
                metadata_summary.append(yaml_str)
            except yaml.YAMLError:
                logger.exception(f"Error dumping metadata to YAML summary for {library}")
                metadata_summary.append(str(processed_metadata))  # Dump processed version on error

            metadata_summary.append("```\n")
    return "\n".join(metadata_summary)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare video benchmark results")
    parser.add_argument("-r", "--results-dir", required=True, help="Directory containing benchmark results")
    parser.add_argument("--update-readme", type=Path, help="Path to README file to update with results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    results = load_results(results_dir)
    table = generate_comparison_table(results)
    metadata_summary_str = get_metadata_summary(results)

    # Determine the output directory for metadata YAML files
    metadata_output_dir = None
    if args.update_readme:
        metadata_output_dir = args.update_readme.parent

    # Save metadata YAML files if an output directory is determined
    if metadata_output_dir:
        save_metadata_files(results, metadata_output_dir)
    else:
        logger.warning("Could not determine output directory for metadata YAML files. Skipping save.")

    # Create full report string, including introductory text
    full_report = f"""<!-- This file is auto-generated. Do not edit directly. -->

# Video Augmentation Benchmarks

This directory contains benchmark results for video augmentation libraries.

## Overview

The video benchmarks measure the performance of various augmentation libraries on video transformations.
The benchmarks compare CPU-based processing (Albumentations) with GPU-accelerated processing (Kornia).

## Dataset

The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains 13,320 videos from
101 action categories. The videos are realistic, collected from YouTube, and include a wide variety of camera
motion, object appearance, pose, scale, viewpoint, and background. This makes it an excellent dataset for
benchmarking video augmentation performance across diverse real-world scenarios.

You can download the dataset from: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

## Methodology

1. **Video Loading**: Videos are loaded using library-specific loaders:
   - OpenCV for Albumentations
   - PyTorch tensors for Kornia

2. **Warmup Phase**:
   - Performs adaptive warmup until performance variance stabilizes
   - Uses configurable parameters for stability detection
   - Implements early stopping for slow transforms

3. **Measurement Phase**:
   - Multiple runs of each transform
   - Measures throughput (videos/second)
   - Calculates statistical metrics (median, standard deviation)

4. **Environment Control**:
   - CPU benchmarks are run single-threaded
   - GPU benchmarks utilize the specified GPU device
   - Thread settings are controlled for consistent results

## Hardware Comparison

The benchmarks compare:
- Albumentations: CPU-based processing (single thread)
- Kornia: GPU-accelerated processing (NVIDIA GPUs)

This provides insights into the trade-offs between CPU and GPU processing for video augmentation.

## Running the Benchmarks

To run the video benchmarks:

```bash
./run_video_single.sh -l albumentations -d /path/to/videos -o /path/to/output
```

To run all libraries and generate a comparison:

```bash
./run_video_all.sh -d /path/to/videos -o /path/to/output
```

## Benchmark Results

### Video Benchmark Results

Number shows how many videos per second can be processed. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

{table}

{metadata_summary_str}

## Analysis
The benchmark results show interesting trade-offs between CPU and GPU processing:
- **CPU Advantages**:
  - Better for simple transformations with low computational complexity
  - No data transfer overhead between CPU and GPU
  - More consistent performance across different transform types
- **GPU Advantages**:
  - Significantly faster for complex transformations
  - Better scaling with video resolution
  - More efficient for batch processing

## Recommendations
Based on the benchmark results, we recommend:
1. For simple transformations on a small number of videos, CPU processing may be sufficient
2. For complex transformations or batch processing, GPU acceleration provides significant benefits
3. Consider the specific transformations you need and their relative performance on CPU vs GPU
"""

    # Overwrite README if requested
    if args.update_readme:
        args.update_readme.write_text(full_report)
        logger.info(f"Overwrote {args.update_readme} with new content.")


if __name__ == "__main__":
    main()
