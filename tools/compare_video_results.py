#!/usr/bin/env python3
"""Compare video benchmark results and generate a markdown table."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    """Generate a summary of metadata for each library."""
    metadata_summary = []
    for library, result in results.items():
        if "metadata" in result:
            metadata_summary.append(f"## {library.capitalize()} Metadata\n")
            metadata_summary.append("```")
            for key, value in result["metadata"].items():
                if isinstance(value, dict):
                    metadata_summary.append(f"{key}:")
                    for subkey, subvalue in value.items():
                        metadata_summary.append(f"  {subkey}: {subvalue}")
                else:
                    metadata_summary.append(f"{key}: {value}")
            metadata_summary.append("```\n")
    return "\n".join(metadata_summary)


def update_readme(readme_path: Path, content: str, start_marker: str, end_marker: str) -> None:
    """Update a section of the README file between markers"""
    auto_generated_comment = "<!-- This file is auto-generated. Do not edit directly. -->\n\n"

    if not readme_path.exists():
        # Create a new README if it doesn't exist
        readme_path.write_text(f"{auto_generated_comment}{start_marker}\n{content}\n{end_marker}")
        return

    # Read the existing README
    readme_content = readme_path.read_text()

    # Add auto-generated comment at the top if it doesn't exist
    if not readme_content.startswith(auto_generated_comment.strip()):
        readme_content = auto_generated_comment + readme_content

    # Find the section to update
    start_index = readme_content.find(start_marker)
    end_index = readme_content.find(end_marker)

    if start_index == -1 or end_index == -1:
        # If markers not found, append to the end
        readme_content += f"\n\n{start_marker}\n{content}\n{end_marker}"
    else:
        # Replace the section between markers
        readme_content = (
            readme_content[: start_index + len(start_marker)] + "\n" + content + "\n" + readme_content[end_index:]
        )

    # Write the updated README
    readme_path.write_text(readme_content)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare video benchmark results")
    parser.add_argument("-r", "--results-dir", required=True, help="Directory containing benchmark results")
    parser.add_argument("-o", "--output", required=True, help="Output markdown file")
    parser.add_argument("--update-readme", type=Path, help="Path to README file to update with results")
    parser.add_argument(
        "--start-marker",
        default="<!-- BENCHMARK_RESULTS_START -->",
        help="Marker for start of results section in README",
    )
    parser.add_argument(
        "--end-marker",
        default="<!-- BENCHMARK_RESULTS_END -->",
        help="Marker for end of results section in README",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output)

    results = load_results(results_dir)
    table = generate_comparison_table(results)
    metadata = get_metadata_summary(results)

    # Create full report with auto-generated comment
    full_report = f"""<!-- This file is auto-generated. Do not edit directly. -->

# Video Benchmark Results

Number shows how many videos per second can be processed. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

{table}

{metadata}
"""

    # Write to file
    with output_file.open("w") as f:
        f.write(full_report)

    logger.info(f"Results written to {output_file}")

    # Update README if requested
    if args.update_readme:
        update_readme(args.update_readme, full_report, args.start_marker, args.end_marker)
        logger.info(f"Updated {args.update_readme}")


if __name__ == "__main__":
    main()
