#!/usr/bin/env python3
"""Compare video benchmark results and generate a markdown table."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all video benchmark results from the directory."""
    results = {}
    for file_path in results_dir.glob("*_video_results.json"):
        library = file_path.stem.replace("_video_results", "")
        with open(file_path, "r") as f:
            results[library] = json.load(f)
    return results


def format_throughput(throughput: float, std: Optional[float] = None, is_max: bool = False) -> str:
    """Format throughput value with optional standard deviation and bold if it's the max value."""
    formatted = f"{throughput:.2f}"
    if std is not None:
        formatted = f"{formatted} ± {std:.2f}"

    if is_max:
        formatted = f"**{formatted}**"

    return formatted


def format_time(time_ms: float, std: Optional[float] = None) -> str:
    """Format time value with optional standard deviation."""
    return f"{time_ms:.2f} ± {std:.2f}" if std is not None else f"{time_ms:.2f}"


def get_hardware_info(results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
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
                    cpu_count = result["metadata"]["system_info"].get("cpu_count", "1")
                    hardware_info[library] = f"{cpu_info} ({cpu_count} cores)"

                # For GPU (Kornia)
                elif library.lower() == "kornia" and "pytorch" in thread_settings:
                    # Handle the case where pytorch settings are stored as a string
                    pytorch_settings = thread_settings["pytorch"]
                    if isinstance(pytorch_settings, str):
                        # Try to extract GPU info from the string
                        if "gpu_available': True" in pytorch_settings:
                            # Extract GPU name if available
                            if "gpu_name':" in pytorch_settings:
                                import re
                                gpu_name_match = re.search(r"gpu_name': '([^']+)'", pytorch_settings)
                                if gpu_name_match:
                                    gpu_name = gpu_name_match.group(1)
                                    hardware_info[library] = gpu_name
                                else:
                                    # Extract GPU device if name not available
                                    gpu_device_match = re.search(r"gpu_device': ([^,}]+)", pytorch_settings)
                                    if gpu_device_match:
                                        gpu_device = gpu_device_match.group(1)
                                        hardware_info[library] = f"GPU {gpu_device}"
                                    else:
                                        hardware_info[library] = "GPU (details unknown)"
                            else:
                                hardware_info[library] = "GPU (details unknown)"
                        else:
                            hardware_info[library] = "CPU (GPU not available)"
                    else:
                        # Handle the case where pytorch settings are a dictionary
                        if pytorch_settings.get("gpu_available", False):
                            gpu_name = pytorch_settings.get("gpu_name", None)
                            if gpu_name:
                                hardware_info[library] = gpu_name
                            else:
                                gpu_device = pytorch_settings.get("gpu_device", "Unknown")
                                hardware_info[library] = f"GPU {gpu_device}"
                        else:
                            hardware_info[library] = "CPU (GPU not available)"
                else:
                    hardware_info[library] = "Unknown hardware"
            else:
                # Default hardware info if thread_settings not found
                if library.lower() == "albumentations":
                    hardware_info[library] = "CPU (1 core)"
                elif library.lower() == "kornia":
                    hardware_info[library] = "GPU"
                else:
                    hardware_info[library] = "Unknown"
        else:
            # Default hardware info if metadata not found
            if library.lower() == "albumentations":
                hardware_info[library] = "CPU (1 core)"
            elif library.lower() == "kornia":
                hardware_info[library] = "GPU"
            else:
                hardware_info[library] = "Unknown"
    return hardware_info


def generate_comparison_table(results: Dict[str, Dict[str, Any]]) -> str:
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
        clean_name = transform.split('(')[0].strip()
        clean_transforms[transform] = clean_name

    # Sort transforms for consistent ordering
    all_transforms = sorted(all_transforms, key=lambda x: clean_transforms[x])

    # Get list of libraries
    libraries = sorted(results.keys())

    # Get hardware info
    hardware_info = get_hardware_info(results)

    # Create table header with hardware info
    header = "| Transform | " + " | ".join(f"{lib} (videos per second)<br>{hardware_info.get(lib, 'Unknown')}" for lib in libraries) + " | Speedup<br>(Alb/fastest other) |\n"

    # Create table separator
    separator = "|" + "|".join("---" for _ in range(len(libraries) + 2)) + "|\n"

    # Create table rows
    rows = []
    for transform in all_transforms:
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
        for lib, throughput, std in throughputs:
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
            row += f" {speedup:.2f}× |"
        else:
            row += " N/A |"

        rows.append(row)

    # Combine all parts of the table
    return header + separator + "\n".join(rows)


def get_metadata_summary(results: Dict[str, Dict[str, Any]]) -> str:
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
    if not readme_path.exists():
        # Create a new README if it doesn't exist
        readme_path.write_text(f"{start_marker}\n{content}\n{end_marker}")
        return

    # Read the existing README
    readme_content = readme_path.read_text()

    # Find the section to update
    start_index = readme_content.find(start_marker)
    end_index = readme_content.find(end_marker)

    if start_index == -1 or end_index == -1:
        # If markers not found, append to the end
        readme_content += f"\n\n{start_marker}\n{content}\n{end_marker}"
    else:
        # Replace the section between markers
        readme_content = (
            readme_content[:start_index + len(start_marker)]
            + "\n"
            + content
            + "\n"
            + readme_content[end_index:]
        )

    # Write the updated README
    readme_path.write_text(readme_content)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare video benchmark results")
    parser.add_argument("-r", "--results-dir", required=True, help="Directory containing benchmark results")
    parser.add_argument("-o", "--output", required=True, help="Output markdown file")
    parser.add_argument("--update-readme", type=Path, help="Path to README file to update with results")
    parser.add_argument("--start-marker", default="<!-- BENCHMARK_RESULTS_START -->", help="Marker for start of results section in README")
    parser.add_argument("--end-marker", default="<!-- BENCHMARK_RESULTS_END -->", help="Marker for end of results section in README")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output)

    results = load_results(results_dir)
    table = generate_comparison_table(results)
    metadata = get_metadata_summary(results)

    # Create full report
    full_report = f"""# Video Benchmark Results

Number shows how many videos per second can be processed. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other library for each transform.

{table}

{metadata}
"""

    # Write to file
    with open(output_file, "w") as f:
        f.write(full_report)

    print(f"Results written to {output_file}")

    # Update README if requested
    if args.update_readme:
        update_readme(args.update_readme, full_report, args.start_marker, args.end_marker)
        print(f"Updated {args.update_readme}")


if __name__ == "__main__":
    main()
