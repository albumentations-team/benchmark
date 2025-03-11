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
    if std is not None:
        return f"{time_ms:.2f} ± {std:.2f}"
    return f"{time_ms:.2f}"


def get_hardware_info(results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Extract hardware information from metadata."""
    hardware_info = {}
    for library, result in results.items():
        if "metadata" in result and "hardware" in result["metadata"]:
            hardware_info[library] = result["metadata"]["hardware"]
        else:
            # Default hardware info if not found
            if library.lower() == "albumentations":
                hardware_info[library] = "CPU (1 core)"
            elif library.lower() == "kornia":
                hardware_info[library] = "GTX 4090"
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

    # Sort transforms for consistent ordering
    all_transforms = sorted(all_transforms)

    # Get list of libraries
    libraries = sorted(results.keys())

    # Get hardware info
    hardware_info = get_hardware_info(results)

    # Create table header with hardware info
    header = "| Transform | " + " | ".join(f"{lib} (videos per second)<br>{hardware_info.get(lib, 'Unknown')}" for lib in libraries) + " |\n"

    # Create table separator
    separator = "|" + "|".join("---" for _ in range(len(libraries) + 1)) + "|\n"

    # Create table rows
    rows = []
    for transform in all_transforms:
        row = f"| {transform} |"

        # Get throughput values for each library
        throughputs = []
        for lib in libraries:
            lib_results = results[lib]["results"].get(transform, {})
            if lib_results.get("supported", False) and not lib_results.get("early_stopped", False):
                throughput = lib_results.get("median_throughput", 0)
                std = lib_results.get("std_throughput", 0)
                throughputs.append((throughput, std))
            else:
                throughputs.append((0, 0))

        # Find the max throughput for this transform
        max_throughput = max((t for t, _ in throughputs if t > 0), default=0)

        # Add formatted throughput values to the row
        for i, (throughput, std) in enumerate(throughputs):
            if throughput > 0:
                is_max = throughput == max_throughput
                row += f" {format_throughput(throughput, std, is_max)} |"
            else:
                row += " N/A |"

        rows.append(row)

    # Combine all parts of the table
    return header + separator + "\n".join(rows)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare video benchmark results")
    parser.add_argument("-r", "--results-dir", required=True, help="Directory containing benchmark results")
    parser.add_argument("-o", "--output", required=True, help="Output markdown file")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output)

    results = load_results(results_dir)
    table = generate_comparison_table(results)

    # Add metadata
    metadata = []
    for library, result in results.items():
        if "metadata" in result:
            metadata.append(f"## {library.capitalize()} Metadata\n")
            metadata.append("```")
            for key, value in result["metadata"].items():
                if isinstance(value, dict):
                    metadata.append(f"{key}:")
                    for subkey, subvalue in value.items():
                        metadata.append(f"  {subkey}: {subvalue}")
                else:
                    metadata.append(f"{key}: {value}")
            metadata.append("```\n")

    # Write to file
    with open(output_file, "w") as f:
        f.write("# Video Benchmark Results\n\n")
        f.write(table)
        f.write("\n\n")
        f.write("\n".join(metadata))


if __name__ == "__main__":
    main()
