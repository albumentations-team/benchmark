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

    # Find maximum values in each row
    max_values = df_medians.max(axis=1)

    # Calculate speedup: Albumentations / fastest among other libraries
    speedups = []
    for idx in df_medians.index:
        if "albumentations" in df_medians.columns and not pd.isna(df_medians.loc[idx, "albumentations"]):
            # Get all libraries except albumentations
            other_libs = [col for col in df_medians.columns if col != "albumentations"]
            # Filter out NaN values
            other_values = [df_medians.loc[idx, lib] for lib in other_libs if not pd.isna(df_medians.loc[idx, lib])]

            if other_values:  # If there are other libraries with this transform
                fastest_other = max(other_values)
                speedup = df_medians.loc[idx, "albumentations"] / fastest_other
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
    formatted_data["Speedup<br>(Alb/fastest other)"] = speedups

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

    # Extract GPU information from each library's results
    gpu_info = {}
    for file_path in result_files:
        with file_path.open() as f:
            lib_data = json.load(f)

        library = file_path.stem.replace("_results", "")
        lib_metadata = lib_data.get("metadata", {})

        # Check for GPU info in thread_settings
        if "thread_settings" in lib_metadata:
            thread_settings = lib_metadata["thread_settings"]

            # For GPU-based libraries (Kornia, TorchVision)
            if "pytorch" in thread_settings:
                pytorch_settings = thread_settings["pytorch"]

                # Handle string representation
                if isinstance(pytorch_settings, str):
                    import re

                    # Check if GPU is available
                    if "gpu_available': True" in pytorch_settings:
                        # Try to extract GPU name
                        gpu_name_match = re.search(r"gpu_name': '([^']+)'", pytorch_settings)
                        if gpu_name_match:
                            gpu_info[library] = gpu_name_match.group(1)
                        else:
                            # Try to extract GPU device
                            gpu_device_match = re.search(r"gpu_device': ([^,}]+)", pytorch_settings)
                            if gpu_device_match:
                                gpu_info[library] = f"GPU {gpu_device_match.group(1)}"
                            else:
                                gpu_info[library] = "GPU (details unknown)"
                # Handle dictionary representation
                elif isinstance(pytorch_settings, dict):
                    if pytorch_settings.get("gpu_available", False):
                        gpu_name = pytorch_settings.get("gpu_name")
                        if gpu_name:
                            gpu_info[library] = gpu_name
                        else:
                            gpu_device = pytorch_settings.get("gpu_device", "Unknown")
                            gpu_info[library] = f"GPU {gpu_device}"

    # Add GPU information if available
    if gpu_info:
        summary.append("### GPU Information")
        summary.append("")
        for library, gpu in sorted(gpu_info.items()):
            summary.append(f"- {library.capitalize()}: {gpu}")
        summary.append("")

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
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparison table from benchmark results")
    parser.add_argument("-r", "--results-dir", type=Path, help="Directory containing benchmark result JSON files")
    parser.add_argument("-o", "--output", type=Path, help="Output markdown file path")
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

    # Get system summary
    system_summary = get_system_summary(args.results_dir)

    # Create comparison table
    df = create_comparison_table(args.results_dir)
    markdown_table = df.to_markdown(index=False)

    # Save CSV version
    if args.output:
        df.to_csv(args.output.with_suffix(".csv"), index=False)

    # Combine summary and table with auto-generated comment
    full_report = f"""<!-- This file is auto-generated. Do not edit directly. -->

# Image Benchmark Results

{system_summary}

## Performance Comparison

Number shows how many uint8 images per second can be processed on one CPU thread. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

{markdown_table}
"""

    # Save to file
    if args.output:
        args.output.write_text(full_report)

    # Update README if requested
    if args.update_readme:
        update_readme(args.update_readme, full_report, args.start_marker, args.end_marker)
        logger.info(f"Updated {args.update_readme}")

    # Log preview
    logger.info("\nBenchmark Report Preview:")
    logger.info(full_report)


if __name__ == "__main__":
    main()
