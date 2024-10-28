from pathlib import Path
import json
import pandas as pd
import numpy as np

def load_results(file_path: Path) -> tuple[str, dict[str, float], dict[str, float], str]:
    """Load results from a JSON file and extract mean and std throughputs and version"""
    with open(file_path) as f:
        data = json.load(f)

    library = file_path.stem.replace("_results", "")
    medians = {}
    stds = {}
    version = data["metadata"]["library_versions"].get(library, "N/A")

    for transform_name, results in data["results"].items():
        transform_name = transform_name.split('(')[0].strip()
        if results["supported"]:
            medians[transform_name] = results["median_throughput"]
            stds[transform_name] = results["std_throughput"]

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
                if median == max_values[idx]:
                    value = f"**{median:.0f} ± {std:.0f}**"
                else:
                    value = f"{median:.0f} ± {std:.0f}"
            column_values.append(value)

        formatted_data[f"{library}<br>{versions[library]}"] = column_values

    return pd.DataFrame(formatted_data)

def get_system_summary(results_dir: Path) -> str:
    """Extract and format system information from any result file"""
    result_files = list(results_dir.glob("*_results.json"))
    if not result_files:
        return "No result files found"

    # Use first file for system info (should be same across all)
    with open(result_files[0]) as f:
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
        "### Benchmark Parameters",
        "",
        f"- Number of images: {bench_params['num_images']}",
        f"- Runs per transform: {bench_params['num_runs']}",
        f"- Max warmup iterations: {bench_params['max_warmup_iterations']}",
        "",
    ]

    # Add library versions
    summary.extend([
        "",
        "### Library Versions",
        "",
    ])

    # Collect versions from all result files
    versions = {}
    for file_path in result_files:
        with open(file_path) as f:
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
    parser.add_argument("-o", "--output", type=Path, help="Output markdown file path")

    args = parser.parse_args()

    # Get system summary
    system_summary = get_system_summary(args.results_dir)

    # Create comparison table
    df = create_comparison_table(args.results_dir)
    markdown_table = df.to_markdown(index=False)

    # Combine summary and table
    full_report = f"""# Benchmark Results

{system_summary}

## Performance Comparison

{markdown_table}
"""

    # Save to file
    if args.output:
        args.output.write_text(full_report)

    # Print preview
    print("\nBenchmark Report Preview:")
    print(full_report)


if __name__ == "__main__":
    main()
