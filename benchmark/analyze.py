import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Any
import numpy as np

def load_results(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)

def create_metadata_summary(metadata: dict[str, Any], output_dir: Path) -> None:
    """Create a summary of benchmark metadata"""
    metadata_text = [
        f"Benchmark Run Summary",
        f"-------------------",
        f"Date: {metadata['system_info']['timestamp']}",
        f"Python: {metadata['system_info']['python_version'].split()[0]}",
        f"Platform: {metadata['system_info']['platform']}",
        f"CPU: {metadata['system_info']['processor']} ({metadata['system_info']['cpu_count']} cores)",
        f"\nLibrary Versions",
        f"--------------"
    ]

    for lib, version in sorted(metadata["library_versions"].items()):
        metadata_text.append(f"{lib}: {version}")

    metadata_text.extend([
        f"\nBenchmark Parameters",
        f"-------------------",
        f"Images: {metadata['benchmark_params']['num_images']}",
        f"Runs: {metadata['benchmark_params']['num_runs']}",
        f"Max Warmup: {metadata['benchmark_params']['max_warmup_iterations']}",
        f"Warmup Window: {metadata['benchmark_params']['warmup_window']}",
        f"Warmup Threshold: {metadata['benchmark_params']['warmup_threshold']}"
    ])

    with open(output_dir / "metadata.txt", "w") as f:
        f.write("\n".join(metadata_text))

def create_dashboard(data: dict[str, Any], output_dir: Path) -> None:
    """Create interactive dashboard from benchmark results"""
    metadata = data["metadata"]
    results = data["results"]

    # Convert to DataFrame for easier analysis
    records = []
    for transform_name, data in results.items():
        if not data["supported"]:
            continue

        records.append({
            "transform": transform_name,
            "mean_throughput": data["mean_throughput"],
            "std_throughput": data["std_throughput"],
            "warmup_iterations": data["warmup_iterations"],
            "cv": data["std_throughput"] / data["mean_throughput"],
            "warmup_throughputs": data["warmup_throughputs"],
            "throughputs": data["throughputs"]
        })

    df = pd.DataFrame(records)
    df = df.sort_values("mean_throughput", ascending=True)

    # Create main dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Mean Throughput (images/sec)",
            "Warmup Iterations Required",
            "Coefficient of Variation (%)",
            "Throughput Distribution",
            "Throughput vs Warmup Iterations",
            "Performance Stability"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "box"}],
               [{"type": "scatter"}, {"type": "heatmap"}]],
        vertical_spacing=0.12
    )

    # Mean throughput with error bars
    fig.add_trace(
        go.Bar(
            x=df["transform"],
            y=df["mean_throughput"],
            error_y=dict(type='data', array=df["std_throughput"]),
            name="Mean Throughput"
        ),
        row=1, col=1
    )

    # Warmup iterations
    fig.add_trace(
        go.Bar(
            x=df["transform"],
            y=df["warmup_iterations"],
            name="Warmup Iterations"
        ),
        row=1, col=2
    )

    # Coefficient of variation
    fig.add_trace(
        go.Bar(
            x=df["transform"],
            y=df["cv"] * 100,
            name="CV (%)"
        ),
        row=2, col=1
    )

    # Box plots
    for transform in df["transform"]:
        fig.add_trace(
            go.Box(
                y=results[transform]["throughputs"],
                name=transform,
                showlegend=False
            ),
            row=2, col=2
        )

    # Warmup convergence (sample of transforms)
    top_transforms = df.nlargest(5, "mean_throughput")["transform"]
    for transform in top_transforms:
        warmup_data = results[transform]["warmup_throughputs"]
        fig.add_trace(
            go.Scatter(
                y=warmup_data,
                name=transform,
                mode='lines',
                showlegend=True
            ),
            row=3, col=1
        )

    # Stability heatmap
    stability_matrix = []
    for transform in df["transform"]:
        throughputs = results[transform]["throughputs"]
        relative_stds = [np.std(throughputs[:i+1])/np.mean(throughputs[:i+1])
                        for i in range(len(throughputs))]
        stability_matrix.append(relative_stds)

    fig.add_trace(
        go.Heatmap(
            z=stability_matrix,
            x=[f"Run {i+1}" for i in range(len(relative_stds))],
            y=df["transform"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Relative Std Dev")
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        height=1500,
        width=1500,
        title_text=(
            f"Benchmark Results Dashboard<br>"
            f"<sup>Library: {metadata['library_versions'].get(metadata['benchmark_params'].get('library', 'unknown'), 'unknown')}, "
            f"Images: {metadata['benchmark_params']['num_images']}, "
            f"Runs: {metadata['benchmark_params']['num_runs']}</sup>"
        ),
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )

    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata summary
    create_metadata_summary(metadata, output_dir)

    # Save visualizations
    fig.write_html(output_dir / "dashboard.html")
    fig.write_image(output_dir / "dashboard.png", scale=2)  # Higher resolution

    # Generate summary statistics
    summary = pd.DataFrame({
        "Mean Throughput (img/s)": df["mean_throughput"],
        "Std Dev": df["std_throughput"],
        "CV (%)": df["cv"] * 100,
        "Warmup Iterations": df["warmup_iterations"]
    }).round(2)

    summary.to_csv(output_dir / "summary.csv")

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Path to benchmark results JSON")
    parser.add_argument("-o", "--output-dir", type=Path, default="benchmark_analysis",
                       help="Output directory for analysis")

    args = parser.parse_args()
    results = load_results(args.results)
    create_dashboard(results, args.output_dir)

if __name__ == "__main__":
    main()
