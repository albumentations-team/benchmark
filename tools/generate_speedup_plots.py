#!/usr/bin/env python3
"""Generate speedup analysis plots for benchmark results."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


def load_results(results_dir: Path, file_pattern: str) -> Dict[str, Dict]:
    """Load benchmark results from JSON files."""
    results = {}
    for file_path in results_dir.glob(file_pattern):
        library = file_path.stem.split("_")[0]
        with open(file_path, "r") as f:
            results[library] = json.load(f)
    return results


def calculate_speedups(results: Dict[str, Dict], reference_library: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate speedups relative to the fastest non-reference library for each transform.

    If reference_library is provided, speedups are calculated as reference_library / other_library.
    Otherwise, speedups are calculated as library / fastest_other_library.
    """
    # Extract all transform names
    all_transforms = set()
    for library_results in results.values():
        all_transforms.update(library_results["results"].keys())

    # Create a DataFrame to store throughputs
    throughputs = pd.DataFrame(index=sorted(all_transforms))

    # Fill the DataFrame with median throughputs
    for library, library_results in results.items():
        library_throughputs = {}
        for transform, transform_results in library_results["results"].items():
            if transform_results.get("supported", False) and not transform_results.get("early_stopped", False):
                library_throughputs[transform] = transform_results.get("median_throughput", 0)
        throughputs[library] = pd.Series(library_throughputs)

    # Calculate speedups
    if reference_library:
        # Calculate speedups relative to the reference library
        reference_throughputs = throughputs[reference_library]
        speedups = pd.DataFrame(index=throughputs.index)

        for library in throughputs.columns:
            if library != reference_library:
                # Skip NaN values (unsupported transforms)
                mask = ~throughputs[library].isna() & ~reference_throughputs.isna() & (throughputs[library] > 0)
                speedups[library] = reference_throughputs[mask] / throughputs[library][mask]

        # For the reference library column, we need to calculate its speedup compared to the fastest other library
        # For each transform, find the fastest non-reference library
        for transform in throughputs.index:
            # Get throughputs for all libraries except the reference library
            other_libs = [lib for lib in throughputs.columns if lib != reference_library]
            other_throughputs = [throughputs.loc[transform, lib] for lib in other_libs
                                if not pd.isna(throughputs.loc[transform, lib]) and throughputs.loc[transform, lib] > 0]

            if other_throughputs and not pd.isna(reference_throughputs[transform]) and reference_throughputs[transform] > 0:
                # Calculate speedup as reference / fastest_other
                fastest_other = max(other_throughputs)
                speedups.loc[transform, reference_library] = reference_throughputs[transform] / fastest_other

        # Filter out transforms that only the reference library supports
        # Count how many libraries support each transform
        support_count = throughputs.notna().sum(axis=1)
        # Keep only transforms supported by more than one library
        speedups = speedups.loc[support_count > 1]
    else:
        # For each transform, find the fastest library
        speedups = pd.DataFrame(index=throughputs.index)

        for transform in throughputs.index:
            transform_throughputs = throughputs.loc[transform].dropna()
            if len(transform_throughputs) > 0:
                fastest_library = transform_throughputs.idxmax()
                fastest_throughput = transform_throughputs[fastest_library]

                for library in throughputs.columns:
                    if not pd.isna(throughputs.loc[transform, library]) and throughputs.loc[transform, library] > 0:
                        speedups.loc[transform, library] = throughputs.loc[transform, library] / fastest_throughput

        # Filter out transforms that only one library supports
        support_count = throughputs.notna().sum(axis=1)
        speedups = speedups.loc[support_count > 1]

    return speedups


def plot_speedup_distribution(speedups: pd.DataFrame, reference_library: str, output_path: Path, max_speedup: float = 20.0):
    """Plot the distribution of speedups for a reference library."""
    # Create a DataFrame with Transform and Speedup columns for easier plotting
    comparison_df = pd.DataFrame({
        'Transform': speedups.index,
        'Speedup': speedups[reference_library].values
    }).dropna()

    # Clean transform names - remove parameters in parentheses
    comparison_df['Transform'] = comparison_df['Transform'].apply(lambda x: x.split('(')[0].strip())

    # Check if we have any data to plot
    if len(comparison_df) == 0:
        print(f"Warning: No speedup data available for {reference_library}")
        # Create a simple plot with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No speedup data available for {reference_library}",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, width_ratios=[1.5, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Use colorblind-friendly colors
    hist_color = '#4878D0'    # Blue
    top_color = '#60BD68'     # Green
    bottom_color = '#EE6677'  # Red

    # 1. Histogram of typical speedups (< max_speedup)
    typical_speedups = comparison_df[comparison_df['Speedup'] < max_speedup]['Speedup']

    if len(typical_speedups) > 0:
        ax1.hist(typical_speedups, bins=15, color=hist_color, alpha=0.7, edgecolor='black')
        ax1.axvline(typical_speedups.median(), color='#404040', linestyle='--', linewidth=1.5,
                    label=f'Median: {typical_speedups.median():.2f}×')
        ax1.axvline(1, color='#404040', linestyle=':', linewidth=1.5, alpha=0.7,
                    label='No speedup (1×)')
        ax1.grid(True, alpha=0.3)

        ax1.set_xlabel('Speedup (×)', fontsize=12)
        ax1.set_ylabel('Number of transforms', fontsize=12)
        ax1.set_title(f'(a) Distribution of Typical Speedups\n(< {max_speedup}×)', fontsize=14)
        ax1.legend(fontsize=10)
    else:
        ax1.text(0.5, 0.5, "No speedup data < 20×", ha='center', va='center', fontsize=12)
        ax1.set_axis_off()

    # 2. Top 10 speedups (or all if less than 10)
    top_n = min(10, len(comparison_df))
    if top_n > 0:
        top_10 = comparison_df.nlargest(top_n, 'Speedup')
        bars = ax2.barh(np.arange(len(top_10)), top_10['Speedup'],
                        color=top_color, alpha=0.7, edgecolor='black')
        ax2.set_yticks(np.arange(len(top_10)))
        ax2.set_yticklabels(top_10['Transform'], fontsize=10)
        ax2.grid(True, alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}×',
                    ha='left', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax2.set_xlabel('Speedup (×)', fontsize=12)
        ax2.set_title('(b) Top 10 Speedups', fontsize=14)
    else:
        ax2.text(0.5, 0.5, "No speedup data", ha='center', va='center', fontsize=12)
        ax2.set_axis_off()

    # 3. Bottom 10 speedups (or all if less than 10)
    bottom_n = min(10, len(comparison_df))
    if bottom_n > 0:
        bottom_10 = comparison_df.nsmallest(bottom_n, 'Speedup')
        bars = ax3.barh(np.arange(len(bottom_10)), bottom_10['Speedup'],
                        color=bottom_color, alpha=0.7, edgecolor='black')
        ax3.set_yticks(np.arange(len(bottom_10)))
        ax3.set_yticklabels(bottom_10['Transform'], fontsize=10)
        ax3.grid(True, alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}×',
                    ha='left', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax3.set_xlabel('Speedup (×)', fontsize=12)
        ax3.set_title('(c) Bottom 10 Speedups', fontsize=14)
    else:
        ax3.text(0.5, 0.5, "No speedup data", ha='center', va='center', fontsize=12)
        ax3.set_axis_off()

    # Add reference lines if axes are not empty
    if top_n > 0:
        ax2.axvline(1, color='#404040', linestyle=':', alpha=0.5)
    if bottom_n > 0:
        ax3.axvline(1, color='#404040', linestyle=':', alpha=0.5)

    # Add summary statistics
    total_transforms = len(comparison_df)
    faster_transforms = (comparison_df['Speedup'] > 1).sum()
    high_speedup_transforms = (comparison_df['Speedup'] >= 5).sum()
    median_speedup = comparison_df['Speedup'].median()
    mean_speedup = comparison_df['Speedup'].mean()
    std_speedup = comparison_df['Speedup'].std()

    stats_text = (
        f"Median speedup: {median_speedup:.2f}×\n"
        f"Mean speedup: {mean_speedup:.2f}×\n"
        f"Std dev: {std_speedup:.2f}\n"
        f"{faster_transforms}/{total_transforms} transforms faster\n"
        f"{high_speedup_transforms} transforms with 5×+ speedup\n"
        f"{total_transforms} transforms with multiple library support"
    )

    # Add the stats text to the bottom right of the figure
    plt.figtext(0.98, 0.02, stats_text,
                ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                fontsize=10)

    # Add title with information about the reference library
    plt.suptitle(f'Speedup Analysis: {reference_library.capitalize()} vs Other Libraries', fontsize=16)

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)  # Make room for the suptitle and stats
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate speedup analysis plots')
    parser.add_argument('-r', '--results-dir', type=Path, required=True, help='Directory containing benchmark results')
    parser.add_argument('-o', '--output-dir', type=Path, required=True, help='Output directory for plots')
    parser.add_argument('-t', '--type', choices=['images', 'videos'], required=True, help='Type of benchmark (images or videos)')
    parser.add_argument('-l', '--reference-library', default='albumentations', help='Reference library for speedup calculation')
    args = parser.parse_args()

    # Determine file pattern based on benchmark type
    file_pattern = '*_results.json' if args.type == 'images' else '*_video_results.json'

    # Load results
    results = load_results(args.results_dir, file_pattern)

    if args.reference_library not in results:
        print(f"Error: Reference library '{args.reference_library}' not found in results.")
        return

    # Calculate speedups
    speedups = calculate_speedups(results, args.reference_library)

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plot
    output_path = args.output_dir / f'{args.type}_speedup_analysis.png'
    plot_speedup_distribution(speedups, args.reference_library, output_path)

    print(f"Speedup analysis plot saved to {output_path}")

    # Save speedups to CSV for reference
    csv_path = args.output_dir / f'{args.type}_speedups.csv'
    speedups.to_csv(csv_path)
    print(f"Speedup data saved to {csv_path}")

    # Also save the raw throughputs for reference
    throughputs = pd.DataFrame(index=sorted(set(transform for lib_results in results.values() for transform in lib_results["results"].keys())))
    for library, library_results in results.items():
        library_throughputs = {}
        for transform, transform_results in library_results["results"].items():
            if transform_results.get("supported", False) and not transform_results.get("early_stopped", False):
                library_throughputs[transform] = transform_results.get("median_throughput", 0)
        throughputs[library] = pd.Series(library_throughputs)

    # Save raw throughputs
    throughputs_csv_path = args.output_dir / f'{args.type}_throughputs.csv'
    throughputs.to_csv(throughputs_csv_path)
    print(f"Raw throughput data saved to {throughputs_csv_path}")

    # Generate a summary of the speedup analysis
    summary = {
        "total_transforms": len(speedups),
        "median_speedup": speedups[args.reference_library].median() if not speedups.empty else 0,
        "mean_speedup": speedups[args.reference_library].mean() if not speedups.empty else 0,
        "std_speedup": speedups[args.reference_library].std() if not speedups.empty else 0,
        "faster_transforms": len(speedups[speedups[args.reference_library] > 1.0]) if not speedups.empty else 0,
        "max_speedup": speedups[args.reference_library].max() if not speedups.empty else 0,
        "max_speedup_transform": speedups[args.reference_library].idxmax() if not speedups.empty else "N/A",
        "min_speedup": speedups[args.reference_library].min() if not speedups.empty else 0,
        "min_speedup_transform": speedups[args.reference_library].idxmin() if not speedups.empty else "N/A"
    }

    # Print summary
    print("\nSpeedup Analysis Summary:")
    print(f"Total transforms with multiple library support: {summary['total_transforms']}")
    print(f"Median speedup: {summary['median_speedup']:.2f}×")
    print(f"Mean speedup: {summary['mean_speedup']:.2f}×")
    print(f"Standard deviation: {summary['std_speedup']:.2f}")
    print(f"Transforms where {args.reference_library} is faster: {summary['faster_transforms']}/{summary['total_transforms']}")
    print(f"Max speedup: {summary['max_speedup']:.2f}× ({summary['max_speedup_transform']})")
    print(f"Min speedup: {summary['min_speedup']:.2f}× ({summary['min_speedup_transform']})")


if __name__ == '__main__':
    main()
