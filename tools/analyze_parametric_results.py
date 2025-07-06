#!/usr/bin/env python3
"""Analyze results from parametric benchmarks.

This tool helps analyze benchmark results when testing the same transform
with different parameters, making it easy to identify optimal settings.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def analyze_parametric_results(json_file: Path) -> pd.DataFrame:
    """Analyze results and group by transform type"""
    with json_file.open() as f:
        data = json.load(f)

    results = []
    for transform_name, result in data["results"].items():
        if result.get("supported", False) and not result.get("skipped", False):
            # Extract base transform name (before first parenthesis or space)
            base_name = transform_name.split("(")[0].split()[0]

            results.append(
                {
                    "transform": base_name,
                    "full_name": transform_name,
                    "throughput": result.get("median_throughput", 0),
                    "time_per_video": result.get("mean_time", 0) / data["metadata"]["benchmark_params"]["num_videos"],
                    "std_throughput": result.get("std_throughput", 0),
                },
            )

    return pd.DataFrame(results)


def print_analysis(df: pd.DataFrame) -> None:
    """Print analysis of parametric results"""
    print("\n" + "=" * 80)
    print("PARAMETRIC BENCHMARK ANALYSIS")
    print("=" * 80)

    # Group by transform type
    for transform in sorted(df["transform"].unique()):
        transform_df = df[df["transform"] == transform].sort_values("throughput", ascending=False)

        if len(transform_df) > 1:
            print(f"\n{transform}:")
            print("-" * 40)

            best = transform_df.iloc[0]
            worst = transform_df.iloc[-1]

            print(f"  Best config:  {best['full_name']}")
            print(f"    Throughput: {best['throughput']:.2f} videos/s")
            print(f"    Time/video: {best['time_per_video']:.4f} s")

            print(f"\n  Worst config: {worst['full_name']}")
            print(f"    Throughput: {worst['throughput']:.2f} videos/s")
            print(f"    Time/video: {worst['time_per_video']:.4f} s")

            speedup = best["throughput"] / worst["throughput"]
            print(f"\n  Speedup (best/worst): {speedup:.2f}x")

            # Show all configs if not too many
            if len(transform_df) <= 10:
                print(f"\n  All {len(transform_df)} configurations:")
                for _, row in transform_df.iterrows():
                    print(f"    {row['full_name']:<50} {row['throughput']:>8.2f} videos/s")


def export_summary(df: pd.DataFrame, output_file: Path) -> None:
    """Export summary to CSV"""
    # Create summary with best config for each transform
    summary = []
    for transform in df["transform"].unique():
        transform_df = df[df["transform"] == transform]
        best = transform_df.loc[transform_df["throughput"].idxmax()]
        summary.append(
            {
                "transform": transform,
                "best_config": best["full_name"],
                "throughput": best["throughput"],
                "time_per_video": best["time_per_video"],
                "num_configs_tested": len(transform_df),
            },
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary exported to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze parametric benchmark results")
    parser.add_argument("json_file", type=Path, help="JSON results file from benchmark")
    parser.add_argument("-o", "--output", type=Path, help="Export summary to CSV")
    parser.add_argument("--full", action="store_true", help="Show full results table")

    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"Error: File {args.json_file} not found")
        return

    df = analyze_parametric_results(args.json_file)

    if args.full:
        print("\nFull Results:")
        print(df.to_string(index=False))

    print_analysis(df)

    if args.output:
        export_summary(df, args.output)


if __name__ == "__main__":
    main()
