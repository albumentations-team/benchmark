#!/usr/bin/env python3
"""Compare transforms between two benchmark result files.

This script compares performance metrics for transforms that exist in both files.
Useful for comparing different library versions or implementations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_results(file_path: Path) -> dict[str, Any]:
    """Load benchmark results from JSON file."""
    with file_path.open() as f:
        return json.load(f)


def extract_library_info(data: dict[str, Any]) -> str:
    """Extract library and version information."""
    lib_versions = data.get("metadata", {}).get("library_versions", {})

    # Find the main library (not numpy, pillow, etc.)
    main_libs = ["albumentationsx", "torchvision", "kornia", "imgaug", "augly"]
    for lib in main_libs:
        if lib in lib_versions:
            return f"{lib} {lib_versions[lib]}"

    # If no main library found, return first library
    if lib_versions:
        lib, version = next(iter(lib_versions.items()))
        return f"{lib} {version}"

    return "Unknown"


def find_common_transforms(results1: dict[str, Any], results2: dict[str, Any]) -> list[str]:
    """Find transforms that exist in both result sets."""
    transforms1 = set(results1.get("results", {}).keys())
    transforms2 = set(results2.get("results", {}).keys())

    common = transforms1.intersection(transforms2)
    return sorted(common)


def calculate_speedup(throughput1: float, throughput2: float) -> float:
    """Calculate speedup ratio (positive means file2 is faster)."""
    if throughput1 == 0:
        return float("inf")
    return (throughput2 / throughput1 - 1) * 100


def format_comparison_table(comparisons: list[tuple[str, dict[str, Any]]]) -> str:
    """Format comparison data as a readable table."""
    if not comparisons:
        return "No common transforms found."

    # Calculate column widths
    transform_width = max(len(t[0]) for t in comparisons)
    transform_width = max(transform_width, 20)

    # Headers
    headers = [
        "Transform".ljust(transform_width),
        "File1 (videos/s)".rjust(16),
        "File2 (videos/s)".rjust(16),
        "Speedup (%)".rjust(12),
        "File1 Time (ms)".rjust(15),
        "File2 Time (ms)".rjust(15),
    ]

    table = []
    table.append(" | ".join(headers))
    table.append("-" * (sum(len(h) for h in headers) + len(headers) * 3))

    # Sort by speedup
    comparisons.sort(key=lambda x: x[1]["speedup"], reverse=True)

    for transform, metrics in comparisons:
        row = [
            transform.ljust(transform_width),
            f"{metrics['throughput1']:.2f}".rjust(16),
            f"{metrics['throughput2']:.2f}".rjust(16),
            f"{metrics['speedup']:+.1f}".rjust(12),
            f"{metrics['time1']:.2f}".rjust(15),
            f"{metrics['time2']:.2f}".rjust(15),
        ]
        table.append(" | ".join(row))

    return "\n".join(table)


def compare_results(file1: Path, file2: Path, output_format: str = "table") -> None:
    """Compare benchmark results from two files."""
    # Load results
    results1 = load_results(file1)
    results2 = load_results(file2)

    # Extract library info
    lib1 = extract_library_info(results1)
    lib2 = extract_library_info(results2)

    print("\nComparing transforms between:")
    print(f"  File 1: {file1.name} ({lib1})")
    print(f"  File 2: {file2.name} ({lib2})")
    print()

    # Find common transforms
    common_transforms = find_common_transforms(results1, results2)

    if not common_transforms:
        print("No common transforms found between the two files.")
        return

    print(f"Found {len(common_transforms)} common transforms\n")

    # Compare each transform
    comparisons = []

    for transform in common_transforms:
        t1 = results1["results"][transform]
        t2 = results2["results"][transform]

        # Skip if either didn't run successfully
        if not t1.get("supported", True) or not t2.get("supported", True):
            continue

        if t1.get("early_stopped", False) or t2.get("early_stopped", False):
            continue

        throughput1 = t1.get("median_throughput", 0)
        throughput2 = t2.get("median_throughput", 0)

        # Calculate times in milliseconds per video
        time1 = t1.get("mean_time", 0) * 1000  # Convert to ms
        time2 = t2.get("mean_time", 0) * 1000

        speedup = calculate_speedup(throughput1, throughput2)

        comparisons.append(
            (
                transform,
                {
                    "throughput1": throughput1,
                    "throughput2": throughput2,
                    "speedup": speedup,
                    "time1": time1,
                    "time2": time2,
                },
            ),
        )

    # Output results
    if output_format == "table":
        print(format_comparison_table(comparisons))

        # Summary statistics
        if comparisons:
            speedups = [c[1]["speedup"] for c in comparisons]
            avg_speedup = sum(speedups) / len(speedups)

            print("\nSummary:")
            print(f"  Average speedup: {avg_speedup:+.1f}%")
            best_speedup_transform = next(c[0] for c in comparisons if c[1]["speedup"] == max(speedups))
            print(f"  Best speedup: {max(speedups):+.1f}% ({best_speedup_transform})")
            worst_speedup_transform = next(c[0] for c in comparisons if c[1]["speedup"] == min(speedups))
            print(f"  Worst speedup: {min(speedups):+.1f}% ({worst_speedup_transform})")

            faster_count = sum(1 for s in speedups if s > 0)
            slower_count = sum(1 for s in speedups if s < 0)
            same_count = sum(1 for s in speedups if abs(s) < 1.0)

            print("\n  File2 vs File1:")
            print(f"    Faster: {faster_count} transforms")
            print(f"    Slower: {slower_count} transforms")
            print(f"    Similar (±1%): {same_count} transforms")

    elif output_format == "csv":
        # CSV output
        print("Transform,File1_Throughput,File2_Throughput,Speedup_Percent,File1_Time_ms,File2_Time_ms")
        for transform, metrics in comparisons:
            print(
                f"{transform},{metrics['throughput1']:.2f},{metrics['throughput2']:.2f},"
                f"{metrics['speedup']:.1f},{metrics['time1']:.2f},{metrics['time2']:.2f}",
            )

    elif output_format == "json":
        # JSON output
        output = {
            "file1": {
                "path": str(file1),
                "library": lib1,
            },
            "file2": {
                "path": str(file2),
                "library": lib2,
            },
            "comparisons": dict(comparisons),
            "summary": {
                "common_transforms": len(comparisons),
                "average_speedup": sum(c[1]["speedup"] for c in comparisons) / len(comparisons) if comparisons else 0,
            },
        }
        print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare transforms between two benchmark result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two result files
  python tools/compare_transform_results.py result1.json result2.json

  # Output as CSV
  python tools/compare_transform_results.py result1.json result2.json --format csv

  # Output as JSON
  python tools/compare_transform_results.py result1.json result2.json --format json
        """,
    )

    parser.add_argument("file1", type=Path, help="First benchmark result file")
    parser.add_argument("file2", type=Path, help="Second benchmark result file")
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)",
    )

    args = parser.parse_args()

    # Validate files exist
    if not args.file1.exists():
        print(f"Error: File not found: {args.file1}", file=sys.stderr)
        sys.exit(1)

    if not args.file2.exists():
        print(f"Error: File not found: {args.file2}", file=sys.stderr)
        sys.exit(1)

    # Run comparison
    compare_results(args.file1, args.file2, args.format)


if __name__ == "__main__":
    main()
