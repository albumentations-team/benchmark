import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_throughput(value: str) -> tuple[float, float]:
    """Parse throughput string like '3662 ± 54' or '-' into (value, std)"""
    if value == "-":
        return (0, 0)
    # Remove bold markers if present
    value = value.replace("**", "")
    parts = value.split("±")
    return (float(parts[0].strip()), float(parts[1].strip()))


def load_results_from_csv(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Load results from CSV file into a dictionary mapping transform names to lists of (value, std) tuples"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    results = {}

    # Select only the columns we want, using partial matches
    albumentationsx_col = next(col for col in df.columns if col.startswith("albumentationsx"))
    torchvision_col = next(col for col in df.columns if col.startswith("torchvision"))
    kornia_col = next(col for col in df.columns if col.startswith("kornia"))

    columns = ["Transform", albumentationsx_col, torchvision_col, kornia_col]
    df = df[columns]

    for _, row in df.iterrows():
        transform = row["Transform"]
        # Get throughput values for each library (skip the Transform column)
        throughputs = [parse_throughput(val) for val in row.iloc[1:]]
        results[transform] = throughputs

    return results


def format_throughput(value: float, std: float) -> str:
    if value == 0:
        return "-"
    return f"{int(value)} ± {int(std)}"


def calculate_speedup(values: list[tuple[float, float]]) -> float:
    """Calculate speedup ratio between Albumentationsx and the best of other libraries.

    Returns > 1 if Albumentationsx is faster, < 1 if another library is faster.
    """
    albumentationsx_value = values[0][0]  # First value is Albumentationsx (get value, not tuple)
    other_values = [v[0] for v in values[1:]]  # Skip std dev

    if albumentationsx_value == 0 or not other_values:
        return 0

    best_other = max(other_values)
    return albumentationsx_value / best_other


# Sets of transform names to look for in results
SPATIAL_TRANSFORMS: set[str] = {
    "Resize",
    "RandomCrop128",
    "HorizontalFlip",
    "VerticalFlip",
    "Rotate",
    "Affine",
    "Perspective",
    "Elastic",
    "Shear",
    "RandomResizedCrop",
    "Pad",
    "Erasing",
    "OpticalDistortion",
    "ThinPlateSpline",
}

PIXEL_TRANSFORMS: set[str] = {
    "ChannelShuffle",
    "Grayscale",
    "GaussianBlur",
    "GaussianNoise",
    "Invert",
    "Posterize",
    "Solarize",
    "Sharpen",
    "Equalize",
    "JpegCompression",
    "RandomGamma",
    "MedianBlur",
    "MotionBlur",
    "CLAHE",
    "Brightness",
    "Contrast",
    "Blur",
    "Saturation",
    "ColorJitter",
    "AutoContrast",
    "Normalize",
    "RGBShift",
    "PlankianJitter",
    "ChannelDropout",
    "LinearIllumination",
    "CornerIllumination",
    "GaussianIllumination",
    "Hue",
    "PlasmaBrightness",
    "PlasmaContrast",
    "PlasmaShadow",
    "Rain",
    "SaltAndPepper",
    "Snow",
}


def generate_comparison_tables(csv_path: Path) -> str:
    results = load_results_from_csv(csv_path)

    markdown = "## Performance Comparison (images/second, higher is better)\n\n"

    # Generate tables for both categories
    for category, transforms in [
        ("Spatial Transformations", SPATIAL_TRANSFORMS),
        ("Pixel-Level Transformations", PIXEL_TRANSFORMS),
    ]:
        markdown += f"### {category}\n"
        markdown += "| Transform | Albumentationsx | TorchVision | Kornia | Speedup* |\n"
        markdown += "|-----------|---------------|-------------|--------|----------|\n"

        for transform_name in sorted(transforms):
            if transform_name not in results:
                continue

            throughputs = results[transform_name]
            values = [t[0] for t in throughputs]

            # Format strings and bold the highest value
            max_val = max(values)
            formatted = []
            for val, std in throughputs:
                if val == 0:
                    formatted.append("-")
                elif val == max_val:
                    formatted.append(f"**{format_throughput(val, std)}**")
                else:
                    formatted.append(format_throughput(val, std))

            # Calculate speedup
            speedup = calculate_speedup(throughputs)
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "-"

            markdown += f"| {transform_name} | {' | '.join(formatted)} | {speedup_str} |\n"

        markdown += "\n*Speedup shows how many times the fastest library is faster than the second-best library\n\n"

    return markdown


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparison tables from benchmark results")
    parser.add_argument("-f", "--file", type=str, help="Path to CSV file containing benchmark results")
    args = parser.parse_args()

    tables = generate_comparison_tables(Path(args.file))
    logger.info("Generated comparison tables:\n%s", tables)
