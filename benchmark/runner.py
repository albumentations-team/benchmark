import importlib
import importlib.util
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
from tqdm import tqdm

from .utils import get_image_loader, get_library_versions, get_system_info, time_transform, verify_thread_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables for various libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class BenchmarkRunner:
    def __init__(
        self,
        library: str,
        data_dir: Path,
        transforms: list[dict[str, Any]],
        call_fn: Callable[[Any, Any], Any],
        num_images: int = 1000,
        num_runs: int = 5,
        max_warmup_iterations: int = 1000,
        warmup_window: int = 5,
        warmup_threshold: float = 0.05,
        min_warmup_windows: int = 3,
    ):
        self.library = library
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.call_fn = call_fn
        self.num_images = num_images
        self.num_runs = num_runs
        self.max_warmup_iterations = max_warmup_iterations
        self.warmup_window = warmup_window
        self.warmup_threshold = warmup_threshold
        self.min_warmup_windows = min_warmup_windows

        # Get image loader for the library
        self.image_loader = get_image_loader(library)

    def load_images(self) -> list[Any]:
        """Load images using appropriate loader - only RGB images"""
        image_paths = sorted(self.data_dir.glob("*.*"))
        images = []

        with tqdm(image_paths, desc="Loading RGB images") as pbar:
            for path in pbar:
                try:
                    # Pre-check if image is RGB using OpenCV (works for all formats)
                    import cv2

                    img_check = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if img_check is None:
                        continue
                    # Skip grayscale images (1 or 2 channels) - we only want RGB
                    if img_check.ndim < 3 or img_check.shape[2] < 3:
                        continue

                    # Load image using library-specific loader
                    img = self.image_loader(path)
                    images.append(img)

                    if len(images) >= self.num_images:
                        break
                except Exception:  # noqa: S112
                    # Skip problematic images
                    continue

                pbar.set_postfix({"loaded": len(images)})

        if not images:
            raise ValueError("No valid RGB images found in the directory (only RGB images are used for benchmarking)")

        if len(images) < self.num_images:
            logger.warning("Only found %d valid RGB images, requested %d", len(images), self.num_images)

        return images

    def _perform_warmup(
        self,
        transform: Any,
        transform_name: str,
        warmup_subset: list[Any],
    ) -> tuple[list[float], float, bool, str | None]:
        """Perform adaptive warmup until performance stabilizes or early stopping conditions are met.

        Args:
            transform: The transform to benchmark
            transform_name: Name of the transform for display
            warmup_subset: Subset of images to use for warmup

        Returns:
            tuple containing:
            - list of throughput measurements
            - time per image from last measurement
            - whether early stopping occurred
            - reason for early stopping (if any)
        """
        warmup_throughputs = []
        slow_transform_threshold = 0.1  # seconds per image
        min_iterations_before_stopping = 10
        max_time_per_transform = 60
        start_time = time.time()
        time_per_image = 0.0  # Initialize here

        with tqdm(total=self.max_warmup_iterations, desc=f"Warming up {transform_name}", leave=False) as pbar:
            for i in range(self.max_warmup_iterations):
                elapsed = time_transform(lambda x: self.call_fn(transform, x), warmup_subset)
                throughput = len(warmup_subset) / elapsed
                time_per_image = elapsed / len(warmup_subset)
                warmup_throughputs.append(throughput)

                # Early stopping conditions
                total_time = time.time() - start_time

                # Stop if transform is too slow
                if i >= min_iterations_before_stopping and time_per_image > slow_transform_threshold:
                    return (
                        warmup_throughputs,
                        time_per_image,
                        True,
                        (
                            f"Transform too slow: {time_per_image:.3f} sec/image > {slow_transform_threshold}"
                            " sec/image threshold"
                        ),
                    )

                # Stop if total time exceeds maximum
                if total_time > max_time_per_transform:
                    return (
                        warmup_throughputs,
                        time_per_image,
                        True,
                        f"Transform timeout: {total_time:.1f} sec > {max_time_per_transform} sec limit",
                    )

                # Variance stability check
                if (
                    i >= self.warmup_window * self.min_warmup_windows
                    and len(warmup_throughputs) >= self.warmup_window * 2
                ):
                    recent_mean = np.mean(warmup_throughputs[-self.warmup_window :])
                    overall_mean = np.mean(warmup_throughputs)
                    relative_diff = abs(recent_mean - overall_mean) / overall_mean

                    if relative_diff < self.warmup_threshold:
                        pbar.update(self.max_warmup_iterations - i - 1)
                        break

                pbar.update(1)

        return warmup_throughputs, time_per_image, False, None

    def run_transform(self, transform_dict: dict[str, Any], images: list[Any]) -> dict[str, Any]:
        """Run benchmark for a single transform"""
        transform = transform_dict["transform"]
        transform_name = transform_dict["name"]

        # Perform warmup
        warmup_subset = images[: min(10, len(images))]
        warmup_throughputs, time_per_image, early_stopped, early_stop_reason = self._perform_warmup(
            transform,
            transform_name,
            warmup_subset,
        )

        if early_stopped:
            return {
                "supported": True,
                "warmup_iterations": len(warmup_throughputs),
                "throughputs": [],
                "median_throughput": np.median(warmup_throughputs),
                "std_throughput": np.std(warmup_throughputs, ddof=1),
                "times": [],
                "mean_time": time_per_image,
                "std_time": 0.0,
                "variance_stable": False,
                "early_stopped": True,
                "early_stop_reason": early_stop_reason,
            }

        # Benchmark runs
        throughputs = []
        times = []

        for _ in tqdm(range(self.num_runs), desc=f"Benchmarking {transform_name}", leave=False):
            elapsed = time_transform(lambda x: self.call_fn(transform, x), images)
            throughput = len(images) / elapsed
            throughputs.append(throughput)
            times.append(elapsed)

        median_throughput = np.median(throughputs)
        std_throughput = np.std(throughputs, ddof=1)

        return {
            "supported": True,
            "warmup_iterations": len(warmup_throughputs),
            "throughputs": throughputs,
            "median_throughput": median_throughput,
            "std_throughput": std_throughput,
            "times": times,
            "mean_time": len(images) / median_throughput,
            "std_time": std_throughput / (median_throughput**2) * len(images),
            "variance_stable": True,
            "early_stopped": False,
            "early_stop_reason": None,
        }

    def run(self, output_path: Path | None = None) -> dict[str, Any]:
        """Run all benchmarks"""
        logger.info("Running benchmarks for %s", self.library)
        images = self.load_images()

        # Collect metadata
        metadata = {
            "system_info": get_system_info(),
            "library_versions": get_library_versions(self.library),
            "thread_settings": verify_thread_settings(),
            "benchmark_params": {
                "num_images": self.num_images,
                "num_runs": self.num_runs,
                "max_warmup_iterations": self.max_warmup_iterations,
                "warmup_window": self.warmup_window,
                "warmup_threshold": self.warmup_threshold,
                "min_warmup_windows": self.min_warmup_windows,
            },
        }

        # Run benchmarks
        results = {}
        for transform_dict in tqdm(self.transforms, desc="Running transforms"):
            try:
                transform_name = transform_dict["name"]
                results[transform_name] = self.run_transform(transform_dict, images)
            except Exception as e:
                transform_name = transform_dict.get("name", "Unknown")
                warn(f"Transform {transform_name} failed: {e}", stacklevel=2)

        # Combine results and metadata
        full_results = {
            "metadata": metadata,
            "results": results,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(full_results, f, indent=2)

        return full_results


def load_from_python_file(specs_file: Path) -> tuple[str, Callable[[Any, Any], Any], list[dict[str, Any]]]:
    """Load library name, __call__ function, and transforms from a Python file

    The Python file must define:
    - LIBRARY: str (e.g., "albumentationsx")
    - __call__: function to apply transforms to images
    - TRANSFORMS: list of dicts with 'name' and 'transform' keys

    Returns:
        tuple of (library name, __call__ function, list of transform dicts)
    """
    spec = importlib.util.spec_from_file_location("custom_transforms", specs_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load from {specs_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Validate required attributes
    if not hasattr(module, "LIBRARY"):
        raise ValueError(f"Python file {specs_file} must define LIBRARY string")

    if not callable(module):
        raise TypeError(f"Python file {specs_file} must define __call__ function")

    if not hasattr(module, "TRANSFORMS"):
        raise ValueError(f"Python file {specs_file} must define TRANSFORMS list")

    # Validate __call__ is callable
    if not callable(module.__call__):
        raise TypeError("__call__ must be a callable function")

    # Validate transform structure
    for i, t in enumerate(module.TRANSFORMS):
        if not isinstance(t, dict):
            raise TypeError(f"TRANSFORMS[{i}] must be a dictionary")

        required_keys = {"name", "transform"}
        missing = required_keys - t.keys()
        if missing:
            raise ValueError(f"TRANSFORMS[{i}] missing keys: {missing}")

    return module.LIBRARY, module.__call__, module.TRANSFORMS


def main() -> None:
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run augmentation benchmarks")
    parser.add_argument(
        "-s",
        "--specs-file",
        type=Path,
        required=True,
        help="Python file defining LIBRARY and TRANSFORMS",
    )
    parser.add_argument("-d", "--data-dir", required=True, type=Path, help="Directory with images")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("-n", "--num-images", type=int, default=1000, help="Number of images")
    parser.add_argument("-r", "--num-runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--max-warmup", type=int, default=5000, help="Maximum warmup iterations")
    parser.add_argument("--warmup-window", type=int, default=5, help="Window size for variance check")
    parser.add_argument("--warmup-threshold", type=float, default=0.05, help="Variance stability threshold")
    parser.add_argument("--min-warmup-windows", type=int, default=3, help="Minimum windows to check")

    args = parser.parse_args()

    # Load library and transforms from file
    if not args.specs_file.exists():
        raise ValueError(f"Specs file {args.specs_file} does not exist")

    logger.info(f"Loading from {args.specs_file}")
    library, call_fn, transforms = load_from_python_file(args.specs_file)
    logger.info(f"Library: {library}")
    logger.info(f"Loaded {len(transforms)} transforms")

    runner = BenchmarkRunner(
        library=library,
        data_dir=args.data_dir,
        transforms=transforms,
        call_fn=call_fn,
        num_images=args.num_images,
        num_runs=args.num_runs,
        max_warmup_iterations=args.max_warmup,
        warmup_window=args.warmup_window,
        warmup_threshold=args.warmup_threshold,
        min_warmup_windows=args.min_warmup_windows,
    )

    runner.run(args.output)


if __name__ == "__main__":
    main()
