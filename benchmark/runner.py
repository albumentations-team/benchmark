import json
import numpy as np
from pathlib import Path
from typing import Any
import importlib
from tqdm import tqdm
import time
import json

from pathlib import Path
from typing import Any
import importlib

from .transforms.specs import TRANSFORM_SPECS
from .utils import get_image_loader, get_system_info, time_transform, verify_thread_settings, get_library_versions, is_variance_stable


import os

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
        num_images: int = 1000,
        num_runs: int = 5,
        max_warmup_iterations: int = 1000,
        warmup_window: int = 5,
        warmup_threshold: float = 0.05,
        min_warmup_windows: int = 3,
    ):
        self.library = library
        self.data_dir = Path(data_dir)
        self.num_images = num_images
        self.num_runs = num_runs
        self.max_warmup_iterations = max_warmup_iterations
        self.warmup_window = warmup_window
        self.warmup_threshold = warmup_threshold
        self.min_warmup_windows = min_warmup_windows

        # Load implementation
        self.impl = self._get_implementation()
        self.image_loader = get_image_loader(library)

    def _get_implementation(self) -> Any:
        """Import library-specific implementation"""
        try:
            module = importlib.import_module(f".transforms.{self.library.lower()}_impl", package="benchmark")
            return getattr(module, f"{self.library.capitalize()}Impl")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Library {self.library} not supported: {e}")

    def load_images(self) -> list[Any]:
        """Load images using appropriate loader"""
        image_paths = sorted(self.data_dir.glob("*.*"))
        rgb_images = []

        with tqdm(image_paths, desc="Loading images") as pbar:
            for path in pbar:
                try:
                    img = self.image_loader(path)
                    # Check if image is RGB (3 channels)
                    if hasattr(img, 'shape'):  # numpy array or tensor
                        if len(img.shape) == 4:  # batched tensor (B,C,H,W)
                            if img.shape[1] != 3:  # check channels
                                continue
                        elif len(img.shape) == 3:  # unbatched array/tensor
                            if img.shape[0] != 3 and img.shape[-1] != 3:
                                continue
                        else:
                            continue
                    elif hasattr(img, 'mode'):  # PIL Image
                        if img.mode != 'RGB':
                            continue
                    rgb_images.append(img)

                    if len(rgb_images) >= self.num_images:
                        break
                except Exception as e:
                    # Skip problematic images
                    continue

                pbar.set_postfix({'loaded': len(rgb_images)})

        if not rgb_images:
            raise ValueError("No valid RGB images found in the directory")

        if len(rgb_images) < self.num_images:
            print(f"Warning: Only found {len(rgb_images)} valid RGB images, requested {self.num_images}")

        return rgb_images

    def _perform_warmup(
        self,
        transform: Any,
        transform_spec: Any,
        warmup_subset: list[Any]
    ) -> tuple[list[float], float, bool, str | None]:
        """Perform adaptive warmup until performance stabilizes or early stopping conditions are met.

        Args:
            transform: The transform to benchmark
            transform_spec: The transform specification containing name and parameters
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

        with tqdm(total=self.max_warmup_iterations,
                desc=f"Warming up {transform_spec.name}",
                leave=False) as pbar:

            for i in range(self.max_warmup_iterations):
                elapsed = time_transform(lambda x: self.impl.__call__(transform, x), warmup_subset)
                throughput = len(warmup_subset) / elapsed
                time_per_image = elapsed / len(warmup_subset)
                warmup_throughputs.append(throughput)

                # Early stopping conditions
                total_time = time.time() - start_time

                # Stop if transform is too slow
                if (i >= min_iterations_before_stopping and
                    time_per_image > slow_transform_threshold):
                    return warmup_throughputs, time_per_image, True, \
                           f"Transform too slow: {time_per_image:.3f} sec/image > {slow_transform_threshold} sec/image threshold"

                # Stop if total time exceeds maximum
                if total_time > max_time_per_transform:
                    return warmup_throughputs, time_per_image, True, \
                           f"Transform timeout: {total_time:.1f} sec > {max_time_per_transform} sec limit"

                # Variance stability check
                if (i >= self.warmup_window * self.min_warmup_windows and
                    len(warmup_throughputs) >= self.warmup_window * 2):
                    recent_mean = np.mean(warmup_throughputs[-self.warmup_window:])
                    overall_mean = np.mean(warmup_throughputs)
                    relative_diff = abs(recent_mean - overall_mean) / overall_mean

                    if relative_diff < self.warmup_threshold:
                        pbar.update(self.max_warmup_iterations - i - 1)
                        break

                pbar.update(1)

        return warmup_throughputs, time_per_image, False, None


    def run_transform(self, transform_spec: Any, images: list[Any]) -> dict[str, Any]:
        """Run benchmark for a single transform"""
        if not hasattr(self.impl, transform_spec.name):
            return {"supported": False}

        # Create transform
        transform_fn = getattr(self.impl, transform_spec.name)
        transform = transform_fn(transform_spec.params)

        # Perform warmup
        warmup_subset = images[:min(10, len(images))]
        warmup_throughputs, time_per_image, early_stopped, early_stop_reason = \
            self._perform_warmup(transform, transform_spec, warmup_subset)

        if early_stopped:
            return {
                "supported": True,
                "warmup_iterations": len(warmup_throughputs),
                "throughputs": [],
                "median_throughput": np.median(warmup_throughputs),
                "std_throughput": np.std(warmup_throughputs),
                "times": [],
                "mean_time": time_per_image,
                "std_time": 0.0,
                "variance_stable": False,
                "early_stopped": True,
                "early_stop_reason": early_stop_reason
            }

        # Benchmark runs
        throughputs = []
        times = []

        for _ in tqdm(range(self.num_runs),
                    desc=f"Benchmarking {transform_spec.name}",
                    leave=False):
            elapsed = time_transform(lambda x: self.impl.__call__(transform, x), images)
            throughput = len(images) / elapsed
            throughputs.append(throughput)
            times.append(elapsed)

        median_throughput = np.median(throughputs)
        std_throughput = np.std(throughputs)

        return {
            "supported": True,
            "warmup_iterations": len(warmup_throughputs),
            "throughputs": throughputs,
            "median_throughput": median_throughput,
            "std_throughput": std_throughput,
            "times": times,
            "mean_time": len(images) / median_throughput,
            "std_time": std_throughput / (median_throughput ** 2) * len(images),
            "variance_stable": True,
            "early_stopped": False,
            "early_stop_reason": None
        }

    def run(self, output_path: Path | None = None) -> dict[str, Any]:
        """Run all benchmarks"""
        print(f"\nRunning benchmarks for {self.library}")
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
                "min_warmup_windows": self.min_warmup_windows
            }
        }

        # Run benchmarks
        results = {
            str(spec): self.run_transform(spec, images)
            for spec in tqdm(TRANSFORM_SPECS, desc="Running transforms")
        }

        # Combine results and metadata
        full_results = {
            "metadata": metadata,
            "results": results
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(full_results, f, indent=2)

        return full_results

def main() -> None:
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run augmentation benchmarks")
    parser.add_argument("-l", "--library", required=True, help="Library to benchmark")
    parser.add_argument("-d", "--data-dir", required=True, type=Path, help="Directory with images")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("-n", "--num-images", type=int, default=1000, help="Number of images")
    parser.add_argument("-r", "--num-runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--max-warmup", type=int, default=5000, help="Maximum warmup iterations")
    parser.add_argument("--warmup-window", type=int, default=5, help="Window size for variance check")
    parser.add_argument("--warmup-threshold", type=float, default=0.05, help="Variance stability threshold")
    parser.add_argument("--min-warmup-windows", type=int, default=3, help="Minimum windows to check")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        library=args.library,
        data_dir=args.data_dir,
        num_images=args.num_images,
        num_runs=args.num_runs,
        max_warmup_iterations=args.max_warmup,
        warmup_window=args.warmup_window,
        warmup_threshold=args.warmup_threshold,
        min_warmup_windows=args.min_warmup_windows
    )

    runner.run(args.output)

if __name__ == "__main__":
    main()
