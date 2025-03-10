import importlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .transforms.specs import TRANSFORM_SPECS
from .utils import get_library_versions, get_system_info, get_video_loader, time_transform, verify_thread_settings

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Environment variables for various libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class VideoBenchmarkRunner:
    def __init__(
        self,
        library: str,
        data_dir: Path,
        num_videos: int = 50,
        num_runs: int = 5,
        max_warmup_iterations: int = 100,
        warmup_window: int = 5,
        warmup_threshold: float = 0.05,
        min_warmup_windows: int = 3,
    ):
        self.library = library
        self.data_dir = Path(data_dir)
        self.num_videos = num_videos
        self.num_runs = num_runs
        self.max_warmup_iterations = max_warmup_iterations
        self.warmup_window = warmup_window
        self.warmup_threshold = warmup_threshold
        self.min_warmup_windows = min_warmup_windows

        # Load implementation
        self.impl = self._get_implementation()
        self.video_loader = get_video_loader(library)

    def _get_implementation(self) -> Any:
        """Import library-specific implementation"""
        try:
            module = importlib.import_module(f".transforms.{self.library.lower()}_video_impl", package="benchmark")
            return getattr(module, f"{self.library.capitalize()}VideoImpl")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Library {self.library} not supported for video: {e}")  # noqa: B904

    def load_videos(self) -> list[Any]:
        """Load videos using appropriate loader"""
        # Search recursively for video files
        video_paths = []
        for ext in ["mp4", "avi", "mov"]:
            video_paths.extend(list(self.data_dir.rglob(f"*.{ext}")))

        video_paths = sorted(video_paths)
        logger.info(f"Found {len(video_paths)} video files in {self.data_dir} (including subdirectories)")

        videos = []

        with tqdm(video_paths, desc="Loading videos") as pbar:
            for path in pbar:
                try:
                    video = self.video_loader(path)
                    videos.append(video)

                    if len(videos) >= self.num_videos:
                        break
                except Exception as e:
                    # Skip problematic videos
                    logger.warning(f"Error loading video {path}: {e}")
                    continue

                pbar.set_postfix({"loaded": len(videos)})

        if not videos:
            raise ValueError("No valid videos found in the directory (searched recursively)")

        if len(videos) < self.num_videos:
            logger.warning(f"Only found {len(videos)} valid videos, requested {self.num_videos}")

        return videos

    def _perform_warmup(
        self,
        transform: Any,
        transform_spec: Any,
        warmup_subset: list[Any],
    ) -> tuple[list[float], float, bool, str | None]:
        """Perform adaptive warmup until performance stabilizes or early stopping conditions are met.

        Args:
            transform: The transform to benchmark
            transform_spec: The transform specification containing name and parameters
            warmup_subset: Subset of videos to use for warmup

        Returns:
            tuple containing:
            - list of throughput measurements
            - time per video from last measurement
            - whether early stopping occurred
            - reason for early stopping (if any)
        """
        warmup_throughputs = []
        slow_transform_threshold = 2.0  # seconds per video
        min_iterations_before_stopping = 5
        max_time_per_transform = 120
        start_time = time.time()

        with tqdm(total=self.max_warmup_iterations, desc=f"Warming up {transform_spec.name}", leave=False) as pbar:
            for i in range(self.max_warmup_iterations):
                elapsed = time_transform(lambda x: self.impl.__call__(transform, x), warmup_subset)
                throughput = len(warmup_subset) / elapsed
                time_per_video = elapsed / len(warmup_subset)
                warmup_throughputs.append(throughput)

                # Early stopping conditions
                total_time = time.time() - start_time

                # Stop if transform is too slow
                if i >= min_iterations_before_stopping and time_per_video > slow_transform_threshold:
                    return (
                        warmup_throughputs,
                        time_per_video,
                        True,
                        (
                            f"Transform too slow: {time_per_video:.3f} sec/video > {slow_transform_threshold}"
                            " sec/video threshold"
                        ),
                    )

                # Stop if total time exceeds maximum
                if total_time > max_time_per_transform:
                    return (
                        warmup_throughputs,
                        time_per_video,
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

        return warmup_throughputs, time_per_video, False, None

    def run_transform(self, transform_spec: Any, videos: list[Any]) -> dict[str, Any]:
        """Run benchmark for a single transform"""
        if not hasattr(self.impl, transform_spec.name):
            return {"supported": False}

        # Create transform
        transform_fn = getattr(self.impl, transform_spec.name)
        transform = transform_fn(transform_spec.params)

        # Perform warmup
        warmup_subset = videos[: min(3, len(videos))]
        warmup_throughputs, time_per_video, early_stopped, early_stop_reason = self._perform_warmup(
            transform,
            transform_spec,
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
                "mean_time": time_per_video,
                "std_time": 0.0,
                "variance_stable": False,
                "early_stopped": True,
                "early_stop_reason": early_stop_reason,
            }

        # Benchmark runs
        throughputs = []
        times = []

        for _ in tqdm(range(self.num_runs), desc=f"Benchmarking {transform_spec.name}", leave=False):
            elapsed = time_transform(lambda x: self.impl.__call__(transform, x), videos)
            throughput = len(videos) / elapsed
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
            "mean_time": len(videos) / median_throughput,
            "std_time": std_throughput / (median_throughput**2) * len(videos),
            "variance_stable": True,
            "early_stopped": False,
            "early_stop_reason": None,
        }

    def run(self, output_path: Path | None = None) -> dict[str, Any]:
        """Run all benchmarks"""
        logger.info(f"\nRunning video benchmarks for {self.library}")
        videos = self.load_videos()

        # Collect metadata
        metadata = {
            "system_info": get_system_info(),
            "library_versions": get_library_versions(self.library),
            "thread_settings": verify_thread_settings(),
            "benchmark_params": {
                "num_videos": self.num_videos,
                "num_runs": self.num_runs,
                "max_warmup_iterations": self.max_warmup_iterations,
                "warmup_window": self.warmup_window,
                "warmup_threshold": self.warmup_threshold,
                "min_warmup_windows": self.min_warmup_windows,
            },
        }

        # Run benchmarks
        results = {}
        for transform_spec in TRANSFORM_SPECS:
            # Skip problematic transforms for Kornia due to compatibility issues
            problematic_transforms = ["Elastic", "GaussianBlur", "MotionBlur"]
            if transform_spec.name in problematic_transforms and self.library == "kornia":
                logger.info(f"Skipping {transform_spec.name} for {self.library} due to compatibility issues...")
                results[transform_spec.name] = {
                    "skipped": True,
                    "reason": "Compatibility issues with video tensors",
                    "throughput": None,
                    "time_per_video": None,
                    "warmup_iterations": 0,
                    "early_stopped": True,
                    "early_stop_reason": "Skipped",
                }
                continue

            logger.info(f"Benchmarking {transform_spec.name}...")
            results[transform_spec.name] = self.run_transform(transform_spec, videos)

        # Combine results and metadata
        output = {
            "metadata": metadata,
            "results": results,
        }

        # Save results
        if output_path:
            with output_path.open("w") as f:
                json.dump(output, f, indent=2)

        return output


def main() -> None:
    """Command-line entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run video augmentation benchmarks")
    parser.add_argument("-l", "--library", required=True, help="Library to benchmark")
    parser.add_argument("-d", "--data-dir", required=True, help="Directory containing videos")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("-n", "--num-videos", type=int, default=50, help="Number of videos to process")
    parser.add_argument("-r", "--num-runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--max-warmup", type=int, default=100, help="Maximum warmup iterations")
    parser.add_argument("--warmup-window", type=int, default=5, help="Window size for variance check")
    parser.add_argument("--warmup-threshold", type=float, default=0.05, help="Variance stability threshold")
    parser.add_argument("--min-warmup-windows", type=int, default=3, help="Minimum windows to check")

    args = parser.parse_args()

    runner = VideoBenchmarkRunner(
        library=args.library,
        data_dir=args.data_dir,
        num_videos=args.num_videos,
        num_runs=args.num_runs,
        max_warmup_iterations=args.max_warmup,
        warmup_window=args.warmup_window,
        warmup_threshold=args.warmup_threshold,
        min_warmup_windows=args.min_warmup_windows,
    )

    runner.run(Path(args.output))


if __name__ == "__main__":
    main()
