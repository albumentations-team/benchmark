import importlib
import importlib.util
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

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
        transforms: list[dict[str, Any]],
        call_fn: Callable[[Any, Any], Any],
        num_videos: int = 50,
        num_runs: int = 5,
        max_warmup_iterations: int = 100,
        warmup_window: int = 5,
        warmup_threshold: float = 0.05,
        min_warmup_windows: int = 3,
    ):
        self.library = library
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.call_fn = call_fn
        self.num_videos = num_videos
        self.num_runs = num_runs
        self.max_warmup_iterations = max_warmup_iterations
        self.warmup_window = warmup_window
        self.warmup_threshold = warmup_threshold
        self.min_warmup_windows = min_warmup_windows

        # Get video loader for the library
        self.video_loader = get_video_loader(library)

    def load_videos(self) -> list[Any]:
        """Load videos using appropriate loader"""
        # Import torch only if needed (for GPU operations or tensor-based libraries)
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gpu_available = torch.cuda.is_available()
        except ImportError:
            torch = None
            device = None
            gpu_available = False

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
                    # Move video to GPU if available and it's a torch tensor
                    if torch and isinstance(video, torch.Tensor) and gpu_available:
                        # For Kornia, ensure we maintain float16 precision when moving to GPU
                        video = video.to(device, non_blocking=True) if self.library == "kornia" else video.to(device)
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
            logger.warning(
                f"Only {len(videos)} valid videos found, which is less than the requested {self.num_videos}",
            )

        logger.info(f"Loaded {len(videos)} videos")

        # Log GPU memory usage if available
        if torch and gpu_available:
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)  # GB
            logger.info(f"GPU memory: {allocated:.2f}GB / {total:.2f}GB")

        return videos

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
            warmup_subset: Subset of videos to use for warmup

        Returns:
            tuple containing:
            - list of throughput measurements
            - time per video from last measurement
            - whether early stopping occurred
            - reason for early stopping (if any)
        """
        warmup_throughputs = []
        time_per_video = 0.0  # Initialize here
        slow_transform_threshold = 2.0  # seconds per video
        min_iterations_before_stopping = 5
        max_time_per_transform = 120
        start_time = time.time()

        with tqdm(total=self.max_warmup_iterations, desc=f"Warming up {transform_name}", leave=False) as pbar:
            for i in range(self.max_warmup_iterations):
                elapsed = time_transform(lambda x: self.call_fn(transform, x), warmup_subset)
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

    def run_transform(self, transform_dict: dict[str, Any], videos: list[Any]) -> dict[str, Any]:
        """Run benchmark for a single transform"""
        transform = transform_dict["transform"]
        transform_name = transform_dict["name"]

        # Perform warmup
        warmup_subset = videos[: min(3, len(videos))]
        warmup_throughputs, time_per_video, early_stopped, early_stop_reason = self._perform_warmup(
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
                "mean_time": time_per_video,
                "std_time": 0.0,
                "variance_stable": False,
                "early_stopped": True,
                "early_stop_reason": early_stop_reason,
            }

        # Benchmark runs
        throughputs = []
        times = []

        for _ in tqdm(range(self.num_runs), desc=f"Benchmarking {transform_name}", leave=False):
            elapsed = time_transform(lambda x: self.call_fn(transform, x), videos)
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
        # Load videos
        videos = self.load_videos()
        logger.info(f"Loaded {len(videos)} videos")

        # Log precision information for Kornia
        try:
            import torch

            if self.library == "kornia" and isinstance(videos[0], torch.Tensor):
                logger.info(f"Using {videos[0].dtype} precision for Kornia")
        except ImportError:
            pass

        # Collect metadata
        metadata: dict[str, Any] = {
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

        # Add precision information for tensor-based libraries
        try:
            import torch

            if isinstance(videos[0], torch.Tensor):
                metadata["precision"] = str(videos[0].dtype)
                logger.info(f"Using {videos[0].dtype} precision for {self.library}")
        except ImportError:
            pass

        # Run benchmarks
        results = {}

        for transform_dict in self.transforms:
            transform_name = transform_dict["name"]
            logger.info(f"Benchmarking {transform_name}...")
            results[transform_name] = self.run_transform(transform_dict, videos)

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


def load_from_python_file(specs_file: Path) -> tuple[str, Callable[[Any, Any], Any], list[dict[str, Any]]]:
    """Load library name, __call__ function, and transforms from a Python file

    The Python file must define:
    - LIBRARY: str (e.g., "albumentationsx")
    - __call__: function to apply transforms to videos
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
    """Command-line entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run video augmentation benchmarks")
    parser.add_argument("-d", "--data-dir", required=True, help="Directory containing videos")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument(
        "-s",
        "--specs-file",
        type=Path,
        required=True,
        help="Python file defining LIBRARY and CUSTOM_TRANSFORMS",
    )
    parser.add_argument("-n", "--num-videos", type=int, default=50, help="Number of videos to process")
    parser.add_argument("-r", "--num-runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--max-warmup", type=int, default=100, help="Maximum warmup iterations")
    parser.add_argument("--warmup-window", type=int, default=5, help="Window size for variance check")
    parser.add_argument("--warmup-threshold", type=float, default=0.05, help="Variance stability threshold")
    parser.add_argument("--min-warmup-windows", type=int, default=3, help="Minimum windows to check")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for benchmarking")

    args = parser.parse_args()

    # Check if GPU is requested and available
    try:
        import torch

        if args.gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Falling back to CPU.")

        # Log GPU information if available
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
            logger.info(f"Total GPU Memory: {memory:.2f} GB")
    except ImportError:
        if args.gpu:
            logger.warning("GPU requested but PyTorch not installed. Falling back to CPU.")

    # Load library and transforms from file
    if not args.specs_file.exists():
        raise ValueError(f"Specs file {args.specs_file} does not exist")

    logger.info(f"Loading from {args.specs_file}")
    library, call_fn, transforms = load_from_python_file(args.specs_file)
    logger.info(f"Library: {library}")
    logger.info(f"Loaded {len(transforms)} transforms")

    runner = VideoBenchmarkRunner(
        library=library,
        data_dir=args.data_dir,
        transforms=transforms,
        call_fn=call_fn,
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
