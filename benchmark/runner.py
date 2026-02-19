import importlib
import importlib.util
import json
import logging
import os
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from warnings import warn

import numpy as np
from tqdm import tqdm

from .utils import (
    get_image_loader,
    get_library_versions,
    get_system_info,
    get_video_loader,
    time_transform,
    verify_thread_settings,
)

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


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class BenchmarkRunner:
    # Defaults differ between image and video modes
    _DEFAULTS: ClassVar[dict[MediaType, dict[str, Any]]] = {
        MediaType.IMAGE: {
            "num_items": 1000,
            "max_warmup_iterations": 1000,
            "warmup_subset_size": 10,
            "slow_threshold": 0.1,
            "min_iterations_before_stopping": 10,
            "max_time_per_transform": 60,
            "item_label": "images",
        },
        MediaType.VIDEO: {
            "num_items": 50,
            "max_warmup_iterations": 100,
            "warmup_subset_size": 3,
            "slow_threshold": 2.0,
            "min_iterations_before_stopping": 5,
            "max_time_per_transform": 120,
            "item_label": "videos",
        },
    }

    def __init__(
        self,
        library: str,
        data_dir: Path,
        transforms: list[dict[str, Any]],
        call_fn: Callable[[Any, Any], Any],
        media_type: MediaType = MediaType.IMAGE,
        num_items: int | None = None,
        num_runs: int = 5,
        max_warmup_iterations: int | None = None,
        warmup_window: int = 5,
        warmup_threshold: float = 0.05,
        min_warmup_windows: int = 3,
    ):
        self.library = library
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.call_fn = call_fn
        self.media_type = media_type
        self.num_runs = num_runs
        self.warmup_window = warmup_window
        self.warmup_threshold = warmup_threshold
        self.min_warmup_windows = min_warmup_windows

        defaults = self._DEFAULTS[media_type]
        self.num_items = num_items if num_items is not None else defaults["num_items"]
        self.max_warmup_iterations = (
            max_warmup_iterations if max_warmup_iterations is not None else defaults["max_warmup_iterations"]
        )
        self._warmup_subset_size: int = defaults["warmup_subset_size"]
        self._slow_threshold: float = defaults["slow_threshold"]
        self._min_iterations_before_stopping: int = defaults["min_iterations_before_stopping"]
        self._max_time_per_transform: int = defaults["max_time_per_transform"]
        self._item_label: str = defaults["item_label"]

        if media_type == MediaType.IMAGE:
            self._loader = get_image_loader(library)
        else:
            self._loader = get_video_loader(library)

    # ------------------------------------------------------------------
    # Media loading
    # ------------------------------------------------------------------

    def load_media(self) -> list[Any]:
        if self.media_type == MediaType.IMAGE:
            return self._load_images()
        return self._load_videos()

    def _load_images(self) -> list[Any]:
        image_paths = sorted(self.data_dir.glob("*.*"))
        images: list[Any] = []

        with tqdm(image_paths, desc="Loading RGB images") as pbar:
            for path in pbar:
                try:
                    import cv2

                    img_check = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if img_check is None:
                        continue
                    if img_check.ndim < 3 or img_check.shape[2] < 3:
                        continue

                    img = self._loader(path)
                    images.append(img)

                    if len(images) >= self.num_items:
                        break
                except Exception:  # noqa: S112
                    continue

                pbar.set_postfix({"loaded": len(images)})

        if not images:
            raise ValueError("No valid RGB images found in the directory (only RGB images are used for benchmarking)")

        if len(images) < self.num_items:
            logger.warning("Only found %d valid RGB images, requested %d", len(images), self.num_items)

        return images

    def _load_videos(self) -> list[Any]:
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gpu_available = torch.cuda.is_available()
        except ImportError:
            torch = None
            device = None
            gpu_available = False

        video_paths: list[Path] = []
        for ext in ["mp4", "avi", "mov"]:
            video_paths.extend(list(self.data_dir.rglob(f"*.{ext}")))
        video_paths = sorted(video_paths)
        logger.info("Found %d video files in %s (including subdirectories)", len(video_paths), self.data_dir)

        videos: list[Any] = []

        with tqdm(video_paths, desc="Loading videos") as pbar:
            for path in pbar:
                try:
                    video = self._loader(path)
                    if torch and isinstance(video, torch.Tensor) and gpu_available:
                        video = video.to(device, non_blocking=True) if self.library == "kornia" else video.to(device)
                    videos.append(video)

                    if len(videos) >= self.num_items:
                        break
                except Exception as e:
                    logger.warning("Error loading video %s: %s", path, e)
                    continue

                pbar.set_postfix({"loaded": len(videos)})

        if not videos:
            raise ValueError("No valid videos found in the directory (searched recursively)")

        if len(videos) < self.num_items:
            logger.warning(
                "Only %d valid videos found, which is less than the requested %d",
                len(videos),
                self.num_items,
            )

        logger.info("Loaded %d videos", len(videos))

        if torch and gpu_available:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
            logger.info("GPU memory: %.2fGB / %.2fGB", allocated, total)

        return videos

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _perform_warmup(
        self,
        transform: Any,
        transform_name: str,
        warmup_subset: list[Any],
    ) -> tuple[list[float], float, bool, str | None]:
        warmup_throughputs: list[float] = []
        time_per_item = 0.0
        start_time = time.time()

        with tqdm(total=self.max_warmup_iterations, desc=f"Warming up {transform_name}", leave=False) as pbar:
            for i in range(self.max_warmup_iterations):
                elapsed = time_transform(lambda x: self.call_fn(transform, x), warmup_subset)
                throughput = len(warmup_subset) / elapsed
                time_per_item = elapsed / len(warmup_subset)
                warmup_throughputs.append(throughput)

                total_time = time.time() - start_time

                if i >= self._min_iterations_before_stopping and time_per_item > self._slow_threshold:
                    return (
                        warmup_throughputs,
                        time_per_item,
                        True,
                        (
                            f"Transform too slow: {time_per_item:.3f} sec/{self._item_label[:-1]}"
                            f" > {self._slow_threshold} sec/{self._item_label[:-1]} threshold"
                        ),
                    )

                if total_time > self._max_time_per_transform:
                    return (
                        warmup_throughputs,
                        time_per_item,
                        True,
                        f"Transform timeout: {total_time:.1f} sec > {self._max_time_per_transform} sec limit",
                    )

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

        return warmup_throughputs, time_per_item, False, None

    # ------------------------------------------------------------------
    # Single transform
    # ------------------------------------------------------------------

    def run_transform(self, transform_dict: dict[str, Any], media: list[Any]) -> dict[str, Any]:
        transform = transform_dict["transform"]
        transform_name = transform_dict["name"]

        warmup_subset = media[: min(self._warmup_subset_size, len(media))]
        warmup_throughputs, time_per_item, early_stopped, early_stop_reason = self._perform_warmup(
            transform,
            transform_name,
            warmup_subset,
        )

        if early_stopped:
            return {
                "supported": True,
                "warmup_iterations": len(warmup_throughputs),
                "throughputs": [],
                "median_throughput": float(np.median(warmup_throughputs)),
                "std_throughput": float(np.std(warmup_throughputs, ddof=1)),
                "times": [],
                "mean_time": time_per_item,
                "std_time": 0.0,
                "variance_stable": False,
                "early_stopped": True,
                "early_stop_reason": early_stop_reason,
            }

        throughputs: list[float] = []
        times: list[float] = []

        for _ in tqdm(range(self.num_runs), desc=f"Benchmarking {transform_name}", leave=False):
            elapsed = time_transform(lambda x: self.call_fn(transform, x), media)
            throughput = len(media) / elapsed
            throughputs.append(throughput)
            times.append(elapsed)

        median_throughput = float(np.median(throughputs))
        std_throughput = float(np.std(throughputs, ddof=1))

        return {
            "supported": True,
            "warmup_iterations": len(warmup_throughputs),
            "throughputs": throughputs,
            "median_throughput": median_throughput,
            "std_throughput": std_throughput,
            "times": times,
            "mean_time": len(media) / median_throughput,
            "std_time": std_throughput / (median_throughput**2) * len(media),
            "variance_stable": True,
            "early_stopped": False,
            "early_stop_reason": None,
        }

    # ------------------------------------------------------------------
    # Transform filter
    # ------------------------------------------------------------------

    @staticmethod
    def filter_transforms(
        transforms: list[dict[str, Any]],
        names: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Keep only transforms whose name is in *names*. None means keep all."""
        if not names:
            return transforms
        allowed = set(names)
        return [t for t in transforms if t["name"] in allowed]

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self, output_path: Path | None = None) -> dict[str, Any]:
        logger.info("Running %s benchmarks for %s", self.media_type.value, self.library)
        media = self.load_media()

        if self.media_type == MediaType.VIDEO:
            try:
                import torch

                if self.library == "kornia" and isinstance(media[0], torch.Tensor):
                    logger.info("Using %s precision for Kornia", media[0].dtype)
            except ImportError:
                pass

        num_key = f"num_{self._item_label}"
        metadata: dict[str, Any] = {
            "system_info": get_system_info(),
            "library_versions": get_library_versions(self.library),
            "thread_settings": verify_thread_settings(),
            "benchmark_params": {
                num_key: self.num_items,
                "num_runs": self.num_runs,
                "max_warmup_iterations": self.max_warmup_iterations,
                "warmup_window": self.warmup_window,
                "warmup_threshold": self.warmup_threshold,
                "min_warmup_windows": self.min_warmup_windows,
            },
        }

        if self.media_type == MediaType.VIDEO:
            try:
                import torch

                if isinstance(media[0], torch.Tensor):
                    metadata["precision"] = str(media[0].dtype)
                    logger.info("Using %s precision for %s", media[0].dtype, self.library)
            except ImportError:
                pass

        results: dict[str, Any] = {}
        for transform_dict in tqdm(self.transforms, desc="Running transforms"):
            try:
                transform_name = transform_dict["name"]
                logger.info("Benchmarking %s...", transform_name)
                results[transform_name] = self.run_transform(transform_dict, media)
            except Exception as e:
                transform_name = transform_dict.get("name", "Unknown")
                warn(f"Transform {transform_name} failed: {e}", stacklevel=2)

        full_results: dict[str, Any] = {
            "metadata": metadata,
            "results": results,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(full_results, f, indent=2)

        return full_results


# ------------------------------------------------------------------
# Spec file loader (shared by CLI and legacy entry points)
# ------------------------------------------------------------------


def load_from_python_file(specs_file: Path) -> tuple[str, Callable[[Any, Any], Any], list[dict[str, Any]]]:
    """Load library name, __call__ function, and transforms from a Python file.

    The Python file must define:
    - LIBRARY: str (e.g., "albumentationsx")
    - __call__: function to apply transforms to images/videos
    - TRANSFORMS: list of dicts with 'name' and 'transform' keys

    Returns:
        tuple of (library name, __call__ function, list of transform dicts)
    """
    spec = importlib.util.spec_from_file_location("custom_transforms", specs_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load from {specs_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "LIBRARY"):
        raise ValueError(f"Python file {specs_file} must define LIBRARY string")

    if not callable(module):
        raise TypeError(f"Python file {specs_file} must define __call__ function")

    if not hasattr(module, "TRANSFORMS"):
        raise ValueError(f"Python file {specs_file} must define TRANSFORMS list")

    if not callable(module.__call__):
        raise TypeError("__call__ must be a callable function")

    for i, t in enumerate(module.TRANSFORMS):
        if not isinstance(t, dict):
            raise TypeError(f"TRANSFORMS[{i}] must be a dictionary")

        required_keys = {"name", "transform"}
        missing = required_keys - t.keys()
        if missing:
            raise ValueError(f"TRANSFORMS[{i}] missing keys: {missing}")

    return module.LIBRARY, module.__call__, module.TRANSFORMS


# ------------------------------------------------------------------
# Legacy entry point (kept so existing shell scripts still work)
# ------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run augmentation benchmarks")
    parser.add_argument(
        "-s",
        "--specs-file",
        type=Path,
        required=True,
        help="Python file defining LIBRARY and TRANSFORMS",
    )
    parser.add_argument("-d", "--data-dir", required=True, type=Path, help="Directory with images or videos")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("--media", choices=["image", "video"], default="image", help="Media type (default: image)")
    parser.add_argument("-n", "--num-items", type=int, help="Number of images/videos")
    parser.add_argument("-r", "--num-runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--max-warmup", type=int, help="Maximum warmup iterations")
    parser.add_argument("--warmup-window", type=int, default=5, help="Window size for variance check")
    parser.add_argument("--warmup-threshold", type=float, default=0.05, help="Variance stability threshold")
    parser.add_argument("--min-warmup-windows", type=int, default=3, help="Minimum windows to check")

    args = parser.parse_args()

    if not args.specs_file.exists():
        raise ValueError(f"Specs file {args.specs_file} does not exist")

    logger.info("Loading from %s", args.specs_file)
    library, call_fn, transforms = load_from_python_file(args.specs_file)
    logger.info("Library: %s", library)
    logger.info("Loaded %d transforms", len(transforms))

    # Optional transform filter (set by benchmark.cli to run only specific transforms)
    filter_env = os.environ.get("BENCHMARK_TRANSFORMS_FILTER", "").strip()
    if filter_env:
        filter_names = [n.strip() for n in filter_env.split(",") if n.strip()]
        transforms = BenchmarkRunner.filter_transforms(transforms, filter_names)
        logger.info("Filtered to %d transforms: %s", len(transforms), filter_names)

    media_type = MediaType.IMAGE if args.media == "image" else MediaType.VIDEO

    runner = BenchmarkRunner(
        library=library,
        data_dir=args.data_dir,
        transforms=transforms,
        call_fn=call_fn,
        media_type=media_type,
        num_items=args.num_items,
        num_runs=args.num_runs,
        max_warmup_iterations=args.max_warmup,
        warmup_window=args.warmup_window,
        warmup_threshold=args.warmup_threshold,
        min_warmup_windows=args.min_warmup_windows,
    )

    runner.run(args.output)


if __name__ == "__main__":
    main()
