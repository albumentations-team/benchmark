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

from .results import build_metadata, summarize_runs
from .term import configure_logging, tqdm_kwargs
from .utils import (
    get_image_loader,
    get_video_loader,
    make_multichannel_loader,
    time_transform,
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
            "item_label_singular": "image",
        },
        MediaType.VIDEO: {
            "num_items": 50,
            "max_warmup_iterations": 100,
            "warmup_subset_size": 3,
            "slow_threshold": 2.0,
            "min_iterations_before_stopping": 5,
            "max_time_per_transform": 120,
            "item_label": "videos",
            "item_label_singular": "video",
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
        num_channels: int = 3,
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
        self._item_label_singular: str = defaults["item_label_singular"]

        if media_type == MediaType.IMAGE:
            self._loader = get_image_loader(library)
            if num_channels != 3:
                self._loader = make_multichannel_loader(self._loader, num_channels)
        else:
            self._loader = get_video_loader(library)
        self.num_channels = num_channels

    def _time_media_simple(self, transform: Any, media: list[Any]) -> float:
        return time_transform(lambda x: self.call_fn(transform, x), media)

    def _time_media(self, transform: Any, media: list[Any]) -> float:
        return self._time_media_simple(transform, media)

    # ------------------------------------------------------------------
    # Media loading
    # ------------------------------------------------------------------

    def load_media(self) -> list[Any]:
        if self.media_type == MediaType.IMAGE:
            return self._load_images()
        return self._load_videos()

    def _load_images(self) -> list[Any]:
        image_paths = sorted(self.data_dir.rglob("*.*"))
        logger.info("Found %d image paths in %s (searching recursively)", len(image_paths), self.data_dir)
        images: list[Any] = []

        with tqdm(image_paths, desc="Loading RGB images", unit="img", **tqdm_kwargs()) as pbar:
            for path in pbar:
                try:
                    import cv2

                    img_check = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if img_check is None:
                        continue
                    # Check the on-disk image (always 3-channel RGB); the loader may
                    # later stack channels to produce num_channels > 3 in memory.
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

        logger.info("Loaded %d images for benchmarking", len(images))
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

        with tqdm(video_paths, desc="Loading videos", **tqdm_kwargs()) as pbar:
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

        logger.debug(
            "Warmup: %s (max %d iters, subset size %d)",
            transform_name,
            self.max_warmup_iterations,
            len(warmup_subset),
        )
        with tqdm(
            total=self.max_warmup_iterations,
            desc=f"Warmup {transform_name}",
            unit="iter",
            leave=False,
            **tqdm_kwargs(),
        ) as pbar:
            for i in range(self.max_warmup_iterations):
                elapsed = self._time_media(transform, warmup_subset)
                throughput = len(warmup_subset) / elapsed
                time_per_item = elapsed / len(warmup_subset)
                warmup_throughputs.append(throughput)

                total_time = time.time() - start_time

                if i >= self._min_iterations_before_stopping and time_per_item > self._slow_threshold:
                    reason = (
                        f"Transform too slow: {time_per_item:.3f} sec/{self._item_label_singular}"
                        f" > {self._slow_threshold} sec/{self._item_label_singular} threshold"
                    )
                    logger.warning("Warmup %s: early stopped — %s", transform_name, reason)
                    return (warmup_throughputs, time_per_item, True, reason)

                if total_time > self._max_time_per_transform:
                    reason = f"Transform timeout: {total_time:.1f} sec > {self._max_time_per_transform} sec limit"
                    logger.warning("Warmup %s: early stopped — %s", transform_name, reason)
                    return (warmup_throughputs, time_per_item, True, reason)

                if (
                    i >= self.warmup_window * self.min_warmup_windows
                    and len(warmup_throughputs) >= self.warmup_window * 2
                ):
                    recent_mean = np.mean(warmup_throughputs[-self.warmup_window :])
                    overall_mean = np.mean(warmup_throughputs)
                    relative_diff = abs(recent_mean - overall_mean) / overall_mean

                    if relative_diff < self.warmup_threshold:
                        logger.info(
                            "Warmup %s: stable after %d iters (%.1f %s/s)",
                            transform_name,
                            i + 1,
                            recent_mean,
                            self._item_label_singular,
                        )
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
            median = float(np.median(warmup_throughputs))
            std_throughput = float(np.std(warmup_throughputs, ddof=1)) if len(warmup_throughputs) > 1 else 0.0
            logger.warning(
                "%s: early stopped (%.1f %s/s from warmup) — %s",
                transform_name,
                median,
                self._item_label_singular,
                early_stop_reason,
            )
            return {
                "supported": True,
                "status": "ok",
                "warmup_iterations": len(warmup_throughputs),
                "throughputs": [],
                "median_throughput": median,
                "mean_throughput": float(np.mean(warmup_throughputs)) if warmup_throughputs else 0.0,
                "std_throughput": std_throughput,
                "cv_throughput": std_throughput / median if median > 0 else 0.0,
                "throughput_ci95": 0.0,
                "times": [],
                "mean_time": time_per_item,
                "std_time": 0.0,
                "num_successful_runs": 0,
                "variance_stable": False,
                "unstable": True,
                "unstable_reason": "early stopped during warmup",
                "early_stopped": True,
                "early_stop_reason": early_stop_reason,
            }

        throughputs: list[float] = []
        times: list[float] = []

        for _ in tqdm(range(self.num_runs), desc=f"Benchmarking {transform_name}", leave=False, **tqdm_kwargs()):
            elapsed = self._time_media(transform, media)
            throughput = len(media) / elapsed
            throughputs.append(throughput)
            times.append(elapsed)

        result = summarize_runs(throughputs, times)
        median_throughput = result["median_throughput"]
        std_throughput = result["std_throughput"]
        logger.info(
            "%s: %.1f %s/s (median, std=%.1f, %d runs)",
            transform_name,
            median_throughput,
            self._item_label_singular,
            std_throughput,
            self.num_runs,
        )

        result.update(
            {
                "warmup_iterations": len(warmup_throughputs),
                "variance_stable": True,
                "early_stopped": False,
                "early_stop_reason": None,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Transform filter
    # ------------------------------------------------------------------

    @staticmethod
    def filter_transforms(
        transforms: list[dict[str, Any]],
        names: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Keep only transforms whose name is in *names*.

        - None means keep all transforms.
        - [] means keep none.
        """
        if names is None:
            return transforms
        allowed = set(names)
        return [t for t in transforms if t["name"] in allowed]

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self, output_path: Path | None = None) -> dict[str, Any]:
        logger.info(
            "Running %s benchmarks for %s: %d %s, %d transforms, %d runs each",
            self.media_type.value,
            self.library,
            self.num_items,
            self._item_label,
            len(self.transforms),
            self.num_runs,
        )
        media = self.load_media()

        if self.media_type == MediaType.VIDEO:
            try:
                import torch

                if self.library == "kornia" and isinstance(media[0], torch.Tensor):
                    logger.info("Using %s precision for Kornia", media[0].dtype)
            except ImportError:
                pass

        num_key = f"num_{self._item_label}"
        metadata = build_metadata(
            scenario=f"{self.media_type.value}-manual",
            mode="micro",
            library=self.library,
            benchmark_params={
                num_key: self.num_items,
                "num_runs": self.num_runs,
                "max_warmup_iterations": self.max_warmup_iterations,
                "warmup_window": self.warmup_window,
                "warmup_threshold": self.warmup_threshold,
                "min_warmup_windows": self.min_warmup_windows,
                "num_channels": self.num_channels,
                "timer_backend": "simple",
            },
            timing_backend="perf_counter",
            measurement_scope="augmentation_only",
            data_source="memory",
            data_dir=self.data_dir,
            media=self.media_type.value,
            includes_decode=False,
            includes_collate=False,
            includes_gpu_transfer=self.media_type == MediaType.VIDEO and self.library == "kornia",
            includes_dataloader_workers=False,
            repo_root=Path(__file__).parent.parent,
        )

        if self.media_type == MediaType.VIDEO:
            try:
                import torch

                if isinstance(media[0], torch.Tensor):
                    metadata["precision"] = str(media[0].dtype)
                    logger.info("Using %s precision for %s", media[0].dtype, self.library)
            except ImportError:
                pass

        results: dict[str, Any] = {}
        for transform_dict in tqdm(
            self.transforms,
            desc=f"Transforms ({self.library})",
            unit="transform",
            **tqdm_kwargs(),
        ):
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
            logger.info("Results written to %s", output_path)

        return full_results


# ------------------------------------------------------------------
# Spec file loader (shared by CLI and direct entry points)
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

    if "__call__" not in module.__dict__:
        raise TypeError(f"Python file {specs_file} must define __call__ function")

    call_attr = module.__dict__["__call__"]
    if not callable(call_attr):
        raise TypeError("__call__ must be a callable function")

    if not hasattr(module, "TRANSFORMS"):
        raise ValueError(f"Python file {specs_file} must define TRANSFORMS list")

    for i, t in enumerate(module.TRANSFORMS):
        if not isinstance(t, dict):
            raise TypeError(f"TRANSFORMS[{i}] must be a dictionary")

        required_keys = {"name", "transform"}
        missing = required_keys - t.keys()
        if missing:
            raise ValueError(f"TRANSFORMS[{i}] missing keys: {missing}")

    return module.LIBRARY, call_attr, module.TRANSFORMS


# ------------------------------------------------------------------
# Direct entry point for simple perf_counter micro timing
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
    parser.add_argument(
        "--num-channels",
        type=int,
        default=3,
        help=(
            "Number of image channels. Must be a multiple of 3. "
            "Values > 3 stack the RGB image to synthesize multi-channel data (default: 3)"
        ),
    )

    args = parser.parse_args()

    _log_level = logging.DEBUG if os.environ.get("BENCHMARK_VERBOSE") == "1" else logging.INFO
    configure_logging(_log_level, fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s")

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
        num_channels=args.num_channels,
    )

    runner.run(args.output)


if __name__ == "__main__":
    main()
