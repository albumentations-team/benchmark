from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

from tqdm import tqdm

from benchmark.term import tqdm_kwargs
from benchmark.utils import get_image_loader, get_video_loader, make_multichannel_loader

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMediaLoader:
    library: str
    data_dir: Path
    media: Literal["image", "video"]
    num_items: int
    num_channels: int = 3

    def load(self) -> list[Any]:
        if self.media == "image":
            return self._load_images()
        return self._load_videos()

    def _load_images(self) -> list[Any]:
        loader = get_image_loader(self.library)
        if self.num_channels != 3:
            loader = make_multichannel_loader(loader, self.num_channels)

        image_paths = sorted(self.data_dir.rglob("*.*"))
        logger.info("Found %d image paths in %s (searching recursively)", len(image_paths), self.data_dir)
        images: list[Any] = []

        with tqdm(
            image_paths,
            desc=f"Load images ({self.library}, {self.num_channels}ch)",
            unit="img",
            **tqdm_kwargs(),
        ) as pbar:
            for path in pbar:
                try:
                    import cv2

                    img_check = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if img_check is None:
                        continue
                    if img_check.ndim < 3 or img_check.shape[2] < 3:
                        continue

                    images.append(loader(path))
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
        torch_module: Any | None = None
        try:
            torch_module = import_module("torch")
            device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
            gpu_available = torch_module.cuda.is_available()
        except ImportError:
            device = None
            gpu_available = False

        video_paths: list[Path] = []
        for ext in ["mp4", "avi", "mov"]:
            video_paths.extend(self.data_dir.rglob(f"*.{ext}"))
        video_paths = sorted(video_paths)
        logger.info("Found %d video files in %s (including subdirectories)", len(video_paths), self.data_dir)

        videos: list[Any] = []
        loader = get_video_loader(self.library)

        with tqdm(video_paths, desc=f"Load videos ({self.library})", unit="video", **tqdm_kwargs()) as pbar:
            for path in pbar:
                try:
                    video = loader(path)
                    if torch_module and isinstance(video, torch_module.Tensor) and gpu_available:
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

        if torch_module and gpu_available:
            allocated = torch_module.cuda.memory_allocated() / (1024**3)
            total = torch_module.cuda.get_device_properties(torch_module.cuda.current_device()).total_memory / (1024**3)
            logger.info("GPU memory: %.2fGB / %.2fGB", allocated, total)

        return videos
