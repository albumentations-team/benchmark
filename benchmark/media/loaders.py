from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tqdm import tqdm

from benchmark.decoders.video import decode_video
from benchmark.term import tqdm_kwargs
from benchmark.utils import get_image_loader, get_video_loader, make_multichannel_loader

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class BenchmarkMediaLoader:
    library: str
    data_dir: Path
    media: Literal["image", "video"]
    num_items: int
    num_channels: int = 3
    clip_length: int = 16

    def load(self) -> list[Any]:
        if self.media == "image":
            return self._load_images()
        return self._load_videos()

    def _load_images(self) -> list[Any]:
        loader = get_image_loader(self.library)
        if self.num_channels != 3:
            loader = make_multichannel_loader(loader, self.num_channels)

        image_paths = sorted(path for path in self.data_dir.rglob("*") if _is_candidate_image(path))
        logger.info("Found %d image paths in %s (searching recursively)", len(image_paths), self.data_dir)
        images: list[Any] = []
        invalid_files = 0
        non_rgb_files = 0
        load_errors = 0

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
                        invalid_files += 1
                        continue
                    if img_check.ndim < 3 or img_check.shape[2] < 3:
                        non_rgb_files += 1
                        continue

                    images.append(loader(path))
                    if len(images) >= self.num_items:
                        break
                except Exception:
                    load_errors += 1
                    continue

                pbar.set_postfix({"loaded": len(images)})

        if not images:
            raise ValueError("No valid RGB images found in the directory (only RGB images are used for benchmarking)")

        if len(images) < self.num_items:
            logger.warning(
                "Only found %d valid RGB images after scanning %d candidate files, requested %d",
                len(images),
                len(image_paths),
                self.num_items,
            )

        logger.info(
            "Loaded %d images for benchmarking (invalid=%d, non_rgb=%d, load_errors=%d)",
            len(images),
            invalid_files,
            non_rgb_files,
            load_errors,
        )
        return images

    def _load_videos(self) -> list[Any]:
        video_paths: list[Path] = []
        for ext in ["mp4", "avi", "mov"]:
            video_paths.extend(self.data_dir.rglob(f"*.{ext}"))
        video_paths = sorted(video_paths)
        logger.info("Found %d video files in %s (including subdirectories)", len(video_paths), self.data_dir)

        videos: list[Any] = []
        progress_desc = f"Load videos ({self.library}, {self.clip_length}f)"
        with tqdm(video_paths, desc=progress_desc, unit="video", **tqdm_kwargs()) as pbar:
            for path in pbar:
                try:
                    video = self._load_video_clip(path)
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

        return videos

    def _load_video_clip(self, path: Path) -> Any:
        if self.library in {"torchvision", "kornia"}:
            import numpy as np

            clip = decode_video("opencv", path, self.clip_length).frames
            clip = np.ascontiguousarray(clip)
            try:
                import torch
            except ImportError:
                tensor = np.ascontiguousarray(clip.transpose(0, 3, 1, 2))
                if self.library == "torchvision":
                    return tensor
                return np.ascontiguousarray((tensor.astype(np.float32) / 255.0).astype(np.float16))

            tensor = torch.from_numpy(clip).permute(0, 3, 1, 2)
            if self.library == "torchvision":
                return tensor.contiguous()
            return tensor.float() / 255.0

        try:
            clip = decode_video("opencv", path, self.clip_length).frames
        except Exception:
            return get_video_loader(self.library)(path)
        return clip


def _is_candidate_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
