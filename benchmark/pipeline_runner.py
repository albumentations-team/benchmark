from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
from tqdm import tqdm

from benchmark.decoders import decode_video
from benchmark.results import build_metadata, summarize_runs, unsupported_result, write_results
from benchmark.runner import BenchmarkRunner, load_from_python_file
from benchmark.term import tqdm_kwargs
from benchmark.utils import get_image_loader, make_multichannel_loader

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

if TYPE_CHECKING:
    from collections.abc import Callable


def _torch_cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _video_clip_for_library(path: Path, library: str, clip_length: int) -> Any:
    clip = decode_video("opencv", path, clip_length).frames
    if library in {"torchvision", "kornia"}:
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)
        return tensor.float() / 255.0
    return clip


class _PathDataset:
    def __init__(
        self,
        *,
        paths: list[Path],
        media: str,
        library: str,
        num_channels: int,
        clip_length: int,
    ) -> None:
        self.paths = paths
        self.media = media
        self.library = library
        self.num_channels = num_channels
        self.clip_length = clip_length
        self._image_loader: Callable[[Path], Any] | None
        if media == "image":
            loader = get_image_loader(library)
            self._image_loader = make_multichannel_loader(loader, num_channels) if num_channels != 3 else loader
        else:
            self._image_loader = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        path = self.paths[index]
        if self.media == "image":
            if self._image_loader is None:
                msg = "Image loader was not initialized"
                raise RuntimeError(msg)
            return self._image_loader(path)
        return _video_clip_for_library(path, self.library, self.clip_length)


class _SimpleDataLoader:
    def __init__(self, dataset: _PathDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Any:
        for start in range(0, len(self.dataset), self.batch_size):
            yield [self.dataset[index] for index in range(start, min(start + self.batch_size, len(self.dataset)))]


class PipelineBenchmarkRunner:
    def __init__(
        self,
        *,
        library: str,
        data_dir: Path,
        output_file: Path,
        transforms: list[dict[str, Any]],
        call_fn: Callable[[Any, Any], Any],
        media: str,
        scenario: str,
        num_items: int | None = None,
        num_runs: int = 5,
        batch_size: int = 32,
        workers: int = 0,
        num_channels: int = 3,
        clip_length: int = 16,
        min_time: float = 0.0,
        min_batches: int = 1,
    ) -> None:
        self.library = library
        self.data_dir = data_dir
        self.output_file = output_file
        self.transforms = transforms
        self.call_fn = call_fn
        self.media = media
        self.scenario = scenario
        self.num_items = num_items
        self.num_runs = num_runs
        self.batch_size = batch_size
        self.workers = workers
        self.num_channels = num_channels
        self.clip_length = clip_length
        self.min_time = min_time
        self.min_batches = min_batches

    def _paths(self) -> list[Path]:
        extensions = _IMAGE_EXTENSIONS if self.media == "image" else _VIDEO_EXTENSIONS
        paths = sorted(path for path in self.data_dir.rglob("*") if path.suffix.lower() in extensions)
        if self.num_items is not None:
            paths = paths[: self.num_items]
        if not paths:
            raise ValueError(f"No {self.media} files found in {self.data_dir}")
        return paths

    def _loader(self, paths: list[Path]) -> Any:
        try:
            from torch.utils.data import DataLoader
        except ImportError as e:
            if self.library in {"torchvision", "kornia"}:
                msg = f"Pipeline benchmarks for {self.library} require torch"
                raise RuntimeError(msg) from e
            dataset = _PathDataset(
                paths=paths,
                media=self.media,
                library=self.library,
                num_channels=self.num_channels,
                clip_length=self.clip_length,
            )
            return _SimpleDataLoader(dataset, self.batch_size)

        dataset = _PathDataset(
            paths=paths,
            media=self.media,
            library=self.library,
            num_channels=self.num_channels,
            clip_length=self.clip_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=list,
            pin_memory=self.library in {"torchvision", "kornia"},
        )

    def _prepare_item(self, item: Any) -> Any:
        if self.media != "video" or self.library not in {"torchvision", "kornia"}:
            return item
        try:
            import torch

            if isinstance(item, torch.Tensor) and torch.cuda.is_available():
                return item.pin_memory().to("cuda", non_blocking=True).half()
        except ImportError:
            return item
        return item

    def _apply_transform_to_batch(self, transform: Any, batch: list[Any]) -> None:
        for item in batch:
            _ = self.call_fn(transform, self._prepare_item(item))

    def _run_transform(self, transform_dict: dict[str, Any], paths: list[Path]) -> dict[str, Any]:
        if self.library == "dali":
            from benchmark.adapters.dali_video import run_dali_video_transform
            from benchmark.decoders import DecoderUnavailableError

            try:
                return run_dali_video_transform(
                    transform_name=str(transform_dict["name"]),
                    params=dict(transform_dict["transform"] or {}),
                    paths=paths,
                    clip_length=self.clip_length,
                    batch_size=self.batch_size,
                    num_runs=self.num_runs,
                    workers=self.workers,
                    min_time=self.min_time,
                    min_batches=self.min_batches,
                )
            except DecoderUnavailableError as e:
                return unsupported_result(str(e))

        transform = transform_dict["transform"]
        throughputs: list[float] = []
        times: list[float] = []

        # Warm the dataloader path once. This includes decode/load by design.
        try:
            warm_loader = self._loader(paths[: min(len(paths), self.batch_size)])
            for batch in warm_loader:
                self._apply_transform_to_batch(transform, batch)
                break
            _torch_cuda_synchronize()
        except Exception as e:
            return unsupported_result(f"Pipeline warmup failed: {type(e).__name__}: {e}")

        for _ in tqdm(range(self.num_runs), desc=f"Pipeline ({self.library})", leave=False, **tqdm_kwargs()):
            processed = 0
            batches = 0
            _torch_cuda_synchronize()
            start = time.perf_counter()
            try:
                while True:
                    loader = self._loader(paths)
                    for batch in loader:
                        self._apply_transform_to_batch(transform, batch)
                        processed += len(batch)
                        batches += 1
                    elapsed_so_far = time.perf_counter() - start
                    if elapsed_so_far >= self.min_time and batches >= self.min_batches:
                        break
                _torch_cuda_synchronize()
            except Exception as e:
                return unsupported_result(f"Pipeline run failed: {type(e).__name__}: {e}")
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            throughputs.append(processed / elapsed)

        return summarize_runs(throughputs, times)

    def run(self) -> dict[str, Any]:
        paths = self._paths()
        logger.info(
            "Running %s pipeline benchmark for %s: %d items, batch=%d, workers=%d",
            self.media,
            self.library,
            len(paths),
            self.batch_size,
            self.workers,
        )

        results: dict[str, Any] = {}
        for transform_dict in tqdm(self.transforms, desc=f"Pipeline transforms ({self.library})", **tqdm_kwargs()):
            transform_name = str(transform_dict["name"])
            try:
                results[transform_name] = self._run_transform(transform_dict, paths)
            except Exception as e:
                warn(f"Pipeline transform {transform_name} failed: {e}", stacklevel=2)
                results[transform_name] = unsupported_result(f"{type(e).__name__}: {e}")

        payload = {
            "metadata": build_metadata(
                scenario=self.scenario,
                mode="pipeline",
                library=self.library,
                benchmark_params={
                    f"num_{self.media}s": len(paths),
                    "num_runs": self.num_runs,
                    "batch_size": self.batch_size,
                    "workers": self.workers,
                    "min_time": self.min_time,
                    "min_batches": self.min_batches,
                    "num_channels": self.num_channels,
                    "clip_length": self.clip_length if self.media == "video" else None,
                    "decode_included": True,
                    "decoder": "opencv" if self.media == "video" and self.library != "dali" else self.library,
                },
                timing_backend="dali_pipeline" if self.library == "dali" else "perf_counter",
                measurement_scope="decode_dataloader_augment",
                data_source="disk",
                data_dir=self.data_dir,
                media=self.media,
                includes_decode=True,
                includes_collate=True,
                includes_gpu_transfer=self.media == "video" and self.library in {"torchvision", "kornia"},
                includes_dataloader_workers=self.workers > 0,
                repo_root=Path(__file__).parent.parent,
            ),
            "results": results,
        }
        write_results(self.output_file, payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataloader/pipeline augmentation benchmarks")
    parser.add_argument("--specs-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media", choices=["image", "video"], default="image")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--num-items", type=int)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-time", type=float, default=0.0)
    parser.add_argument("--min-batches", type=int, default=1)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--clip-length", type=int, default=16)
    args = parser.parse_args()

    library, call_fn, transforms = load_from_python_file(args.specs_file)
    filter_env = os.environ.get("BENCHMARK_TRANSFORMS_FILTER", "").strip()
    if filter_env:
        filter_names = [name.strip() for name in filter_env.split(",") if name.strip()]
        transforms = BenchmarkRunner.filter_transforms(transforms, filter_names)

    runner = PipelineBenchmarkRunner(
        library=library,
        data_dir=args.data_dir,
        output_file=args.output,
        transforms=transforms,
        call_fn=call_fn,
        media=args.media,
        scenario=args.scenario,
        num_items=args.num_items,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        workers=args.workers,
        min_time=args.min_time,
        min_batches=args.min_batches,
        num_channels=args.num_channels,
        clip_length=args.clip_length,
    )
    runner.run()


if __name__ == "__main__":
    main()
