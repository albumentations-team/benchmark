from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from warnings import warn

import numpy as np
from tqdm import tqdm

from benchmark.decoders import decode_video
from benchmark.devices import (
    cuda_memory_allocated,
    cuda_memory_stats,
    move_transform_to_device,
    reset_peak_memory_stats,
    resolve_device,
    synchronize_device,
)
from benchmark.policy import slow_skip_config
from benchmark.results import build_metadata, summarize_runs, unsupported_result, write_results
from benchmark.runner import BenchmarkRunner
from benchmark.slow_threshold import is_slow_time_per_item, slow_threshold_info, slow_threshold_reason
from benchmark.specs import load_from_python_file
from benchmark.term import tqdm_kwargs
from benchmark.thread_policy import ThreadPolicy, apply_thread_policy, worker_init_for_policy
from benchmark.transform_filters import filter_transform_dicts_for_library_device
from benchmark.utils import get_image_loader, make_multichannel_loader, materialize_transform_output

logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
PipelineScope = Literal[
    "memory_dataloader_augment",
    "decode_dataloader_augment",
    "decode_dataloader_augment_batch_copy",
]
DeviceOption = Literal["none", "cuda", "mps", "auto"]

if TYPE_CHECKING:
    from collections.abc import Callable


def _torch_synchronize(device: str | None = None) -> None:
    synchronize_device(device)


def _video_clip_for_library(path: Path, library: str, clip_length: int) -> Any:
    clip = decode_video("opencv", path, clip_length).frames
    if library in {"torchvision", "kornia"}:
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)
        if library == "torchvision":
            return tensor.contiguous()
        return tensor.float() / 255.0
    return clip


class _PathDataset:
    def __init__(
        self,
        *,
        paths: list[Path] | None,
        preloaded: list[Any] | None,
        media: str,
        library: str,
        num_channels: int,
        clip_length: int,
        transform: Any | None,
        call_fn: Callable[[Any, Any], Any],
    ) -> None:
        self.paths = paths
        self.preloaded = preloaded
        self.media = media
        self.library = library
        self.num_channels = num_channels
        self.clip_length = clip_length
        self.transform = transform
        self.call_fn = call_fn
        self._image_loader: Callable[[Path], Any] | None
        if preloaded is not None:
            self._image_loader = None
        elif media == "image":
            loader = get_image_loader(library)
            self._image_loader = make_multichannel_loader(loader, num_channels) if num_channels != 3 else loader
        else:
            self._image_loader = None

    def __len__(self) -> int:
        if self.preloaded is not None:
            return len(self.preloaded)
        if self.paths is None:
            return 0
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        if self.preloaded is not None:
            item = self.preloaded[index]
            if self.transform is None:
                return item
            return self.call_fn(self.transform, item)
        if self.paths is None:
            msg = "Dataset has neither paths nor preloaded samples"
            raise RuntimeError(msg)
        path = self.paths[index]
        if self.media == "image":
            if self._image_loader is None:
                msg = "Image loader was not initialized"
                raise RuntimeError(msg)
            item = self._image_loader(path)
        else:
            item = _video_clip_for_library(path, self.library, self.clip_length)
        if self.transform is None:
            return item
        return self.call_fn(self.transform, item)


def _shutdown_loader_iterator(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _is_tensor_batch(batch: Any) -> bool:
    try:
        import torch

        if isinstance(batch, torch.Tensor):
            return True
    except ImportError:
        pass
    return isinstance(batch, np.ndarray)


def _batch_size(batch: Any) -> int:
    if not _is_tensor_batch(batch):
        msg = (
            "Pipeline recipes must return one fixed-shape tensor or ndarray per sample so PyTorch default collation "
            f"produces a single batched tensor/array; got {type(batch).__name__}"
        )
        raise TypeError(msg)

    shape = getattr(batch, "shape", ())
    if len(shape) == 0:
        msg = "Pipeline batch must have a leading batch dimension"
        raise TypeError(msg)
    return len(batch)


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
        pipeline_scope: PipelineScope = "decode_dataloader_augment",
        device: DeviceOption = "none",
        thread_policy: ThreadPolicy = "pipeline-default",
        slow_threshold_sec_per_item: float | None = None,
        slow_preflight_items: int | None = None,
        disable_slow_skip: bool = False,
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
        self.pipeline_scope = pipeline_scope
        self.device = device
        self.thread_policy: ThreadPolicy = thread_policy
        self.slow_threshold_sec_per_item = slow_threshold_sec_per_item
        self.slow_preflight_items = slow_preflight_items
        self.disable_slow_skip = disable_slow_skip
        self._last_preload_time: float | None = None
        self._last_device: str | None = None
        self._preloaded_items: list[Any] | None = None
        self._image_loader: Callable[[Path], Any] | None = None

    def _paths(self) -> list[Path]:
        extensions = _IMAGE_EXTENSIONS if self.media == "image" else _VIDEO_EXTENSIONS
        paths = sorted(path for path in self.data_dir.rglob("*") if path.suffix.lower() in extensions)
        if self.num_items is not None:
            paths = paths[: self.num_items]
        if not paths:
            raise ValueError(f"No {self.media} files found in {self.data_dir}")
        return paths

    def _load_item(self, path: Path) -> Any:
        if self.media == "image":
            if self._image_loader is None:
                loader = get_image_loader(self.library)
                self._image_loader = (
                    make_multichannel_loader(loader, self.num_channels) if self.num_channels != 3 else loader
                )
            return self._image_loader(path)
        return _video_clip_for_library(path, self.library, self.clip_length)

    def _preload_items(self, paths: list[Path]) -> list[Any] | None:
        if self.pipeline_scope != "memory_dataloader_augment":
            self._last_preload_time = None
            return None
        if self._preloaded_items is not None:
            return self._preloaded_items
        start = time.perf_counter()
        item_label = "images" if self.media == "image" else "clips"
        items = [
            self._load_item(path)
            for path in tqdm(
                paths,
                desc=f"Preload {item_label} ({self.library}, {self.pipeline_scope})",
                unit=self.media,
                **tqdm_kwargs(),
            )
        ]
        self._last_preload_time = time.perf_counter() - start
        self._preloaded_items = items
        return items

    def _loader(self, paths: list[Path], transform: Any | None, preloaded: list[Any] | None = None) -> Any:
        dataset = _PathDataset(
            paths=None if preloaded is not None else paths,
            preloaded=preloaded,
            media=self.media,
            library=self.library,
            num_channels=self.num_channels,
            clip_length=self.clip_length,
            transform=transform,
            call_fn=self.call_fn,
        )
        try:
            from torch.utils.data import DataLoader
        except ImportError as e:
            msg = "Pipeline benchmarks require torch DataLoader"
            raise RuntimeError(msg) from e

        multiprocessing_kwargs: dict[str, Any] = (
            {"multiprocessing_context": "fork"} if self.workers > 0 and hasattr(os, "fork") else {}
        )
        return DataLoader(
            cast("Any", dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=self._resolved_device() == "cuda" and self.pipeline_scope.endswith("_batch_copy"),
            worker_init_fn=worker_init_for_policy(self.thread_policy),
            **multiprocessing_kwargs,
        )

    def _resolved_device(self) -> str | None:
        device = resolve_device(self.device)
        self._last_device = device
        return device

    def _uses_gpu_batch_transform(self) -> bool:
        return (
            self.media in {"image", "video"}
            and self.library in {"torchvision", "kornia"}
            and self._resolved_device() is not None
        )

    def _split_gpu_transform(self, transform: Any) -> tuple[Any | None, Any]:
        if self.media == "video":
            return None, transform
        dataset_transform = getattr(transform, "cpu_transform", None)
        batch_transform = getattr(transform, "gpu_transform", transform)
        return dataset_transform, batch_transform

    def _to_tensor(self, item: Any, *, scale_uint8: bool = True) -> Any:
        import torch

        tensor = item if isinstance(item, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(np.asarray(item)))
        if scale_uint8 and tensor.dtype == torch.uint8:
            return tensor.float() / 255.0
        if scale_uint8:
            return tensor.float() if not tensor.is_floating_point() else tensor.contiguous()
        return tensor.contiguous()

    def _materialize_batch(self, batch: Any) -> int:
        batch_size = _batch_size(batch)
        if self.pipeline_scope != "decode_dataloader_augment_batch_copy":
            materialize_transform_output(batch)
            return batch_size

        stacked = self._to_tensor(batch)
        device = self._resolved_device()
        if device == "cuda":
            with suppress(RuntimeError):
                stacked = stacked.pin_memory()
            stacked = stacked.to("cuda", non_blocking=True)
        elif device == "mps":
            stacked = stacked.to("mps")
        materialize_transform_output(stacked)
        return batch_size

    def _move_batch_to_resolved_device(self, batch: Any, *, scale_uint8: bool = True) -> Any:
        stacked = self._to_tensor(batch, scale_uint8=scale_uint8)
        device = self._resolved_device()
        if device == "cuda":
            with suppress(RuntimeError):
                stacked = stacked.pin_memory()
            return stacked.to("cuda", non_blocking=True)
        if device == "mps":
            return stacked.to("mps")
        return stacked

    def _apply_gpu_batch_transform(self, batch: Any, transform: Any) -> Any:
        if self.media == "video" and self.library == "torchvision":
            import torch

            return torch.stack([transform(clip) for clip in batch], dim=0)
        return transform(batch)

    def _materialize_gpu_batch(self, batch: Any, transform: Any) -> int:
        batch_size = _batch_size(batch)
        device_batch = self._move_batch_to_resolved_device(batch, scale_uint8=False)
        device_transform = move_transform_to_device(transform, self._last_device)
        output = self._apply_gpu_batch_transform(device_batch, device_transform)
        materialize_transform_output(output)
        _torch_synchronize(self._last_device)
        return batch_size

    def _run_loader_once(self, loader: Any, *, desc: str, transform: Any | None = None) -> tuple[int, int]:
        processed = 0
        batches = 0
        iterator = iter(loader)
        try:
            total_batches = len(loader) if hasattr(loader, "__len__") else None
            batch_iter = tqdm(iterator, total=total_batches, desc=desc, unit="batch", leave=False, **tqdm_kwargs())
            for batch in batch_iter:
                if transform is None:
                    processed += self._materialize_batch(batch)
                else:
                    processed += self._materialize_gpu_batch(batch, transform)
                batches += 1
        finally:
            _shutdown_loader_iterator(iterator)
        return processed, batches

    def _warm_loader_once(self, loader: Any, transform: Any | None = None) -> None:
        iterator = iter(loader)
        try:
            batch = next(iterator)
            if transform is None:
                self._materialize_batch(batch)
            else:
                self._materialize_gpu_batch(batch, transform)
        finally:
            _shutdown_loader_iterator(iterator)

    def _slow_skip_config(self) -> tuple[float, int, float]:
        return slow_skip_config(
            self.media,
            threshold_sec_per_item=self.slow_threshold_sec_per_item,
            preflight_items=self.slow_preflight_items,
        )

    def _preflight_limit(self, paths: list[Path], *, min_items: int = 1) -> int:
        _, preflight_items, _ = self._slow_skip_config()
        if not paths:
            return 0
        return max(1, min(max(preflight_items, min_items), len(paths)))

    def _preflight_items(self, paths: list[Path], preloaded: list[Any] | None) -> list[Any]:
        limit = self._preflight_limit(paths)
        if preloaded is not None:
            return preloaded[:limit]
        return [self._load_item(path) for path in paths[:limit]]

    def _preflight_slow_transform(
        self,
        *,
        transform_name: str,
        transform: Any,
        paths: list[Path],
        preloaded: list[Any] | None,
    ) -> dict[str, Any] | None:
        if self.disable_slow_skip:
            return None

        threshold, _, max_preflight_secs = self._slow_skip_config()
        uses_gpu_batch = self._uses_gpu_batch_transform()
        item_limit = 0
        gpu_items: list[Any] | None = None
        cpu_items: list[Any] = []
        if uses_gpu_batch:
            item_limit = self._preflight_limit(paths, min_items=self.batch_size)
            gpu_items = preloaded[:item_limit] if preloaded is not None else None
            if item_limit == 0 or (preloaded is not None and not gpu_items):
                return None
        else:
            cpu_items = self._preflight_items(paths, preloaded)
            if not cpu_items:
                return None

        start = time.perf_counter()
        if uses_gpu_batch:
            dataset_transform, batch_transform = self._split_gpu_transform(transform)
            loader = self._loader(paths[:item_limit], dataset_transform, gpu_items)
            processed, _ = self._run_loader_once(
                loader,
                desc=f"{transform_name} GPU preflight",
                transform=batch_transform,
            )
            item_count = processed
        else:
            for item in cpu_items:
                materialize_transform_output(self.call_fn(transform, item))
            item_count = len(cpu_items)
        elapsed = time.perf_counter() - start
        time_per_item = elapsed / item_count
        throughput = item_count / elapsed if elapsed > 0 else 0.0
        item_label = "video" if self.media == "video" else "image"
        unit = "vid/s" if self.media == "video" else "img/s"

        reason: str | None = None
        if is_slow_time_per_item(time_per_item, threshold):
            reason = slow_threshold_reason(transform_name, time_per_item, threshold, item_label)
        elif elapsed > max_preflight_secs:
            reason = f"{transform_name} pipeline preflight timeout: {elapsed:.1f} sec > {max_preflight_secs:.1f} sec"

        if reason is None:
            return None

        return {
            "status": "ok",
            "supported": True,
            "throughputs": [],
            "median_throughput": throughput,
            "mean_throughput": throughput,
            "std_throughput": 0.0,
            "cv_throughput": 0.0,
            "throughput_ci95": 0.0,
            "p50_throughput": throughput,
            "p75_throughput": throughput,
            "p90_throughput": throughput,
            "p95_throughput": throughput,
            "times": [],
            "mean_time": time_per_item,
            "std_time": 0.0,
            "num_successful_runs": 0,
            "variance_stable": False,
            "unstable": True,
            "unstable_reason": "early stopped during pipeline preflight",
            "early_stopped": True,
            "early_stop_reason": reason,
            **slow_threshold_info(threshold, unit),
            "preflight_items": item_count,
            "preflight_elapsed": elapsed,
        }

    def _run_transform(self, transform_dict: dict[str, Any], paths: list[Path]) -> dict[str, Any]:
        if self.library == "dali":
            from benchmark.decoders import DecoderUnavailableError

            self._last_device = "cuda" if self.device in {"cuda", "auto"} else None
            try:
                if self.media == "image":
                    from benchmark.adapters.dali_image import run_dali_image_transform

                    return run_dali_image_transform(
                        transform_name=str(transform_dict["name"]),
                        spec=dict(transform_dict["transform"] or {}),
                        paths=paths,
                        batch_size=self.batch_size,
                        num_runs=self.num_runs,
                        workers=self.workers,
                        min_time=self.min_time,
                        min_batches=self.min_batches,
                    )

                from benchmark.adapters.dali_video import run_dali_video_transform

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
        gpu_memory_runs: list[dict[str, int | None]] = []
        preloaded = self._preload_items(paths)
        gpu_batch_transform = self._uses_gpu_batch_transform()
        if gpu_batch_transform:
            dataset_transform, batch_transform = self._split_gpu_transform(transform)
        else:
            dataset_transform = transform
            batch_transform = None
        slow_result = self._preflight_slow_transform(
            transform_name=str(transform_dict["name"]),
            transform=transform,
            paths=paths,
            preloaded=preloaded,
        )
        if slow_result is not None:
            return slow_result

        # Warm the dataloader path once. This includes decode/load by design.
        try:
            warm_paths = paths[: min(len(paths), self.batch_size)]
            warm_preloaded = preloaded[: min(len(preloaded), self.batch_size)] if preloaded is not None else None
            warm_loader = self._loader(warm_paths, dataset_transform, warm_preloaded)
            self._warm_loader_once(warm_loader, batch_transform)
            del warm_loader
            _torch_synchronize(self._resolved_device())
        except Exception as e:
            return unsupported_result(f"Pipeline warmup failed: {type(e).__name__}: {e}")

        run_desc = f"{self.library} {self.pipeline_scope} w={self.workers} b={self.batch_size} {transform_dict['name']}"
        for run_index in tqdm(range(self.num_runs), desc=run_desc, leave=False, **tqdm_kwargs()):
            processed = 0
            batches = 0
            loader = self._loader(paths, dataset_transform, preloaded)
            device = self._resolved_device()
            _torch_synchronize(device)
            memory_before = cuda_memory_allocated(device)
            reset_peak_memory_stats(device)
            start = time.perf_counter()
            try:
                while True:
                    batch_desc = f"{run_desc} run={run_index + 1}/{self.num_runs}"
                    processed_epoch, batches_epoch = self._run_loader_once(
                        loader,
                        desc=batch_desc,
                        transform=batch_transform,
                    )
                    processed += processed_epoch
                    batches += batches_epoch
                    elapsed_so_far = time.perf_counter() - start
                    if elapsed_so_far >= self.min_time and batches >= self.min_batches:
                        break
                _torch_synchronize(device)
            except Exception as e:
                return unsupported_result(f"Pipeline run failed: {type(e).__name__}: {e}")
            finally:
                del loader
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            throughputs.append(processed / elapsed)
            if device == "cuda":
                memory_stats = cuda_memory_stats(device)
                memory_stats["gpu_memory_allocated_before_bytes"] = memory_before
                gpu_memory_runs.append(memory_stats)

        result = summarize_runs(throughputs, times)
        if gpu_memory_runs:
            peak_allocated_values = [
                value
                for value in (run["gpu_peak_memory_allocated_bytes"] for run in gpu_memory_runs)
                if value is not None
            ]
            peak_reserved_values = [
                value
                for value in (run["gpu_peak_memory_reserved_bytes"] for run in gpu_memory_runs)
                if value is not None
            ]
            result["gpu_memory"] = {
                "device": self._last_device or "none",
                "measured": bool(peak_allocated_values),
                "runs": gpu_memory_runs,
                "peak_allocated_bytes": max(peak_allocated_values) if peak_allocated_values else None,
                "peak_reserved_bytes": max(peak_reserved_values) if peak_reserved_values else None,
            }
        return result

    def run(self) -> dict[str, Any]:
        apply_thread_policy(self.thread_policy)
        paths = self._paths()
        logger.info(
            "Running %s pipeline benchmark for %s: scope=%s, %d items, batch=%d, workers=%d",
            self.media,
            self.library,
            self.pipeline_scope,
            len(paths),
            self.batch_size,
            self.workers,
        )

        results: dict[str, Any] = {}
        transform_desc = (
            f"Pipeline transforms ({self.library}, {self.pipeline_scope}, w={self.workers}, b={self.batch_size})"
        )
        for transform_dict in tqdm(self.transforms, desc=transform_desc, **tqdm_kwargs()):
            transform_name = str(transform_dict["name"])
            try:
                results[transform_name] = self._run_transform(transform_dict, paths)
            except Exception as e:
                warn(f"Pipeline transform {transform_name} failed: {e}", stacklevel=2)
                results[transform_name] = unsupported_result(f"{type(e).__name__}: {e}")

        includes_decode = self.pipeline_scope != "memory_dataloader_augment"
        includes_gpu_transfer = (
            self.pipeline_scope == "decode_dataloader_augment_batch_copy" and self._last_device is not None
        ) or (
            self.media in {"image", "video"}
            and self.library in {"torchvision", "kornia"}
            and self._last_device is not None
        )
        transform_on_device = (
            self.media in {"image", "video"}
            and self.library in {"torchvision", "kornia", "dali"}
            and self._last_device is not None
        )
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
                    "pipeline_scope": self.pipeline_scope,
                    "decode_included": includes_decode,
                    "decoder": "opencv" if self.media == "video" and self.library != "dali" else self.library,
                    "device": self._last_device or "none",
                    "device_option": self.device,
                    "transform_on_device": transform_on_device,
                    "includes_host_to_device_transfer": includes_gpu_transfer,
                    "video_randomness_scope": "per_clip_same_on_frames"
                    if self.media == "video" and self.library in {"torchvision", "kornia"}
                    else None,
                    "gpu_memory_peak_measured": self._last_device == "cuda",
                    "thread_policy": self.thread_policy,
                    "batch_collate": True,
                    "preload_time": self._last_preload_time,
                    "slow_skip_enabled": not self.disable_slow_skip,
                    "slow_threshold_sec_per_item": self._slow_skip_config()[0],
                    "slow_preflight_items": self._slow_skip_config()[1],
                },
                timing_backend="dali_pipeline" if self.library == "dali" else "perf_counter",
                measurement_scope=self.pipeline_scope,
                data_source="memory" if self.pipeline_scope == "memory_dataloader_augment" else "disk",
                data_dir=self.data_dir,
                media=self.media,
                includes_decode=includes_decode,
                includes_collate=True,
                includes_gpu_transfer=includes_gpu_transfer,
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
    parser.add_argument(
        "--pipeline-scope",
        choices=["memory_dataloader_augment", "decode_dataloader_augment", "decode_dataloader_augment_batch_copy"],
        default="decode_dataloader_augment",
    )
    parser.add_argument("--device", choices=["none", "cuda", "mps", "auto"], default="none")
    parser.add_argument(
        "--thread-policy",
        choices=["micro-single", "pipeline-default", "pipeline-single-worker"],
        default="pipeline-default",
    )
    parser.add_argument("--slow-threshold-sec-per-item", type=float)
    parser.add_argument("--slow-preflight-items", type=int)
    parser.add_argument("--disable-slow-skip", action="store_true")
    args = parser.parse_args()

    apply_thread_policy(args.thread_policy)

    library, call_fn, transforms = load_from_python_file(args.specs_file)
    filter_env = os.environ.get("BENCHMARK_TRANSFORMS_FILTER", "").strip()
    if filter_env:
        filter_names = [name.strip() for name in filter_env.split(",") if name.strip()]
        transforms = BenchmarkRunner.filter_transforms(transforms, filter_names)
    transforms = filter_transform_dicts_for_library_device(
        transforms,
        scenario=args.scenario,
        mode="pipeline",
        library=library,
        media=args.media,
        device=args.device,
    )

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
        pipeline_scope=args.pipeline_scope,
        device=args.device,
        thread_policy=args.thread_policy,
        slow_threshold_sec_per_item=args.slow_threshold_sec_per_item,
        slow_preflight_items=args.slow_preflight_items,
        disable_slow_skip=args.disable_slow_skip,
    )
    runner.run()


if __name__ == "__main__":
    main()
