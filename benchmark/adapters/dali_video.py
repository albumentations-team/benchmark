from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from benchmark.decoders import DecoderUnavailableError
from benchmark.results import summarize_runs, unsupported_result
from benchmark.term import tqdm_kwargs

if TYPE_CHECKING:
    from pathlib import Path


def _build_transform(name: str, videos: Any, params: dict[str, Any]) -> Any:
    from nvidia.dali import fn, types

    if name == "Resize":
        return fn.resize(videos, resize_x=params["target_size"], resize_y=params["target_size"])
    if name == "CenterCrop224":
        return fn.crop(videos, crop=(params["height"], params["width"]))
    if name == "HorizontalFlip":
        return fn.flip(videos, horizontal=1)
    if name == "VerticalFlip":
        return fn.flip(videos, vertical=1)
    if name == "Brightness":
        return fn.brightness_contrast(videos, brightness=1.0 + params["brightness_limit"])
    if name == "Contrast":
        return fn.brightness_contrast(videos, contrast=1.0 + params["contrast_limit"])
    if name == "Normalize":
        return fn.crop_mirror_normalize(
            videos,
            dtype=types.FLOAT,
            mean=[x * 255.0 for x in params["mean"]],
            std=[x * 255.0 for x in params["std"]],
        )
    msg = f"DALI video pipeline does not implement transform {name!r}"
    raise NotImplementedError(msg)


def run_dali_video_transform(
    *,
    transform_name: str,
    params: dict[str, Any],
    paths: list[Path],
    clip_length: int,
    batch_size: int,
    num_runs: int,
    workers: int,
    min_time: float = 0.0,
    min_batches: int = 1,
) -> dict[str, Any]:
    try:
        from nvidia.dali import fn, pipeline_def, types
    except ImportError as e:
        raise DecoderUnavailableError("NVIDIA DALI is not installed") from e

    try:

        @pipeline_def(batch_size=batch_size, num_threads=max(1, workers), device_id=0)
        def pipe() -> Any:
            videos = fn.readers.video(
                device="gpu",
                filenames=[str(path) for path in paths],
                sequence_length=clip_length,
                random_shuffle=False,
                image_type=types.RGB,
                dtype=types.UINT8,
            )
            return _build_transform(transform_name, videos, params)
    except NotImplementedError as e:
        return unsupported_result(str(e))

    try:
        pipeline = pipe()
        pipeline.build()
    except NotImplementedError as e:
        return unsupported_result(str(e))
    except Exception as e:
        return unsupported_result(f"DALI pipeline build failed: {type(e).__name__}: {e}")

    batches_per_epoch = max(1, (len(paths) + batch_size - 1) // batch_size)
    throughputs: list[float] = []
    times: list[float] = []

    try:
        _ = pipeline.run()
    except Exception as e:
        return unsupported_result(f"DALI warmup failed: {type(e).__name__}: {e}")

    for _ in tqdm(range(num_runs), desc=f"DALI ({transform_name})", leave=False, **tqdm_kwargs()):
        start = time.perf_counter()
        processed = 0
        batches = 0
        try:
            while True:
                for _ in range(batches_per_epoch):
                    _ = pipeline.run()
                    processed += batch_size
                    batches += 1
                if time.perf_counter() - start >= min_time and batches >= min_batches:
                    break
        except Exception as e:
            return unsupported_result(f"DALI run failed: {type(e).__name__}: {e}")
        elapsed = time.perf_counter() - start
        processed = min(processed, len(paths))
        times.append(elapsed)
        throughputs.append(processed / elapsed)

    result = summarize_runs(throughputs, times)
    result["clip_length"] = clip_length
    result["batch_size"] = batch_size
    result["min_time"] = min_time
    result["min_batches"] = min_batches
    result["decoder"] = "dali"
    return result
