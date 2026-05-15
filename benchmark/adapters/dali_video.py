from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from benchmark.decoders import DecoderUnavailableError
from benchmark.results import summarize_runs, unsupported_result
from benchmark.term import tqdm_kwargs

if TYPE_CHECKING:
    from pathlib import Path


_SUPPORTED_VIDEO_TRANSFORMS = {
    "Resize",
    "RandomCrop224",
    "RandomResizedCrop",
    "HorizontalFlip",
    "VerticalFlip",
    "Pad",
    "Rotate",
    "Affine",
    "Shear",
    "Brightness",
    "Contrast",
    "ColorJitter",
    "ColorJiggle",
    "Hue",
    "Saturation",
    "CLAHE",
    "Equalize",
    "GaussianBlur",
    "GaussianNoise",
    "SaltAndPepper",
    "Erasing",
    "JpegCompression",
    "Normalize",
}

DALI_VIDEO_READER_BACKENDS = {"readers.video", "experimental.readers.video"}


def _read_video_batch(
    fn: Any,
    types: Any,
    *,
    paths: list[Path],
    clip_length: int,
    reader_backend: str,
) -> Any:
    kwargs = {
        "device": "gpu",
        "filenames": [str(path) for path in paths],
        "sequence_length": clip_length,
        "random_shuffle": False,
        "image_type": types.RGB,
    }
    if reader_backend == "readers.video":
        kwargs["dtype"] = types.UINT8
        return fn.readers.video(**kwargs)
    if reader_backend == "experimental.readers.video":
        return fn.experimental.readers.video(**kwargs)

    msg = f"Unsupported DALI video reader backend {reader_backend!r}"
    raise NotImplementedError(msg)


def _interp_type(interpolation: str | None) -> Any:
    from nvidia.dali import types

    if interpolation == "nearest":
        return types.INTERP_NN
    return types.INTERP_LINEAR


def _uniform(pair: tuple[float, float] | list[float] | float) -> Any:
    from nvidia.dali import fn

    if isinstance(pair, (tuple, list)):
        return fn.random.uniform(range=(float(pair[0]), float(pair[1])))
    return float(pair)


def _random_crop_224(videos: Any) -> Any:
    from nvidia.dali import fn

    videos = fn.resize(videos, resize_shorter=256, interp_type=_interp_type("bilinear"))
    return fn.crop(
        videos,
        crop=(224, 224),
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
    )


def _normalize(videos: Any) -> Any:
    from nvidia.dali import fn, types

    return fn.crop_mirror_normalize(
        videos,
        dtype=types.FLOAT,
        output_layout="FCHW",
        mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
        std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
    )


def _apply_video_transform(name: str, videos: Any, params: dict[str, Any]) -> Any:
    from nvidia.dali import fn

    if name == "Normalize":
        return videos
    if name == "Resize":
        target_size = int(params["target_size"])
        return fn.resize(
            videos,
            resize_x=target_size,
            resize_y=target_size,
            interp_type=_interp_type(params.get("interpolation")),
        )
    if name == "RandomCrop224":
        return _random_crop_224(videos)
    if name == "RandomResizedCrop":
        return fn.random_resized_crop(
            videos,
            size=tuple(params["size"]),
            random_area=tuple(params["scale"]),
            random_aspect_ratio=tuple(params["ratio"]),
            interp_type=_interp_type(params.get("interpolation")),
        )
    if name == "HorizontalFlip":
        return fn.flip(videos, horizontal=1)
    if name == "VerticalFlip":
        return fn.flip(videos, vertical=1)
    if name == "Pad":
        padding = int(params["padding"])
        return fn.pad(
            videos,
            axes=(1, 2),
            shape=(224 + 2 * padding, 224 + 2 * padding),
            fill_value=params.get("fill", 0),
        )
    if name == "Rotate":
        return fn.rotate(
            videos,
            angle=_uniform(params["angle_range"]),
            keep_size=True,
            fill_value=params.get("fill", 0),
            interp_type=_interp_type(params.get("interpolation")),
        )
    if name == "Affine":
        scale = float(params["scale"])
        shift_x, shift_y = params["shift"]
        return fn.warp_affine(
            videos,
            matrix=[1.0 / scale, 0.0, -float(shift_x), 0.0, 1.0 / scale, -float(shift_y)],
            size=(224, 224),
            fill_value=params.get("fill", 0),
            interp_type=_interp_type(params.get("interpolation")),
        )
    if name == "Shear":
        shear = float(params["shear"]) / 57.29577951308232
        return fn.warp_affine(videos, matrix=[1.0, shear, 0.0, 0.0, 1.0, 0.0], size=(224, 224))
    if name == "Brightness":
        return fn.brightness_contrast(videos, brightness=1.0 + float(params["brightness_limit"][0]))
    if name == "Contrast":
        return fn.brightness_contrast(videos, contrast=1.0 + float(params["contrast_limit"][0]))
    if name in {"ColorJitter", "ColorJiggle"}:
        return fn.color_twist(
            videos,
            brightness=_uniform(params.get("brightness", 1.0)),
            contrast=_uniform(params.get("contrast", 1.0)),
            saturation=_uniform(params.get("saturation", 1.0)),
            hue=_uniform(params.get("hue", 0.0)),
        )
    if name == "Hue":
        return fn.hue(videos, hue=float(params["hue"]))
    if name == "Saturation":
        return fn.saturation(videos, saturation=1.0 + float(params["saturation_factor"]))
    if name == "CLAHE":
        tiles_y, tiles_x = params["tile_grid_size"]
        return fn.clahe(
            videos,
            clip_limit=float(params["clip_limit"][0]),
            tiles_x=int(tiles_x),
            tiles_y=int(tiles_y),
        )
    if name == "Equalize":
        return fn.equalize(videos)
    if name == "GaussianBlur":
        return fn.gaussian_blur(videos, sigma=float(params["sigma"]), window_size=tuple(params["kernel_size"]))
    if name == "GaussianNoise":
        return fn.noise.gaussian(videos, mean=float(params["mean"]), stddev=float(params["std"]) * 255.0)
    if name == "SaltAndPepper":
        return fn.noise.salt_and_pepper(
            videos,
            prob=_uniform(params["amount"]),
            salt_vs_pepper=_uniform(params["salt_vs_pepper"]),
        )
    if name == "Erasing":
        return fn.erase(videos, anchor=(32, 32), shape=(64, 64), axes=(1, 2), fill_value=params.get("fill", 0))
    if name == "JpegCompression":
        return fn.jpeg_compression_distortion(videos, quality=int(params["quality"]))

    msg = f"DALI video pipeline does not implement transform {name!r}"
    raise NotImplementedError(msg)


def _build_video_recipe(spec: dict[str, Any], videos: Any) -> Any:
    name = str(spec.get("name", ""))
    params = dict(spec.get("params") or {})
    if name not in _SUPPORTED_VIDEO_TRANSFORMS:
        msg = f"DALI video pipeline does not implement transform {name!r}"
        raise NotImplementedError(msg)

    if name == "RandomResizedCrop":
        videos = _apply_video_transform(name, videos, params)
    elif name != "RandomCrop224":
        videos = _random_crop_224(videos)
        videos = _apply_video_transform(name, videos, params)
    else:
        videos = _apply_video_transform(name, videos, params)
    return _normalize(videos)


def _batch_clip_count(batch_index: int, *, total_items: int, batch_size: int) -> int:
    start = batch_index * batch_size
    remaining = total_items - start
    if remaining <= 0:
        return 0
    return min(batch_size, remaining)


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
    reader_backend: str = "readers.video",
) -> dict[str, Any]:
    spec_name = str(params.get("name", ""))
    if spec_name not in _SUPPORTED_VIDEO_TRANSFORMS:
        return unsupported_result(f"DALI video pipeline does not implement transform {spec_name!r}")
    if reader_backend not in DALI_VIDEO_READER_BACKENDS:
        return unsupported_result(f"Unsupported DALI video reader backend {reader_backend!r}")

    try:
        from nvidia.dali import fn, pipeline_def, types
    except ImportError as e:
        raise DecoderUnavailableError("NVIDIA DALI is not installed") from e

    try:

        @pipeline_def(batch_size=batch_size, num_threads=max(1, workers), device_id=0)
        def pipe() -> Any:
            videos = _read_video_batch(
                fn,
                types,
                paths=paths,
                clip_length=clip_length,
                reader_backend=reader_backend,
            )
            return _build_video_recipe(params, videos)

    except NotImplementedError as e:
        return unsupported_result(str(e))

    try:
        pipeline = pipe()
        pipeline.build()
    except NotImplementedError as e:
        return unsupported_result(str(e))
    except Exception as e:
        return unsupported_result(f"DALI video pipeline build failed: {type(e).__name__}: {e}")

    batches_per_epoch = max(1, (len(paths) + batch_size - 1) // batch_size)
    throughputs: list[float] = []
    times: list[float] = []

    try:
        _ = pipeline.run()
    except Exception as e:
        return unsupported_result(f"DALI video warmup failed: {type(e).__name__}: {e}")

    for _ in tqdm(range(num_runs), desc=f"DALI video ({transform_name})", leave=False, **tqdm_kwargs()):
        start = time.perf_counter()
        processed = 0
        batches = 0
        try:
            while True:
                for batch_index in range(batches_per_epoch):
                    _ = pipeline.run()
                    processed += _batch_clip_count(batch_index, total_items=len(paths), batch_size=batch_size)
                    batches += 1
                if time.perf_counter() - start >= min_time and batches >= min_batches:
                    break
        except Exception as e:
            return unsupported_result(f"DALI video run failed: {type(e).__name__}: {e}")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        throughputs.append(processed / elapsed)

    result = summarize_runs(throughputs, times)
    result["clip_length"] = clip_length
    result["batch_size"] = batch_size
    result["min_time"] = min_time
    result["min_batches"] = min_batches
    result["decoder"] = "dali"
    result["pipeline_backend"] = "dali_native"
    result["dali_video_reader_backend"] = reader_backend
    return result
