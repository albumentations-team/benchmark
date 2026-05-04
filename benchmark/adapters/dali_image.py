from __future__ import annotations

import subprocess
import time
from shutil import which
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from benchmark.decoders import DecoderUnavailableError
from benchmark.results import summarize_runs, unsupported_result
from benchmark.term import tqdm_kwargs

if TYPE_CHECKING:
    from pathlib import Path


_SUPPORTED_IMAGE_TRANSFORMS = {
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


def _random_crop_224(images: Any) -> Any:
    from nvidia.dali import fn

    images = fn.resize(images, resize_shorter=256, interp_type=_interp_type("bilinear"))
    return fn.crop(
        images,
        crop=(224, 224),
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
    )


def _normalize(images: Any) -> Any:
    from nvidia.dali import fn, types

    return fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
        std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
    )


def _apply_image_transform(name: str, images: Any, params: dict[str, Any]) -> Any:
    from nvidia.dali import fn

    if name == "Normalize":
        return images
    if name == "Resize":
        target_size = int(params["target_size"])
        return fn.resize(
            images,
            resize_x=target_size,
            resize_y=target_size,
            interp_type=_interp_type(params.get("interpolation")),
        )
    if name == "RandomCrop224":
        return _random_crop_224(images)
    if name == "RandomResizedCrop":
        return fn.random_resized_crop(
            images,
            size=tuple(params["size"]),
            random_area=tuple(params["scale"]),
            random_aspect_ratio=tuple(params["ratio"]),
            interp_type=_interp_type(params.get("interpolation")),
        )
    if name == "HorizontalFlip":
        return fn.flip(images, horizontal=1)
    if name == "VerticalFlip":
        return fn.flip(images, vertical=1)
    if name == "Pad":
        padding = int(params["padding"])
        return fn.pad(
            images,
            axes=(0, 1),
            shape=(224 + 2 * padding, 224 + 2 * padding),
            fill_value=params.get("fill", 0),
        )
    if name == "Rotate":
        return fn.rotate(
            images,
            angle=_uniform(params["angle_range"]),
            keep_size=True,
            fill_value=params.get("fill", 0),
            interp_type=_interp_type(params.get("interpolation")),
        )
    if name == "Affine":
        scale = float(params["scale"])
        shift_x, shift_y = params["shift"]
        return fn.warp_affine(
            images,
            matrix=[1.0 / scale, 0.0, -float(shift_x), 0.0, 1.0 / scale, -float(shift_y)],
            size=(224, 224),
            fill_value=params.get("fill", 0),
            interp_type=_interp_type(params.get("interpolation")),
        )
    if name == "Shear":
        shear = float(params["shear"]) / 57.29577951308232
        return fn.warp_affine(images, matrix=[1.0, shear, 0.0, 0.0, 1.0, 0.0], size=(224, 224))
    if name == "Brightness":
        return fn.brightness_contrast(images, brightness=1.0 + float(params["brightness_limit"][0]))
    if name == "Contrast":
        return fn.brightness_contrast(images, contrast=1.0 + float(params["contrast_limit"][0]))
    if name in {"ColorJitter", "ColorJiggle"}:
        return fn.color_twist(
            images,
            brightness=_uniform(params.get("brightness", 1.0)),
            contrast=_uniform(params.get("contrast", 1.0)),
            saturation=_uniform(params.get("saturation", 1.0)),
            hue=_uniform(params.get("hue", 0.0)),
        )
    if name == "Hue":
        return fn.hue(images, hue=float(params["hue"]))
    if name == "Saturation":
        return fn.saturation(images, saturation=1.0 + float(params["saturation_factor"]))
    if name == "CLAHE":
        tiles_y, tiles_x = params["tile_grid_size"]
        return fn.clahe(
            images,
            clip_limit=float(params["clip_limit"][0]),
            tiles_x=int(tiles_x),
            tiles_y=int(tiles_y),
        )
    if name == "Equalize":
        return fn.equalize(images)
    if name == "GaussianBlur":
        return fn.gaussian_blur(images, sigma=float(params["sigma"]), window_size=tuple(params["kernel_size"]))
    if name == "GaussianNoise":
        return fn.noise.gaussian(images, mean=float(params["mean"]), stddev=float(params["std"]) * 255.0)
    if name == "SaltAndPepper":
        return fn.noise.salt_and_pepper(
            images,
            prob=_uniform(params["amount"]),
            salt_vs_pepper=_uniform(params["salt_vs_pepper"]),
        )
    if name == "Erasing":
        return fn.erase(images, anchor=(32, 32), shape=(64, 64), fill_value=params.get("fill", 0))
    if name == "JpegCompression":
        return fn.jpeg_compression_distortion(images, quality=int(params["quality"]))

    msg = f"DALI image pipeline does not implement transform {name!r}"
    raise NotImplementedError(msg)


def _build_image_recipe(spec: dict[str, Any], encoded: Any) -> Any:
    from nvidia.dali import fn, types

    name = str(spec["name"])
    params = dict(spec["params"])
    if name not in _SUPPORTED_IMAGE_TRANSFORMS:
        msg = f"DALI image pipeline does not implement transform {name!r}"
        raise NotImplementedError(msg)

    images = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
    if name == "RandomResizedCrop":
        images = _apply_image_transform(name, images, params)
    elif name != "RandomCrop224":
        images = _random_crop_224(images)
        images = _apply_image_transform(name, images, params)
    else:
        images = _apply_image_transform(name, images, params)
    return _normalize(images)


def _nvidia_smi_used_memory_bytes() -> int | None:
    nvidia_smi = which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - executable is resolved with shutil.which for local GPU telemetry.
            [
                nvidia_smi,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    first = completed.stdout.strip().splitlines()[0].strip()
    if not first:
        return None
    try:
        return int(first) * 1024 * 1024
    except ValueError:
        return None


def run_dali_image_transform(
    *,
    transform_name: str,
    spec: dict[str, Any],
    paths: list[Path],
    batch_size: int,
    num_runs: int,
    workers: int,
    min_time: float = 0.0,
    min_batches: int = 1,
) -> dict[str, Any]:
    spec_name = str(spec.get("name", ""))
    if spec_name not in _SUPPORTED_IMAGE_TRANSFORMS:
        return unsupported_result(f"DALI image pipeline does not implement transform {spec_name!r}")

    try:
        from nvidia.dali import fn, pipeline_def
    except ImportError as e:
        raise DecoderUnavailableError("NVIDIA DALI is not installed") from e

    try:

        @pipeline_def(batch_size=batch_size, num_threads=max(1, workers), device_id=0)
        def pipe() -> Any:
            encoded, _ = fn.readers.file(files=[str(path) for path in paths], random_shuffle=False)
            return _build_image_recipe(spec, encoded)

    except NotImplementedError as e:
        return unsupported_result(str(e))

    try:
        pipeline = pipe()
        pipeline.build()
    except NotImplementedError as e:
        return unsupported_result(str(e))
    except Exception as e:
        return unsupported_result(f"DALI image pipeline build failed: {type(e).__name__}: {e}")

    batches_per_epoch = max(1, (len(paths) + batch_size - 1) // batch_size)
    throughputs: list[float] = []
    times: list[float] = []
    memory_runs: list[dict[str, int | None]] = []

    try:
        _ = pipeline.run()
    except Exception as e:
        return unsupported_result(f"DALI image warmup failed: {type(e).__name__}: {e}")

    for _ in tqdm(range(num_runs), desc=f"DALI image ({transform_name})", leave=False, **tqdm_kwargs()):
        memory_before = _nvidia_smi_used_memory_bytes()
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
            return unsupported_result(f"DALI image run failed: {type(e).__name__}: {e}")
        elapsed = time.perf_counter() - start
        processed = min(processed, len(paths))
        times.append(elapsed)
        throughputs.append(processed / elapsed)
        memory_after = _nvidia_smi_used_memory_bytes()
        memory_runs.append(
            {
                "gpu_memory_allocated_before_bytes": memory_before,
                "gpu_memory_allocated_after_bytes": memory_after,
                "gpu_peak_memory_allocated_bytes": memory_after,
                "gpu_peak_memory_reserved_bytes": memory_after,
            },
        )

    result = summarize_runs(throughputs, times)
    measured_values = [
        run["gpu_peak_memory_allocated_bytes"]
        for run in memory_runs
        if run["gpu_peak_memory_allocated_bytes"] is not None
    ]
    result["gpu_memory"] = {
        "device": "cuda",
        "measured": bool(measured_values),
        "runs": memory_runs,
        "peak_allocated_bytes": max(measured_values) if measured_values else None,
        "peak_reserved_bytes": max(measured_values) if measured_values else None,
    }
    result["batch_size"] = batch_size
    result["min_time"] = min_time
    result["min_batches"] = min_batches
    result["decoder"] = "dali"
    return result
