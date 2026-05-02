import multiprocessing
import os
import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def read_img_cv2(path: Path) -> np.ndarray:
    """Read image using OpenCV (for AlbumentationsX)"""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_img_torch(path: Path) -> Any:  # torch.Tensor
    """Read image using torchvision"""
    import torchvision
    from torchvision.io import ImageReadMode

    return torchvision.io.read_image(str(path), mode=ImageReadMode.RGB)  # Shape: (C, H, W)


def read_img_kornia(path: Path) -> Any:  # torch.Tensor
    """Read image using kornia format"""
    return read_img_torch(path) / 255.0  # Keep as float32 on CPU


def read_img_pil(path: Path) -> Any:  # PIL.Image.Image
    """Read image using Pillow (RGB mode)"""
    from PIL import Image

    return Image.open(str(path)).convert("RGB")


def read_video_cv2(path: Path) -> np.ndarray:
    """Read video using OpenCV (for AlbumentationsX)

    Returns a 4D NumPy array with shape (num_frames, height, width, num_channels)
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to load video: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        raise ValueError(f"No frames found in video: {path}")

    return np.array(frames)


def read_video_torch(path: Path) -> Any:  # torch.Tensor
    """Read video using torchvision"""
    import torchvision

    # Returns tensor of shape (T, C, H, W)
    video, _, _ = torchvision.io.read_video(str(path))
    # Convert to (T, C, H, W) format
    return video.permute(0, 3, 1, 2)


def read_video_torch_float16(path: Path) -> Any:  # torch.Tensor
    """Read video using torchvision and convert to float16"""
    video = read_video_torch(path)
    return (video.float() / 255.0).half()  # Convert to float16


def read_video_kornia(path: Path) -> Any:  # torch.Tensor
    """Read video using kornia format"""
    video = read_video_torch(path)
    return (video.float() / 255.0).half()  # Convert to float16


def make_multichannel_loader(base_loader: Callable[[Path], Any], num_channels: int) -> Callable[[Path], Any]:
    """Wrap a base loader to produce images with num_channels by stacking the RGB source.

    num_channels must be a multiple of 3. The loader reads a standard RGB image and
    concatenates it (num_channels // 3) times along the channel axis.
    Supports numpy (H, W, C) and torch (C, H, W) outputs.
    """
    if num_channels % 3 != 0:
        raise ValueError(f"num_channels must be a multiple of 3, got {num_channels}")
    repeats = num_channels // 3

    def load(path: Path) -> Any:
        img = base_loader(path)
        try:
            import torch

            if isinstance(img, torch.Tensor):
                return torch.cat([img] * repeats, dim=0)  # (C, H, W) -> (num_channels, H, W)
        except ImportError:
            pass
        return np.concatenate([img] * repeats, axis=-1)  # (H, W, C) -> (H, W, num_channels)

    return load


def materialize_transform_output(output: Any) -> Any:
    """Force transform outputs to be realized before the timer stops."""
    if isinstance(output, np.ndarray):
        return np.ascontiguousarray(output)

    contiguous = getattr(output, "contiguous", None)
    if callable(contiguous):
        return contiguous()

    load = getattr(output, "load", None)
    if callable(load) and output.__class__.__module__.startswith("PIL."):
        load()

    return output


def make_contiguous_transform_output(output: Any) -> Any:
    """Return micro-benchmark outputs as contiguous arrays/tensors when possible."""
    if isinstance(output, np.ndarray):
        return np.ascontiguousarray(output)

    load = getattr(output, "load", None)
    if callable(load) and output.__class__.__module__.startswith("PIL."):
        load()
        return np.ascontiguousarray(output)

    contiguous = getattr(output, "contiguous", None)
    if callable(contiguous):
        return contiguous()

    return output


def time_transform(transform: Any, images: list[Any]) -> float:
    """Time the execution of a transform on a list of images"""
    import time

    start = time.perf_counter()

    for img in images:
        _ = make_contiguous_transform_output(transform(img))

    return time.perf_counter() - start


@cache
def get_image_loader(library: str) -> Callable[[Path], Any]:
    """Get the appropriate image loader for the library"""
    loaders = {
        "albumentationsx": read_img_cv2,
        "albumentations_mit": read_img_cv2,
        "ultralytics": read_img_cv2,
        "torchvision": read_img_torch,
        "kornia": read_img_kornia,
        "pillow": read_img_pil,
    }

    if library not in loaders:
        raise ValueError(f"Unsupported library: {library}. Supported libraries are: {list(loaders.keys())}")

    return loaders[library]


@cache
def get_video_loader(library: str) -> Callable[[Path], Any]:
    """Get the appropriate video loader for the library"""
    loaders = {
        "albumentationsx": read_video_cv2,
        "albumentations_mit": read_video_cv2,
        "torchvision": read_video_torch_float16,
        "kornia": read_video_kornia,
    }

    if library not in loaders:
        raise ValueError(f"Unsupported library for video: {library}. Supported libraries are: {list(loaders.keys())}")

    return loaders[library]


def verify_thread_settings() -> dict[str, Any]:
    """Verify single-thread settings across libraries"""
    import os

    thread_vars: dict[str, Any] = {
        "environment": {
            "OMP_NUM_THREADS": str(os.environ.get("OMP_NUM_THREADS")),
            "OPENBLAS_NUM_THREADS": str(os.environ.get("OPENBLAS_NUM_THREADS")),
            "MKL_NUM_THREADS": str(os.environ.get("MKL_NUM_THREADS")),
            "VECLIB_MAXIMUM_THREADS": str(os.environ.get("VECLIB_MAXIMUM_THREADS")),
            "NUMEXPR_NUM_THREADS": str(os.environ.get("NUMEXPR_NUM_THREADS")),
        },
    }

    # OpenCV
    try:
        import cv2

        thread_vars["opencv"] = {
            "threads": cv2.getNumThreads(),
            "opencl": cv2.ocl.useOpenCL(),
        }
    except ImportError:
        thread_vars["opencv"] = "not installed"

    # PyTorch
    try:
        import torch

        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_device": torch.cuda.current_device(),
                "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "gpu_memory_total": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                / (1024**3),  # GB
                "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
            }
        else:
            gpu_info = {
                "gpu_available": False,
                "gpu_device": None,
            }

        thread_vars["pytorch"] = {
            "threads": torch.get_num_threads(),
            **gpu_info,
        }
    except ImportError:
        thread_vars["pytorch"] = "not installed"

    # Pillow
    try:
        from PIL import Image

        thread_vars["pillow"] = {
            "threads": Image.core.get_threads() if hasattr(Image.core, "get_threads") else "unknown",
            "simd": hasattr(Image.core, "simd_support"),
        }
    except ImportError:
        thread_vars["pillow"] = "not installed"

    # Convert all values to strings, replacing None with "Not set"
    return {k: str(v) if v is not None else "Not set" for k, v in thread_vars.items()}


def get_system_info() -> dict[str, str]:
    """Get system information"""
    memory_total = "unknown"
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            memory_total = str(int(pages) * int(page_size))
    except (OSError, ValueError):
        memory_total = "unknown"

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": str(multiprocessing.cpu_count()),
        "memory_total_bytes": memory_total,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def get_library_versions(library: str) -> dict[str, str]:
    """Get versions of relevant libraries"""
    versions = {}

    def get_version(package: str) -> str:
        try:
            return str(metadata.version(package))
        except metadata.PackageNotFoundError:
            return "not installed"

    # PyPI package names (may differ from import: e.g. albumentationsx → import albumentations)
    _pkg_names: dict[str, list[str]] = {
        "albumentationsx": ["albumentationsx", "albumentations"],
        "albumentations_mit": ["albumentations"],
    }
    if library in _pkg_names:
        for pkg in _pkg_names[library]:
            v = get_version(pkg)
            if v != "not installed":
                versions[library] = v
                break
        else:
            versions[library] = "not installed"
    else:
        versions[library] = get_version(library)

    decoder_packages = {
        "opencv": ["opencv-python-headless", "opencv-python"],
        "pyav": ["av"],
        "decord": ["decord"],
        "torchcodec": ["torchcodec"],
        "torchvision": ["torchvision"],
        "dali": ["nvidia-dali-cuda120", "nvidia-dali-cuda110", "nvidia-dali"],
    }
    if library in decoder_packages:
        for pkg in decoder_packages[library]:
            v = get_version(pkg)
            if v != "not installed":
                versions[library] = v
                break

    for extra_librarties in [
        "numpy",
        "pillow",
        "opencv-python-headless",
        "torch",
        "torchvision",
        "opencv-python",
        "av",
        "decord",
        "torchcodec",
    ]:
        versions[extra_librarties] = get_version(extra_librarties)

    return versions


def is_variance_stable(
    throughputs: list[float],
    window: int = 5,
    threshold: float = 0.05,
    min_windows: int = 3,
) -> bool:
    """Check if throughput variance has stabilized"""
    if len(throughputs) < window * min_windows:
        return False

    # Get variances of last few windows
    variances = []
    for i in range(min_windows):
        start_idx = -(i + 1) * window
        end_idx = -i * window if i > 0 else None
        window_data = throughputs[start_idx:end_idx]
        variances.append(np.var(window_data))

    # Check if all variance ratios are below threshold
    for i in range(len(variances) - 1):
        denom = max(variances[i], variances[i + 1])
        if denom == 0:
            continue
        if abs(variances[i] - variances[i + 1]) / denom > threshold:
            return False

    return True
