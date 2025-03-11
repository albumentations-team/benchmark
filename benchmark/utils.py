import multiprocessing
import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pkg_resources


def read_img_cv2(path: Path) -> np.ndarray:
    """Read image using OpenCV (for Albumentations and imgaug)"""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_img_torch(path: Path) -> Any:  # torch.Tensor
    """Read image using torchvision"""
    import torchvision

    img = torchvision.io.read_image(str(path))
    return img.unsqueeze(0)


def read_img_pillow(path: Path) -> Any:  # PIL.Image.Image
    """Read image using PIL (for augly)"""
    from PIL import Image

    return Image.open(path).convert("RGB")


def read_img_kornia(path: Path) -> Any:  # torch.Tensor
    """Read image using kornia format"""
    return (read_img_torch(path) / 255.0).half()  # Convert to float16


def read_video_cv2(path: Path) -> np.ndarray:
    """Read video using OpenCV (for Albumentations)

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


def read_video_kornia(path: Path) -> Any:  # torch.Tensor
    """Read video using kornia format"""
    video = read_video_torch(path)
    return (video.float() / 255.0).half()  # Convert to float16


def time_transform(transform: Any, images: list[Any]) -> float:
    """Time the execution of a transform on a list of images"""
    import time

    start = time.perf_counter()

    for img in images:
        _ = transform(img)

    return time.perf_counter() - start


@cache
def get_image_loader(library: str) -> Callable[[Path], Any]:
    """Get the appropriate image loader for the library"""
    loaders = {
        "albumentations": read_img_cv2,
        "ultralytics": read_img_cv2,
        "imgaug": read_img_cv2,
        "torchvision": read_img_torch,
        "kornia": read_img_kornia,
        "augly": read_img_pillow,
    }

    if library not in loaders:
        raise ValueError(f"Unsupported library: {library}. Supported libraries are: {list(loaders.keys())}")

    return loaders[library]


@cache
def get_video_loader(library: str) -> Callable[[Path], Any]:
    """Get the appropriate video loader for the library"""
    loaders = {
        "albumentations": read_video_cv2,
        "torchvision": read_video_torch,
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
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": str(multiprocessing.cpu_count()),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def get_library_versions(library: str) -> dict[str, str]:
    """Get versions of relevant libraries"""
    versions = {}

    def get_version(package: str) -> str:
        try:
            return str(pkg_resources.get_distribution(package).version)
        except pkg_resources.DistributionNotFound:
            return "not installed"

    versions[library] = get_version(library)

    for extra_librarties in ["numpy", "pillow", "opencv-python-headless", "torch", "opencv-python"]:
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
        var_ratio = abs(variances[i] - variances[i + 1]) / max(variances[i], variances[i + 1])
        if var_ratio > threshold:
            return False

    return True
