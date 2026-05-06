from __future__ import annotations

from typing import Any, Literal

DeviceOption = Literal["none", "cuda", "mps", "auto"]


class DeviceUnavailableError(RuntimeError):
    """Requested benchmark device is not available in the active environment."""


def resolve_device(device: str) -> str | None:
    if device == "none":
        return None
    try:
        import torch
    except ImportError as e:
        if device == "auto":
            return None
        raise DeviceUnavailableError(f"Requested device {device!r}, but torch is not installed") from e

    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise DeviceUnavailableError("Requested --device cuda, but CUDA is not available")
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        raise DeviceUnavailableError("Requested --device mps, but MPS is not available")
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return None
    raise ValueError(f"Unknown device option {device!r}")


def ensure_supported_device(library: str, media: str, device: str, *, mode: str | None = None) -> None:
    if device == "none":
        return
    if media == "video":
        return
    if media == "image" and mode == "pipeline" and library in {"torchvision", "kornia", "dali"}:
        return
    if media == "image" and library in {"torchvision", "kornia"}:
        return
    raise ValueError(f"{library} {media} benchmarks do not support --device {device}")


def is_tensor_image_gpu_path(*, library: str, media: str, device: str) -> bool:
    return media == "image" and library in {"torchvision", "kornia"} and device != "none"


def move_to_device(value: Any, device: str | None) -> Any:
    if device is None:
        return value
    to = getattr(value, "to", None)
    if callable(to):
        return to(device)
    return value


def move_transform_to_device(transform: Any, device: str | None) -> Any:
    return move_to_device(transform, device)


def synchronize_device(device: str | None = None) -> None:
    try:
        import torch
    except ImportError:
        return

    if (device in {None, "cuda"}) and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device in {None, "mps"} and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def reset_peak_memory_stats(device: str | None) -> None:
    if device != "cuda":
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def cuda_memory_stats(device: str | None) -> dict[str, int | None]:
    stats: dict[str, int | None] = {
        "gpu_memory_allocated_before_bytes": None,
        "gpu_memory_allocated_after_bytes": None,
        "gpu_peak_memory_allocated_bytes": None,
        "gpu_peak_memory_reserved_bytes": None,
    }
    if device != "cuda":
        return stats
    try:
        import torch
    except ImportError:
        return stats
    if not torch.cuda.is_available():
        return stats
    torch.cuda.synchronize()
    stats["gpu_memory_allocated_after_bytes"] = int(torch.cuda.memory_allocated())
    stats["gpu_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    stats["gpu_peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    return stats


def cuda_memory_allocated(device: str | None) -> int | None:
    if device != "cuda":
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.memory_allocated())
