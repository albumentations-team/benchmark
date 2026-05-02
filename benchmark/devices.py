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


def ensure_supported_device(library: str, media: str, device: str) -> None:
    if device == "none":
        return
    if media == "video":
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
