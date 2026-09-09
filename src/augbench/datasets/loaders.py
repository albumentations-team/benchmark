from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pathlib import Path

type Uint8Array = NDArray[np.uint8]
_HWC_DIMENSIONS = 3


def _validate_uint8_hwc(sample: NDArray[np.generic], *, channels: int, source: Path) -> Uint8Array:
    if sample.dtype != np.uint8:
        raise ValueError(f"{source} must contain uint8 data, got {sample.dtype}")
    if sample.ndim != _HWC_DIMENSIONS or sample.shape[-1] != channels:
        raise ValueError(f"{source} must have HWC shape with {channels} channels, got {sample.shape}")
    return cast("Uint8Array", np.ascontiguousarray(sample, dtype=np.uint8))


class SimpleJpegRgbLoader:
    """Read one JPEG from disk and decode it directly to RGB HWC uint8."""

    def load(self, path: Path) -> Uint8Array:
        import simplejpeg

        sample = simplejpeg.decode_jpeg(path.read_bytes(), colorspace="RGB")
        return _validate_uint8_hwc(sample, channels=3, source=path)
