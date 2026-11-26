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


class SimpleJpegLoader:
    """Read a JPEG directly into an HWC uint8 array."""

    def load(self, path: Path, *, channels: int = 3) -> Uint8Array:
        import simplejpeg

        colorspace = "GRAY" if channels == 1 else "RGB"
        sample = simplejpeg.decode_jpeg(path.read_bytes(), colorspace=colorspace)
        return _validate_uint8_hwc(sample, channels=channels, source=path)
