from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augbench.datasets.loaders import SimpleJpegRgbLoader

if TYPE_CHECKING:
    from pathlib import Path

_JPEG_MARKER_PREFIX = 0xFF
_JPEG_APP1 = 0xE1
_JPEG_SCAN_OR_END = (0xDA, 0xD9)
_JPEG_LENGTH_BYTES = 2


@dataclass(frozen=True)
class SourceRef:
    path: Path


def load_albumentationsx_rgb(source: SourceRef) -> Any:
    return SimpleJpegRgbLoader().load(source.path)


def load_pillow_rgb(source: SourceRef) -> Any:
    from PIL import Image

    with Image.open(source.path) as image:
        return image.convert("RGB")


def load_torchvision_rgb(source: SourceRef) -> Any:
    from torchvision.io import ImageReadMode, decode_image

    return decode_image(str(source.path), mode=ImageReadMode.RGB)


def load_jpeg_bytes(source: SourceRef) -> bytes:
    return _without_jpeg_exif(source.path.read_bytes())


def _without_jpeg_exif(encoded: bytes) -> bytes:
    if not encoded.startswith(b"\xff\xd8"):
        return encoded

    # TorchCodec always applies EXIF orientation. Keep the stored pixel orientation.
    view = memoryview(encoded)
    parts = []
    start = 0
    offset = 2
    while offset + 4 <= len(encoded) and encoded[offset] == _JPEG_MARKER_PREFIX:
        marker_start = offset
        while offset < len(encoded) and encoded[offset] == _JPEG_MARKER_PREFIX:
            offset += 1
        if offset + 3 > len(encoded) or encoded[offset] in _JPEG_SCAN_OR_END:
            break
        segment_size = int.from_bytes(view[offset + 1 : offset + 3], "big")
        segment_end = offset + 1 + segment_size
        if segment_size < _JPEG_LENGTH_BYTES or segment_end > len(encoded):
            break
        if encoded[offset] == _JPEG_APP1 and encoded.startswith(b"Exif\x00\x00", offset + 3, segment_end):
            parts.append(view[start:marker_start])
            start = segment_end
        offset = segment_end
    return b"".join([*parts, view[start:]]) if parts else encoded


def load_kornia_rgb(source: SourceRef) -> Any:
    from kornia.io import ImageLoadType, load_image

    return load_image(source.path, ImageLoadType.RGB32)
