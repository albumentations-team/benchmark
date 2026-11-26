from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augbench.datasets.loaders import SimpleJpegLoader

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SourceRef:
    path: Path
    channels: int = 3


def load_albumentationsx_rgb(source: SourceRef) -> Any:
    return SimpleJpegLoader().load(source.path, channels=source.channels)


def load_pillow_rgb(source: SourceRef) -> Any:
    from PIL import Image

    with Image.open(source.path) as image:
        if source.channels == 1:
            if image.mode != "L":
                raise ValueError(f"{source.path} must be a single-channel JPEG")
            return image.copy()
        return image.convert("RGB")


def load_torchvision_rgb(source: SourceRef) -> Any:
    from torchvision.io import ImageReadMode, decode_image

    return decode_image(str(source.path), mode=ImageReadMode.RGB, apply_exif_orientation=False)


def load_jpeg_bytes(source: SourceRef) -> bytes:
    return source.path.read_bytes()


def load_kornia_rgb(source: SourceRef) -> Any:
    from kornia.io import ImageLoadType, load_image

    image_type = ImageLoadType.GRAY32 if source.channels == 1 else ImageLoadType.RGB32
    return load_image(source.path, image_type)
