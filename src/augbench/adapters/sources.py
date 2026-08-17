from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augbench.datasets.loaders import SimpleJpegRgbLoader

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SourceRef:
    path: Path


def load_albumentationsx_rgb(source: SourceRef) -> Any:
    return SimpleJpegRgbLoader().load(source.path)


def load_pillow_rgb(source: SourceRef) -> Any:
    from PIL import Image

    with Image.open(source.path) as image:
        return image.convert("RGB").copy()


def load_torchvision_rgb(source: SourceRef) -> Any:
    from torchvision.io import ImageReadMode, decode_image

    return decode_image(str(source.path), mode=ImageReadMode.RGB)


def load_kornia_rgb(source: SourceRef) -> Any:
    return load_torchvision_rgb(source).float().div_(255.0)
