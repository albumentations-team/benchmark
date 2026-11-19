"""Prepare standalone grayscale JPEGs from the ImageNet validation archive."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

JPEG_QUALITY = 95


def prepare_grayscale_archive(source: Path, output: Path) -> dict[str, str | int]:
    from PIL import Image, __version__

    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    with TemporaryDirectory(dir=output.parent, prefix=".grayscale-") as directory:
        temporary = Path(directory) / "dataset.tar"
        with tarfile.open(source, "r|*") as original, tarfile.open(temporary, "w") as grayscale:
            for member in original:
                path = PurePosixPath(member.name)
                if not member.isfile() or path.parent != PurePosixPath("val") or path.suffix != ".JPEG":
                    continue
                name = path.as_posix()
                if name in names:
                    raise ValueError(f"duplicate ImageNet validation image: {name}")
                names.add(name)
                encoded = original.extractfile(member)
                if encoded is None:
                    raise ValueError(f"unable to read ImageNet validation image: {name}")
                with Image.open(io.BytesIO(encoded.read())) as image:
                    gray = image.convert("L")
                with io.BytesIO() as buffer:
                    gray.save(buffer, format="JPEG", quality=JPEG_QUALITY)
                    metadata = tarfile.TarInfo(name)
                    metadata.size = buffer.tell()
                    buffer.seek(0)
                    grayscale.addfile(metadata, buffer)
        if not names:
            raise ValueError("the source archive has no val/*.JPEG files")
        with temporary.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        os.link(temporary, output)
    return {
        "archive": str(output),
        "file_count": len(names),
        "sha256": checksum,
        "jpeg_quality": JPEG_QUALITY,
        "pillow_version": __version__,
    }
