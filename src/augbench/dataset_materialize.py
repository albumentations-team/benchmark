"""Stage the configured RGB JPEG selection from one verified source archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augbench.run_config import DatasetConfig

_JPEG_MARKER_PREFIX = 0xFF
_JPEG_APP1 = 0xE1
_JPEG_SCAN_OR_END = (0xDA, 0xD9)
_JPEG_LENGTH_BYTES = 2
_JPEG_CACHE_FORMAT = "jpeg-exif-free-v1"


@dataclass(frozen=True)
class MaterializedDataset:
    root: Path
    files: tuple[Path, ...]
    cache_hit: bool


def materialize_rgb_dataset(*, archive: Path, dataset: DatasetConfig, cache_root: Path) -> MaterializedDataset:
    """Stage the configured selection without EXIF and reuse a complete cache."""
    selected = _selected_member_names(archive, dataset)
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / _selection_digest(dataset)
    files = tuple(destination / relative_path for relative_path in selected)
    if _is_complete(files):
        return MaterializedDataset(root=destination, files=files, cache_hit=True)
    if destination.exists():
        shutil.rmtree(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".rgb-", dir=cache_root))
    try:
        _extract_selected(archive, selected, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return MaterializedDataset(root=destination, files=files, cache_hit=False)


def _selected_member_names(archive: Path, dataset: DatasetConfig) -> tuple[str, ...]:
    with tarfile.open(archive, "r:*") as source:
        selected = tuple(
            sorted(
                member.name
                for member in source.getmembers()
                if member.isfile()
                and member.name.startswith(dataset.archive_member_prefix)
                and member.name.endswith(dataset.archive_member_suffix)
                and _is_safe_relative_path(member.name)
            )[: dataset.item_count]
        )
    if len(selected) != dataset.item_count:
        raise ValueError(f"source archive has only {len(selected)} matching JPEGs")
    return selected


def _extract_selected(archive: Path, selected: tuple[str, ...], destination: Path) -> None:
    with tarfile.open(archive, "r:*") as source:
        members = {member.name: member for member in source.getmembers() if member.name in selected}
        if len(members) != len(selected):
            raise ValueError("source archive changed while RGB data was being staged")
        for relative_path in selected:
            member = members[relative_path]
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"could not extract {relative_path}")
            output = destination / relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            with stream, output.open("wb") as target:
                target.write(_strip_jpeg_exif(stream.read()))


def _strip_jpeg_exif(encoded: bytes) -> bytes:
    """Remove EXIF APP1 headers while preserving the encoded scan and other metadata."""
    if not encoded.startswith(b"\xff\xd8"):
        return encoded

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


def _is_complete(files: tuple[Path, ...]) -> bool:
    return all(path.is_file() for path in files)


def _selection_digest(dataset: DatasetConfig) -> str:
    encoded = json.dumps(dataset.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(_JPEG_CACHE_FORMAT.encode("ascii") + b"\0" + encoded).hexdigest()


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts
