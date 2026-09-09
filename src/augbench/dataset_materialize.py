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


@dataclass(frozen=True)
class MaterializedDataset:
    root: Path
    files: tuple[Path, ...]
    cache_hit: bool


def materialize_rgb_dataset(*, archive: Path, dataset: DatasetConfig, cache_root: Path) -> MaterializedDataset:
    """Extract the deterministic configured selection once and reuse a complete cache."""
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
        raise ValueError(f"source archive has only {len(selected)} matching RGB JPEGs")
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
                shutil.copyfileobj(stream, target)


def _is_complete(files: tuple[Path, ...]) -> bool:
    return all(path.is_file() for path in files)


def _selection_digest(dataset: DatasetConfig) -> str:
    encoded = json.dumps(dataset.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts
