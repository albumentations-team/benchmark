"""Dataset access shared by all implementations in one family run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from augbench.run_config import source_order

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(frozen=True)
class PrewarmResult:
    files: int
    bytes_read: int


def prewarm_files(paths: Iterable[Path]) -> PrewarmResult:
    """Read the selected files once before timing any cell.

    This deliberately uses the configured selection order. The timed cells use a seeded
    permutation instead, so no implementation gains an accidental cold-cache
    advantage simply because it happens to run first.
    """
    bytes_read = 0
    files = 0
    for path in paths:
        files += 1
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                bytes_read += len(chunk)
    return PrewarmResult(files=files, bytes_read=bytes_read)


def reordered_items[Item](items: tuple[Item, ...], *, required_items: int, seed: int) -> tuple[Item, ...]:
    indices = source_order(item_count=len(items), required_items=required_items, seed=seed)
    return tuple(items[index] for index in indices)
