from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def device_suffix(device: str) -> str:
    return f"_dev-{device}" if device != "none" else ""


def micro_output_file(output_dir: Path, library: str, *, device: str) -> Path:
    return output_dir / f"{library}_micro{device_suffix(device)}_results.json"


def pipeline_output_file(
    output_dir: Path,
    library: str,
    *,
    pipeline_scope: str,
    num_items: int | None,
    num_runs: int,
    workers: int,
    batch_size: int,
    device: str,
) -> Path:
    items = f"n{num_items}" if num_items is not None else "nall"
    stem = f"{library}_{pipeline_scope}_{items}_r{num_runs}_w{workers}_b{batch_size}{device_suffix(device)}"
    return output_dir / f"{stem}_results.json"


def manual_micro_output_file(output_dir: Path, library: str, *, media: str, device: str) -> Path:
    media_suffix = "_video" if media == "video" else ""
    return output_dir / f"{library}{media_suffix}{device_suffix(device)}_results.json"
