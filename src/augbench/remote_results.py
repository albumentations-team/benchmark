"""Validate immutable GCS cells before deciding what a VM must run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from augbench.result_store import decode_result

if TYPE_CHECKING:
    from collections.abc import Iterable

    from augbench.run_records import CellKey


class RemoteResultStore(Protocol):
    def list_keys(self, prefix: str) -> tuple[str, ...]: ...

    def read(self, key: str) -> bytes: ...


def completed_cell_ids(
    *,
    remote: RemoteResultStore,
    run_id: str,
    cells: Iterable[CellKey],
) -> frozenset[str]:
    """Return only valid published cells; corruption stops the run explicitly."""
    expected = {cell.cell_id: cell for cell in cells}
    prefix = f"runs/{run_id}/cells"
    completed: set[str] = set()
    for key in remote.list_keys(prefix):
        cell_id = _cell_id_from_key(key, prefix)
        if cell_id is None or cell_id not in expected:
            continue
        result = decode_result(remote.read(key), expected_cell_id=cell_id)
        if result.run_id != run_id or result.cell != expected[cell_id]:
            raise ValueError(f"remote result {key} does not belong to this frozen matrix")
        completed.add(cell_id)
    return frozenset(completed)


def pending_cells(*, cells: Iterable[CellKey], completed: frozenset[str]) -> tuple[CellKey, ...]:
    return tuple(cell for cell in cells if cell.cell_id not in completed)


def _cell_id_from_key(key: str, prefix: str) -> str | None:
    expected_prefix = f"{prefix}/"
    if not key.startswith(expected_prefix) or not key.endswith(".json"):
        return None
    cell_id = key.removeprefix(expected_prefix).removesuffix(".json")
    if "/" in cell_id or len(cell_id) != 64 or any(character not in "0123456789abcdef" for character in cell_id):
        return None
    return cell_id
