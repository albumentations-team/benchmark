from dataclasses import dataclass

import pytest

from augbench.remote_results import completed_cell_ids, pending_cells
from augbench.result_store import encode_result
from augbench.run_records import CellKey, GpuMemory, OutputObservation, ResultRecord, Throughput


@dataclass
class _Remote:
    objects: dict[str, bytes]

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(sorted(key for key in self.objects if key.startswith(prefix)))

    def read(self, key: str) -> bytes:
        return self.objects[key]


def _cell(seed: int = 137) -> CellKey:
    return CellKey(
        run_id="a" * 64,
        family="rgb",
        implementation="pillow_cpu",
        recipe_id="Resize224+Normalize+ToTensor",
        seed=seed,
    )


def _result(cell: CellKey) -> ResultRecord:
    return ResultRecord(
        run_id=cell.run_id,
        cell=cell,
        status="ok",
        throughput=Throughput(completed_items=8192, duration_seconds=1.0),
        gpu_memory=GpuMemory(peak_mib=512.0, poll_interval_ms=50, valid_samples=3),
        output=OutputObservation(shape=(256, 3, 224, 224)),
    )


def test_valid_remote_results_are_not_scheduled_again() -> None:
    first, second = _cell(), _cell(138)
    remote = _Remote({f"runs/{first.run_id}/cells/{first.cell_id}.json": encode_result(_result(first))})

    completed = completed_cell_ids(remote=remote, run_id=first.run_id, cells=(first, second))

    assert completed == {first.cell_id}
    assert pending_cells(cells=(first, second), completed=completed) == (second,)


def test_corrupt_expected_remote_result_stops_resume() -> None:
    cell = _cell()
    remote = _Remote({f"runs/{cell.run_id}/cells/{cell.cell_id}.json": b"not-json"})

    with pytest.raises(ValueError, match="Expecting value"):
        completed_cell_ids(remote=remote, run_id=cell.run_id, cells=(cell,))
