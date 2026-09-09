from pathlib import Path

import pytest

from augbench.result_store import ImmutableResultStore, ResultConflictError, decode_result, encode_result
from augbench.run_records import CellKey, GpuMemory, OutputObservation, ResultRecord, Throughput


def _result() -> ResultRecord:
    cell = CellKey(run_id="a" * 64, family="rgb", implementation="pillow_cpu", recipe_id="Resize", seed=137)
    return ResultRecord(
        run_id=cell.run_id,
        cell=cell,
        status="ok",
        throughput=Throughput(completed_items=8192, duration_seconds=10.0),
        gpu_memory=GpuMemory(peak_mib=512.0, poll_interval_ms=50, valid_samples=3),
        output=OutputObservation(shape=(256, 3, 224, 224)),
    )


def test_result_store_writes_one_immutable_cell_json(tmp_path: Path) -> None:
    store = ImmutableResultStore(tmp_path)
    result = _result()

    assert store.put(result)
    assert not store.put(result)
    assert store.get(result.cell_id) == result


def test_result_store_rejects_a_conflicting_cell_payload(tmp_path: Path) -> None:
    store = ImmutableResultStore(tmp_path)
    result = _result()
    assert store.put(result)
    conflicting = result.model_copy(update={"runtime": {"library_version": "different"}})

    with pytest.raises(ResultConflictError):
        store.put(conflicting)


def test_decoder_rejects_a_result_stored_under_another_cell_id() -> None:
    result = _result()

    with pytest.raises(ValueError, match="cell ID"):
        decode_result(encode_result(result), expected_cell_id="0" * 64)
