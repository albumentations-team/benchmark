from pathlib import Path

from augbench.reporting import build_report
from augbench.result_store import ImmutableResultStore
from augbench.run_records import CellKey, GpuMemory, OutputObservation, ResultRecord, Throughput


def _result(*, run_id: str, seed: int, duration: float) -> ResultRecord:
    cell = CellKey(run_id=run_id, family="rgb", implementation="pillow_cpu", recipe_id="Resize", seed=seed)
    return ResultRecord(
        run_id=run_id,
        cell=cell,
        status="ok",
        throughput=Throughput(completed_items=8192, duration_seconds=duration),
        gpu_memory=GpuMemory(peak_mib=512 + seed, poll_interval_ms=50, valid_samples=3),
        output=OutputObservation(shape=(256, 3, 224, 224)),
    )


def test_report_aggregates_median_speed_and_memory_only_when_matrix_is_complete(tmp_path: Path) -> None:
    run_id = "a" * 64
    results = (_result(run_id=run_id, seed=137, duration=8.0), _result(run_id=run_id, seed=138, duration=10.0))
    store = ImmutableResultStore(tmp_path)
    for result in results:
        store.put(result)

    report = build_report(results=store, expected_cells=tuple(result.cell for result in results))

    assert report.complete
    assert report.rows[0].median_images_per_second == 921.6
    assert report.rows[0].median_peak_gpu_mib == 649.5


def test_report_refuses_claims_when_any_expected_cell_is_missing(tmp_path: Path) -> None:
    run_id = "a" * 64
    present = _result(run_id=run_id, seed=137, duration=8.0)
    missing = _result(run_id=run_id, seed=138, duration=10.0)
    store = ImmutableResultStore(tmp_path)
    store.put(present)

    report = build_report(results=store, expected_cells=(present.cell, missing.cell))

    assert not report.complete
    assert report.rows == ()
    assert report.missing_cell_ids == (missing.cell_id,)
