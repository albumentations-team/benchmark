import pytest

from augbench.run_records import (
    CellKey,
    GpuMemory,
    OutputObservation,
    ResultRecord,
    Throughput,
    build_run_record,
)


def test_run_identity_and_cell_identity_are_deterministic() -> None:
    run = build_run_record(
        family_config={"family": "rgb", "batch_size": 256},
        git_commit="a" * 40,
        code_archive_sha256="b" * 64,
        dataset_archive_sha256="d" * 64,
        recipe_catalog_sha256="e" * 64,
        environment_lock_sha256={"torch-stack": "f" * 64},
        hardware={"machine_type": "g2-standard-16", "gpu": "NVIDIA L4"},
    )

    same_run = build_run_record(
        family_config={"batch_size": 256, "family": "rgb"},
        git_commit="a" * 40,
        code_archive_sha256="b" * 64,
        dataset_archive_sha256="d" * 64,
        recipe_catalog_sha256="e" * 64,
        environment_lock_sha256={"torch-stack": "f" * 64},
        hardware={"gpu": "NVIDIA L4", "machine_type": "g2-standard-16"},
    )
    cell = CellKey(run_id=run.run_id, family="rgb", implementation="pillow_cpu", recipe_id="Resize", seed=137)

    assert run.run_id == same_run.run_id
    assert cell.cell_id == CellKey.model_validate(cell.model_dump()).cell_id


def test_successful_result_requires_throughput_memory_and_model_ready_output() -> None:
    cell = CellKey(run_id="a" * 64, family="rgb", implementation="pillow_cpu", recipe_id="Resize", seed=137)
    result = ResultRecord(
        run_id=cell.run_id,
        cell=cell,
        status="ok",
        throughput=Throughput(completed_items=8192, duration_seconds=10.0),
        gpu_memory=GpuMemory(peak_mib=512.0, poll_interval_ms=50, valid_samples=9),
        output=OutputObservation(shape=(256, 3, 224, 224)),
        runtime={"library_version": "12.0", "cuda_device": "NVIDIA L4"},
    )

    assert result.throughput is not None
    assert result.throughput.value == 819.2
    assert result.cell_id == cell.cell_id


def test_production_result_cannot_be_marked_unsupported() -> None:
    cell = CellKey(run_id="a" * 64, family="rgb", implementation="pillow_cpu", recipe_id="Resize", seed=137)

    with pytest.raises(ValueError, match="status"):
        ResultRecord(
            run_id=cell.run_id,
            cell=cell,
            status="unsupported",  # type: ignore[arg-type]
            throughput=Throughput(completed_items=8192, duration_seconds=10.0),
            gpu_memory=GpuMemory(peak_mib=512.0, poll_interval_ms=50, valid_samples=9),
            output=OutputObservation(shape=(256, 3, 224, 224)),
        )
