from pathlib import Path

import pytest

from augbench.guest_request import GuestRequest
from augbench.guest_worker import _run_pool
from augbench.job_pool import claim_jobs
from augbench.run_records import CellKey, RunInputs, build_run_record


class _Remote:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        if key in self.objects:
            return False
        self.objects[key] = payload
        return True


def _request() -> GuestRequest:
    run = build_run_record(
        family_config={"family": "rgb"},
        inputs=RunInputs(
            git_commit="a" * 40,
            code_archive_sha256="b" * 64,
            dataset_archive_sha256="c" * 64,
            recipe_catalog_sha256="d" * 64,
            environment_lock_sha256={"rgb": "e" * 64},
        ),
        hardware={"machine_type": "g2-standard-16"},
    )
    return GuestRequest(
        run=run,
        gcs_base_uri="gs://bucket",
        code_archive_uri="gs://bucket/code",
        dataset_archive_uri="gs://bucket/data",
        dataset_archive_sha256="c" * 64,
        environment_cache_uri="gs://bucket/env",
        environment_lock_path="lock.txt",
        environment_lock_sha256="e" * 64,
        environment_python_version="3.13.14",
        pending_cell_ids=("0" * 64,),
        pool_id="1" * 32,
    )


def _cells(request: GuestRequest) -> tuple[CellKey, ...]:
    first = CellKey(run_id=request.run.run_id, family="rgb", implementation="pillow_cpu", recipe_id="crop", seed=137)
    return (first, first.model_copy(update={"recipe_id": "resize"}), first.model_copy(update={"seed": 138}))


def test_workers_claim_disjoint_seed_jobs_and_keep_recipes_together() -> None:
    request = _request()
    cells = _cells(request)
    remote = _Remote()
    first_worker = claim_jobs(request=request, cells=cells, remote=remote)
    assert next(first_worker) == cells[:2]
    assert list(claim_jobs(request=request, cells=cells, remote=remote)) == [(cells[2],)]
    assert list(first_worker) == []
    assert len(remote.objects) == 2


def test_resume_uses_new_claims_for_only_missing_cells() -> None:
    request = _request()
    cells = _cells(request)
    remote = _Remote()
    assert len(list(claim_jobs(request=request, cells=cells, remote=remote))) == 2
    resumed = request.model_copy(update={"pool_id": "2" * 32})
    assert list(claim_jobs(request=resumed, cells=cells[1:], remote=remote)) == [(cells[1],), (cells[2],)]


def test_failed_child_does_not_claim_the_next_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    remote = _Remote()
    monkeypatch.setattr("augbench.guest_worker.GcloudObjectStore", lambda **_kwargs: remote)

    def fail_child(argv: list[str], implementation: str, _dataset: str, _site: Path) -> None:
        child = GuestRequest.model_validate_json(Path(argv[1]).read_text())
        assert child.pool_id is None
        assert implementation == "pillow_cpu"
        assert child.pending_cell_ids == tuple(cell.cell_id for cell in _cells(request)[:2])
        raise RuntimeError("worker failed")

    monkeypatch.setattr("augbench.guest_worker._launch_worker", fail_child)
    with pytest.raises(RuntimeError, match="worker failed"):
        _run_pool(request, _cells(request), ["--request", str(tmp_path / "request.json")], "[]", tmp_path)
    assert len(remote.objects) == 1


def test_vm_takes_another_job_after_its_child_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    remote = _Remote()
    jobs: list[tuple[str, ...]] = []
    monkeypatch.setattr("augbench.guest_worker.GcloudObjectStore", lambda **_kwargs: remote)

    def finish_child(argv: list[str], _implementation: str, _dataset: str, _site: Path) -> None:
        child = GuestRequest.model_validate_json(Path(argv[1]).read_text())
        jobs.append(child.pending_cell_ids)

    monkeypatch.setattr("augbench.guest_worker._launch_worker", finish_child)
    _run_pool(request, _cells(request), ["--request", str(tmp_path / "request.json")], "[]", tmp_path)
    assert jobs == [tuple(cell.cell_id for cell in _cells(request)[:2]), (_cells(request)[2].cell_id,)]
