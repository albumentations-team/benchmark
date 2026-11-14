import os
from pathlib import Path
from unittest.mock import patch

import pytest

from augbench.guest_request import GuestRequest
from augbench.guest_worker import _launch_worker, preflight_implementations, select_pending_cells
from augbench.run_records import CellKey, RunInputs, build_run_record


def _request(*, cell_ids: tuple[str, ...]) -> GuestRequest:
    run = build_run_record(
        family_config={"family": "rgb"},
        inputs=RunInputs(
            git_commit="a" * 40,
            code_archive_sha256="b" * 64,
            dataset_archive_sha256="d" * 64,
            recipe_catalog_sha256="e" * 64,
            environment_lock_sha256={"rgb": "f" * 64},
        ),
        hardware={"machine_type": "g2-standard-16"},
    )
    return GuestRequest(
        run=run,
        gcs_base_uri="gs://bucket/results",
        code_archive_uri="gs://bucket/code.tar.gz",
        dataset_archive_uri="gs://bucket/dataset.tar",
        dataset_archive_sha256="d" * 64,
        environment_cache_uri="gs://bucket/environment.tar.gz",
        environment_lock_path="environments/rgb/lock.txt",
        environment_lock_sha256="f" * 64,
        environment_python_version="3.13.14",
        pending_cell_ids=cell_ids,
    )


def test_guest_selects_only_the_requested_known_cells() -> None:
    request = _request(cell_ids=("0" * 64,))
    wanted = CellKey(
        run_id=request.run.run_id,
        family="rgb",
        implementation="pillow_cpu",
        recipe_id="Resize224+Normalize+ToTensor",
        seed=137,
    )
    other = wanted.model_copy(update={"seed": 138})
    allsimd = wanted.model_copy(update={"implementation": "pillow_allsimd_cpu"})
    request = request.model_copy(update={"pending_cell_ids": (wanted.cell_id, allsimd.cell_id)})
    cells = (wanted, other, allsimd)

    assert select_pending_cells(request=request, cells=cells) == (wanted, allsimd)


def test_guest_rejects_a_cell_not_derived_from_its_run() -> None:
    request = _request(cell_ids=("0" * 64,))

    with pytest.raises(ValueError, match="unknown cells"):
        select_pending_cells(request=request, cells=())


def test_ax_starts_with_single_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
    with patch("augbench.guest_worker.subprocess.run") as run:
        _launch_worker([], "albumentationsx_cpu", "[]", Path("/allsimd"))

    env = run.call_args.kwargs["env"]
    assert env["OPENBLAS_NUM_THREADS"] == env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["OPENCV_FOR_THREADS_NUM"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "16"


@pytest.mark.parametrize("implementation", ["torchvision_cpu", "kornia_gpu", "pillow_allsimd_cpu", "dali_gpu"])
def test_other_implementations_start_without_thread_overrides(
    implementation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        monkeypatch.setenv(name, "1")
    with patch("augbench.guest_worker.subprocess.run") as run:
        _launch_worker([], implementation, "[]", Path("/allsimd"))

    env = run.call_args.kwargs["env"]
    assert "OPENBLAS_NUM_THREADS" not in env
    assert "OMP_NUM_THREADS" not in env
    assert "OPENCV_FOR_THREADS_NUM" not in env


def test_preflight_compiles_each_implementation_and_runs_one_batch_per_recipe() -> None:
    request = _request(cell_ids=("0" * 64,))
    first = CellKey(
        run_id=request.run.run_id,
        family="rgb",
        implementation="pillow_cpu",
        recipe_id="first",
        seed=137,
    )
    second = first.model_copy(update={"recipe_id": "second", "seed": 138})
    third = first.model_copy(update={"implementation": "dali_gpu", "recipe_id": "third"})
    duplicate = first.model_copy(update={"seed": 139})
    calls: list[tuple[str, str]] = []

    class Executor:
        def compile_implementation(self, implementation: str) -> None:
            calls.append(("compile", implementation))

        def preflight(self, cell: CellKey) -> None:
            calls.append(("batch", cell.implementation))

    preflight_implementations(Executor(), (first, second, duplicate, third))  # type: ignore[arg-type]

    assert calls == [
        ("compile", "pillow_cpu"),
        ("batch", "pillow_cpu"),
        ("batch", "pillow_cpu"),
        ("compile", "dali_gpu"),
        ("batch", "dali_gpu"),
    ]
