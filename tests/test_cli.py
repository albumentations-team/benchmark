from pathlib import Path

import pytest

from augbench.cli import main
from augbench.frozen_run import FrozenRun
from augbench.gcp_controller import GcpLaunch, LaunchedInstance
from augbench.guest_request import GuestRequest
from augbench.run_config import FamilyRunConfig
from augbench.run_records import RunInputs, build_run_record

ROOT = Path(__file__).parents[1]


def _frozen() -> FrozenRun:
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
    request = GuestRequest(
        run=run,
        gcs_base_uri="gs://bucket/benchmark",
        code_archive_uri="gs://bucket/code.tar.gz",
        dataset_archive_uri="gs://bucket/data.tar",
        dataset_archive_sha256="d" * 64,
        environment_cache_uri="gs://bucket/environment.tar.gz",
        environment_lock_path="environments/rgb/lock.txt",
        environment_lock_sha256="f" * 64,
        environment_python_version="3.13.14",
        pending_cell_ids=("0" * 64,),
    )
    return FrozenRun(run=run, request=request, cells=())


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[FamilyRunConfig]:
    """Replace the launcher and record the config each command resolves."""
    frozen = _frozen()
    state = GcpLaunch(
        status="active",
        run_id=frozen.run.run_id,
        pending_cell_ids=("1" * 64,),
        instances=(LaunchedInstance(name="vm", zone="zone"),),
    )
    configs: list[FamilyRunConfig] = []

    def fake(*, repository_root: Path, config: FamilyRunConfig) -> tuple[FrozenRun, GcpLaunch]:
        assert repository_root == ROOT
        configs.append(config)
        return frozen, state

    monkeypatch.setattr("augbench.cli.launch", fake)
    return configs


def test_launch_rgb_alias_prints_actionable_state(
    launched: list[FamilyRunConfig], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["launch-rgb", "--repository-root", str(ROOT)]) == 0

    output = capsys.readouterr().out
    assert '"status": "active"' in output
    assert '"pending_cells": 1' in output
    assert launched[0].family == "rgb"


def test_launch_takes_any_family_config(launched: list[FamilyRunConfig]) -> None:
    assert main(["launch", "--repository-root", str(ROOT), "--config", "configs/families/rgb.yaml"]) == 0
    assert main(["launch", "--repository-root", str(ROOT), "--config", "configs/families/grayscale.yaml"]) == 0

    assert [config.family for config in launched] == ["rgb", "grayscale"]


def test_launch_grayscale_overrides_the_archive_fields(launched: list[FamilyRunConfig]) -> None:
    uri = "gs://bucket/val-grayscale.tar"
    sha = "c" * 64

    assert (
        main(["launch-grayscale", "--repository-root", str(ROOT), "--archive-uri", uri, "--archive-sha256", sha]) == 0
    )

    dataset = launched[0].dataset
    assert (dataset.archive_uri, dataset.archive_sha256) == (uri, sha)
    assert launched[0].family == "grayscale"
