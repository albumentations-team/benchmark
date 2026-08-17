from pathlib import Path

import pytest

from augbench.cli import main
from augbench.frozen_rgb_run import FrozenRgbRun
from augbench.gcp_controller import GcpLaunch
from augbench.guest_request import GuestRequest
from augbench.run_records import build_run_record


def test_launch_rgb_command_prints_actionable_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    run = build_run_record(
        family_config={"family": "rgb"},
        git_commit="a" * 40,
        code_archive_sha256="b" * 64,
        dataset_archive_sha256="d" * 64,
        recipe_catalog_sha256="e" * 64,
        environment_lock_sha256={"rgb": "f" * 64},
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
    frozen = FrozenRgbRun(run=run, request=request, cells=())
    launch = GcpLaunch(
        status="active",
        run_id=run.run_id,
        pending_cell_ids=("1" * 64,),
        instance_name="vm",
        zone="zone",
    )
    monkeypatch.setattr("augbench.cli.launch_rgb", lambda **_kwargs: (frozen, launch))

    assert main(["launch-rgb", "--repository-root", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert '"status": "active"' in output
    assert '"pending_cells": 1' in output
