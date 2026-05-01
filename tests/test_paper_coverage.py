from __future__ import annotations

from typing import TYPE_CHECKING

from tools.check_paper_coverage import CORE_REQUIREMENTS, _summary_files, missing_artifacts

if TYPE_CHECKING:
    from pathlib import Path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def test_missing_artifacts_reports_empty_tree(tmp_path: Path) -> None:
    missing = missing_artifacts([tmp_path])

    assert "image-rgb/micro: missing result directory" in missing
    assert "video-16f/pipeline: missing result directory" in missing


def test_complete_core_artifacts_pass(tmp_path: Path) -> None:
    for requirement in CORE_REQUIREMENTS:
        for library in requirement.libraries:
            for filename in _summary_files(requirement, library):
                _touch(tmp_path / requirement.relative_dir / filename)
            if requirement.needs_pyperf:
                _touch(tmp_path / requirement.relative_dir / f"{library}_{requirement.mode}_results.pyperf.json")

    assert missing_artifacts([tmp_path]) == []


def test_nested_run_directories_are_scanned(tmp_path: Path) -> None:
    run_dir = tmp_path / "gcp_runs" / "c4-standard-16-rgb-micro"
    for requirement in CORE_REQUIREMENTS:
        for library in requirement.libraries:
            for filename in _summary_files(requirement, library):
                _touch(run_dir / requirement.relative_dir / filename)
            if requirement.needs_pyperf:
                _touch(run_dir / requirement.relative_dir / f"{library}_{requirement.mode}_results.pyperf.json")

    assert missing_artifacts([tmp_path]) == []


def test_optional_dali_is_not_required_by_default(tmp_path: Path) -> None:
    for requirement in CORE_REQUIREMENTS:
        for library in requirement.libraries:
            for filename in _summary_files(requirement, library):
                _touch(tmp_path / requirement.relative_dir / filename)
            if requirement.needs_pyperf:
                _touch(tmp_path / requirement.relative_dir / f"{library}_{requirement.mode}_results.pyperf.json")

    assert missing_artifacts([tmp_path], require_optional_libraries=True) == [
        "video-16f/pipeline: missing dali_decode_dataloader_augment_n*_r*_w*_b*_results.json",
    ]
