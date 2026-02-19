"""Tests for tools/compare.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tools.compare import compare_regression, load_result_file, load_results_dir


class TestLoadResultFile:
    def test_image_result_file(self, tmp_path: Path) -> None:
        data = {"metadata": {"library_versions": {"albumentationsx": "1.0"}}, "results": {"HorizontalFlip": {}}}
        path = tmp_path / "albumentationsx_results.json"
        path.write_text(json.dumps(data))

        library, media, _metadata, results = load_result_file(path)
        assert library == "albumentationsx"
        assert media == "image"
        assert "HorizontalFlip" in results

    def test_video_result_file(self, tmp_path: Path) -> None:
        data = {"metadata": {}, "results": {"Resize": {}}}
        path = tmp_path / "kornia_video_results.json"
        path.write_text(json.dumps(data))

        library, media, _metadata, _results = load_result_file(path)
        assert library == "kornia"
        assert media == "video"

    def test_returns_metadata_dict(self, minimal_result_json: Path) -> None:
        _, _, metadata, _results = load_result_file(minimal_result_json)
        assert isinstance(metadata, dict)
        assert "library_versions" in metadata

    def test_returns_results_dict(self, minimal_result_json: Path) -> None:
        _, _, _metadata, results = load_result_file(minimal_result_json)
        assert isinstance(results, dict)
        assert "HorizontalFlip" in results

    def test_missing_metadata_key_returns_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "augly_results.json"
        path.write_text(json.dumps({"results": {}}))
        _, _, metadata, _results = load_result_file(path)
        assert metadata == {}

    def test_missing_results_key_returns_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "augly_results.json"
        path.write_text(json.dumps({"metadata": {}}))
        _, _, _metadata, results = load_result_file(path)
        assert results == {}


class TestLoadResultsDir:
    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        result = load_results_dir(tmp_path)
        assert result == {}

    def test_loads_two_result_files(self, result_dir_fixture: Path) -> None:
        result = load_results_dir(result_dir_fixture)
        assert len(result) == 2

    def test_image_key_format(self, result_dir_fixture: Path) -> None:
        result = load_results_dir(result_dir_fixture)
        assert "albumentationsx" in result

    def test_video_key_format(self, result_dir_fixture: Path) -> None:
        result = load_results_dir(result_dir_fixture)
        assert "kornia_video" in result

    def test_malformed_json_is_skipped(self, tmp_path: Path, capsys) -> None:
        bad = tmp_path / "bad_results.json"
        bad.write_text("this is not json {{{")
        result = load_results_dir(tmp_path)
        assert result == {}
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_result_entry_has_expected_keys(self, result_dir_fixture: Path) -> None:
        result = load_results_dir(result_dir_fixture)
        for entry in result.values():
            assert "library" in entry
            assert "media" in entry
            assert "metadata" in entry
            assert "results" in entry


class TestCompareRegression:
    def test_regression_detected(self, baseline_and_current_dirs, capsys) -> None:
        baseline_dir, current_dir = baseline_and_current_dirs
        compare_regression(
            baseline_dir=baseline_dir,
            current_dir=current_dir,
            libraries_filter=None,
            transforms_filter=None,
            threshold=0.05,
            fail_on_regression=False,
        )
        captured = capsys.readouterr()
        assert "REGRESSION" in captured.out

    def test_improvement_detected(self, baseline_and_current_dirs, capsys) -> None:
        baseline_dir, current_dir = baseline_and_current_dirs
        compare_regression(
            baseline_dir=baseline_dir,
            current_dir=current_dir,
            libraries_filter=None,
            transforms_filter=None,
            threshold=0.05,
            fail_on_regression=False,
        )
        captured = capsys.readouterr()
        assert "faster" in captured.out

    def test_fail_on_regression_exits_1(self, baseline_and_current_dirs) -> None:
        baseline_dir, current_dir = baseline_and_current_dirs
        with pytest.raises(SystemExit) as exc_info:
            compare_regression(
                baseline_dir=baseline_dir,
                current_dir=current_dir,
                libraries_filter=None,
                transforms_filter=None,
                threshold=0.05,
                fail_on_regression=True,
            )
        assert exc_info.value.code == 1

    def test_no_fail_without_flag(self, baseline_and_current_dirs) -> None:
        baseline_dir, current_dir = baseline_and_current_dirs
        # Should not raise even with regressions present
        compare_regression(
            baseline_dir=baseline_dir,
            current_dir=current_dir,
            libraries_filter=None,
            transforms_filter=None,
            threshold=0.05,
            fail_on_regression=False,
        )

    def test_no_common_libraries_exits_cleanly(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        current.mkdir()
        # Write different library names so no common keys
        (baseline / "albumentationsx_results.json").write_text(json.dumps({"metadata": {}, "results": {}}))
        (current / "kornia_results.json").write_text(json.dumps({"metadata": {}, "results": {}}))
        # Should exit with 0, not raise
        with pytest.raises(SystemExit) as exc_info:
            compare_regression(
                baseline_dir=baseline,
                current_dir=current,
                libraries_filter=None,
                transforms_filter=None,
                threshold=0.05,
                fail_on_regression=False,
            )
        assert exc_info.value.code == 0

    def test_below_threshold_is_same(self, tmp_path: Path, capsys) -> None:
        """A 1% difference with 5% threshold should report 'same'."""
        baseline_dir = tmp_path / "b"
        current_dir = tmp_path / "c"
        baseline_dir.mkdir()
        current_dir.mkdir()

        def _entry(tp: float) -> dict:
            return {
                "supported": True,
                "early_stopped": False,
                "median_throughput": tp,
                "std_throughput": 1.0,
            }

        base_data = {"metadata": {}, "results": {"HorizontalFlip": _entry(1000.0)}}
        curr_data = {"metadata": {}, "results": {"HorizontalFlip": _entry(990.0)}}  # -1%

        (baseline_dir / "albumentationsx_results.json").write_text(json.dumps(base_data))
        (current_dir / "albumentationsx_results.json").write_text(json.dumps(curr_data))

        compare_regression(
            baseline_dir=baseline_dir,
            current_dir=current_dir,
            libraries_filter=None,
            transforms_filter=None,
            threshold=0.05,
            fail_on_regression=False,
        )
        captured = capsys.readouterr()
        assert "same" in captured.out

    def test_libraries_filter_applied(self, baseline_and_current_dirs) -> None:
        baseline_dir, current_dir = baseline_and_current_dirs
        # Filtering to a nonexistent library means no comparable transforms → sys.exit(0)
        with pytest.raises(SystemExit) as exc_info:
            compare_regression(
                baseline_dir=baseline_dir,
                current_dir=current_dir,
                libraries_filter=["nonexistent"],
                transforms_filter=None,
                threshold=0.05,
                fail_on_regression=False,
            )
        assert exc_info.value.code == 0

    def test_transforms_filter_applied(self, baseline_and_current_dirs, capsys) -> None:
        baseline_dir, current_dir = baseline_and_current_dirs
        compare_regression(
            baseline_dir=baseline_dir,
            current_dir=current_dir,
            libraries_filter=None,
            transforms_filter=["GaussianBlur"],  # only the improving transform
            threshold=0.05,
            fail_on_regression=False,
        )
        captured = capsys.readouterr()
        assert "REGRESSION" not in captured.out
