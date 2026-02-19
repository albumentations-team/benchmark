"""Tests for benchmark/cli.py argument parsing and helpers."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from benchmark.cli import _extract_library, build_parser


class TestBuildParser:
    def test_returns_argument_parser(self) -> None:
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_run_subcommand_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.command == "run"

    def test_compare_subcommand_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["compare", "--baseline", "/baseline", "--current", "/current"])
        assert args.command == "compare"

    def test_run_requires_data_dir(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--output", "/out"])

    def test_run_requires_output(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--data-dir", "/data"])

    def test_compare_requires_baseline(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compare", "--current", "/current"])

    def test_compare_requires_current(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compare", "--baseline", "/baseline"])

    def test_media_defaults_to_image(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.media == "image"

    def test_media_accepts_video(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out", "--media", "video"])
        assert args.media == "video"

    def test_media_rejects_invalid(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--data-dir", "/data", "--output", "/out", "--media", "gif"])

    def test_threshold_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["compare", "--baseline", "/baseline", "--current", "/current"])
        assert args.threshold == pytest.approx(0.05)

    def test_threshold_custom(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["compare", "--baseline", "/baseline", "--current", "/current", "--threshold", "0.1"])
        assert args.threshold == pytest.approx(0.1)

    def test_fail_on_regression_defaults_false(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["compare", "--baseline", "/baseline", "--current", "/current"])
        assert args.fail_on_regression is False

    def test_fail_on_regression_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["compare", "--baseline", "/baseline", "--current", "/current", "--fail-on-regression"],
        )
        assert args.fail_on_regression is True

    def test_num_runs_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.num_runs == 5

    def test_libraries_default_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.libraries is None

    def test_libraries_multiple(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run", "--data-dir", "/data", "--output", "/out", "--libraries", "kornia", "torchvision"],
        )
        assert args.libraries == ["kornia", "torchvision"]

    def test_transforms_default_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.transforms is None

    def test_verbose_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--verbose", "run", "--data-dir", "/data", "--output", "/out"])
        assert args.verbose is True

    def test_warmup_window_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.warmup_window == 5

    def test_warmup_threshold_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.warmup_threshold == pytest.approx(0.05)


class TestExtractLibrary:
    def test_extracts_double_quoted_library(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.py"
        spec.write_text('LIBRARY = "albumentationsx"\n')
        assert _extract_library(spec) == "albumentationsx"

    def test_extracts_single_quoted_library(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.py"
        spec.write_text("LIBRARY = 'kornia'\n")
        assert _extract_library(spec) == "kornia"

    def test_raises_when_no_library_line(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.py"
        spec.write_text("TRANSFORMS = []\n")
        with pytest.raises(ValueError, match="LIBRARY"):
            _extract_library(spec)

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ('LIBRARY = "torchvision"\n', "torchvision"),
            ("LIBRARY = 'imgaug'\n", "imgaug"),
            ('LIBRARY = "augly"\n', "augly"),
        ],
    )
    def test_various_library_names(self, tmp_path: Path, content: str, expected: str) -> None:
        spec = tmp_path / "spec.py"
        spec.write_text(content)
        assert _extract_library(spec) == expected

    def test_ignores_comment_lines(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.py"
        spec.write_text("# LIBRARY = 'notthis'\nLIBRARY = \"correct\"\n")
        # _extract_library checks stripped lines starting with "LIBRARY"
        # comments start with # so they won't match
        assert _extract_library(spec) == "correct"
