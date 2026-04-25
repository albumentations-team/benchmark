"""Tests for benchmark/cli.py argument parsing and helpers."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from benchmark.cli import _extract_library, build_gcp_benchmark_cli_argv, build_parser


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

    def test_doctor_subcommand_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["doctor", "--json"])
        assert args.command == "doctor"
        assert args.json is True

    def test_validate_results_subcommand_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["validate-results", "/results"])
        assert args.command == "validate-results"
        assert args.path == "/results"

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

    def test_timer_defaults_to_pyperf(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.timer == "pyperf"

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

    def test_reliability_flags_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--data-dir",
                "/data",
                "--output",
                "/out",
                "--timer",
                "pyperf",
                "--min-time",
                "1.5",
                "--min-batches",
                "3",
            ],
        )
        assert args.timer == "pyperf"
        assert args.min_time == pytest.approx(1.5)
        assert args.min_batches == 3

    def test_gcp_flags_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--data-dir",
                "/data",
                "--output",
                "/out",
                "--cloud",
                "gcp",
                "--gcp-project",
                "p",
                "--gcp-gcs-data-uri",
                "gs://b/d",
                "--gcp-gcs-results-uri",
                "gs://b/r",
                "--gcp-disk-size-gb",
                "200",
                "--gcp-dry-run",
            ],
        )
        assert args.gcp_gcs_data_uri == "gs://b/d"
        assert args.gcp_gcs_results_uri == "gs://b/r"
        assert args.gcp_disk_size_gb == 200
        assert args.gcp_dry_run is True
        assert args.gcp_attached is False

    def test_scenario_flags_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--data-dir",
                "/data",
                "--output",
                "/out",
                "--scenario",
                "video-decode-16f",
                "--mode",
                "decode",
                "--clip-length",
                "16",
                "--decoders",
                "opencv",
                "pyav",
            ],
        )
        assert args.scenario == "video-decode-16f"
        assert args.mode == "decode"
        assert args.clip_length == 16
        assert args.decoders == ["opencv", "pyav"]


class TestBuildGcpBenchmarkCliArgv:
    def test_builds_argv_without_cloud_flags(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--data-dir",
                "/ignored",
                "--output",
                "/ignored",
                "--num-runs",
                "3",
                "--num-items",
                "10",
                "--libraries",
                "kornia",
            ],
        )
        repo_root = tmp_path
        argv = build_gcp_benchmark_cli_argv(
            args,
            data_dir="/remote/data",
            output="/remote/out",
            repo_root=repo_root,
        )
        assert "--data-dir" in argv
        idx = argv.index("--data-dir")
        assert argv[idx + 1] == "/remote/data"
        assert argv[argv.index("--output") + 1] == "/remote/out"
        assert "--num-items" in argv
        assert argv[argv.index("--num-items") + 1] == "10"
        assert "--libraries" in argv
        assert "kornia" in argv
        assert "--cloud" not in argv

    def test_builds_scenario_argv(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--data-dir",
                "/ignored",
                "--output",
                "/ignored",
                "--scenario",
                "video-16f",
                "--mode",
                "pipeline",
                "--batch-size",
                "4",
                "--workers",
                "2",
                "--clip-length",
                "16",
            ],
        )
        argv = build_gcp_benchmark_cli_argv(
            args,
            data_dir="/remote/data",
            output="/remote/out",
            repo_root=tmp_path,
        )
        assert argv[argv.index("--scenario") + 1] == "video-16f"
        assert argv[argv.index("--mode") + 1] == "pipeline"
        assert argv[argv.index("--batch-size") + 1] == "4"
        assert argv[argv.index("--workers") + 1] == "2"
        assert argv[argv.index("--clip-length") + 1] == "16"

    def test_spec_must_be_inside_repo(self, tmp_path: Path) -> None:
        parser = build_parser()
        outside = tmp_path / "outside.py"
        outside.write_text('LIBRARY = "kornia"\n')
        args = parser.parse_args(
            ["run", "--data-dir", "/d", "--output", "/o", "--spec", str(outside)],
        )
        with pytest.raises(ValueError, match="inside the repository"):
            build_gcp_benchmark_cli_argv(
                args,
                data_dir="/d",
                output="/o",
                repo_root=tmp_path / "repo_only",
            )

    def test_spec_inside_repo_uses_relative_path(self, tmp_path: Path) -> None:
        parser = build_parser()
        repo_root = tmp_path / "repo"
        spec_path = repo_root / "configs" / "spec.py"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text('LIBRARY = "kornia"\n')
        args = parser.parse_args(
            ["run", "--data-dir", "/d", "--output", "/o", "--spec", str(spec_path)],
        )
        argv = build_gcp_benchmark_cli_argv(
            args,
            data_dir="/d",
            output="/o",
            repo_root=repo_root,
        )
        assert "--spec" in argv
        assert argv[argv.index("--spec") + 1] == str(spec_path.relative_to(repo_root))


class TestCmdRunGcp:
    """Tests for _cmd_run_gcp orchestration logic via mocked cloud dependencies."""

    def _base_args(self, parser: argparse.ArgumentParser, extra: list[str] | None = None) -> argparse.Namespace:
        base = ["run", "--data-dir", "/d", "--output", "/o", "--cloud", "gcp", "--gcp-project", "proj"]
        return parser.parse_args(base + (extra or []))

    def test_detached_requires_gcs_uris(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = self._base_args(parser)
        from benchmark.cli import _cmd_run_gcp

        with pytest.raises(SystemExit):
            _cmd_run_gcp(args, tmp_path, tmp_path)

    def test_attached_requires_remote_data_dir(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = self._base_args(parser, ["--gcp-attached"])
        from benchmark.cli import _cmd_run_gcp

        with pytest.raises(SystemExit):
            _cmd_run_gcp(args, tmp_path, tmp_path)

    def test_detached_dry_run_does_not_create_vm(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = self._base_args(
            parser,
            [
                "--gcp-gcs-data-uri",
                "gs://b/d",
                "--gcp-gcs-results-uri",
                "gs://b/r",
                "--gcp-dry-run",
            ],
        )
        mock_runner = MagicMock()
        mock_runner.run_detached.return_value = "gs://b/r/dryrunid"

        with (
            patch("benchmark.cloud.gcp.GCPRunner", return_value=mock_runner),
            patch("benchmark.cloud.instance.GCPInstanceConfig"),
            patch("benchmark.cloud.gcp.new_run_id", return_value="testrunid"),
        ):
            from benchmark.cli import _cmd_run_gcp

            _cmd_run_gcp(args, tmp_path, tmp_path)

        mock_runner.run_detached.assert_called_once()
        _, kwargs = mock_runner.run_detached.call_args
        assert kwargs["dry_run"] is True
        mock_runner.create_instance.assert_not_called()

    def test_detached_writes_metadata_json(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--data-dir",
                "/d",
                "--output",
                str(out_dir),
                "--cloud",
                "gcp",
                "--gcp-project",
                "proj",
                "--gcp-gcs-data-uri",
                "gs://b/data",
                "--gcp-gcs-results-uri",
                "gs://b/runs",
            ],
        )
        mock_runner = MagicMock()
        mock_runner.run_detached.return_value = "gs://b/runs/abc123"

        with (
            patch("benchmark.cloud.gcp.GCPRunner", return_value=mock_runner),
            patch("benchmark.cloud.instance.GCPInstanceConfig"),
            patch("benchmark.cloud.gcp.new_run_id", return_value="abc123"),
            patch.object(sys, "argv", ["benchmark.cli"]),
        ):
            from benchmark.cli import _cmd_run_gcp

            _cmd_run_gcp(args, tmp_path, out_dir)

        meta = json.loads((out_dir / "gcp_last_run.json").read_text())
        assert meta["run_id"] == "abc123"
        assert meta["run_prefix"] == "gs://b/runs/abc123"
        assert "fetch_results_hint" in meta

    def test_attached_calls_run_attached_with_correct_argv(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = self._base_args(
            parser,
            [
                "--gcp-attached",
                "--gcp-remote-data-dir",
                "/vm/data",
                "--media",
                "video",
                "--libraries",
                "kornia",
            ],
        )
        mock_runner = MagicMock()

        with (
            patch("benchmark.cloud.gcp.GCPRunner", return_value=mock_runner),
            patch("benchmark.cloud.instance.GCPInstanceConfig"),
        ):
            from benchmark.cli import _cmd_run_gcp

            _cmd_run_gcp(args, tmp_path, tmp_path)

        mock_runner.run_attached.assert_called_once()
        _, kwargs = mock_runner.run_attached.call_args
        argv = kwargs["remote_cli_args"]
        assert "--data-dir" in argv
        assert argv[argv.index("--data-dir") + 1] == "/vm/data"
        assert "--media" in argv
        assert "video" in argv
        assert "kornia" in argv


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
            ('LIBRARY = "kornia"\n', "kornia"),
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
