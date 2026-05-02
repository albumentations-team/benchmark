"""Tests for benchmark/cli.py argument parsing and helpers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.cli import (
    _collect_provided_flags,
    _compile_requirements,
    _extract_library,
    _library_env_group,
    _requirements_cache_key,
    _requirements_for_env_group,
    build_gcp_benchmark_cli_argv,
    build_parser,
)
from benchmark.config import BenchmarkRunConfig, config_to_namespace, load_run_config, resolve_config_transform_set
from benchmark.output_naming import micro_output_file


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

    def test_plan_subcommand_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["plan", "--config", "configs/examples/local_rgb_micro_cpu.yaml"])
        assert args.command == "plan"
        assert args.config == Path("configs/examples/local_rgb_micro_cpu.yaml")

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

    def test_run_allows_config_without_data_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--config", "config.yaml"])
        assert args.config == Path("config.yaml")

    def test_run_allows_config_without_output(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--config", "config.yaml"])
        assert args.output is None

    def test_collect_provided_flags_supports_equals_form(self) -> None:
        assert _collect_provided_flags(["run", "--config=c.yaml", "--num-items=5"]) == {
            "--config",
            "--num-items",
        }

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

    def test_transform_set_paper_parses(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out", "--transform-set", "paper"])
        assert args.transform_set == "paper"

    def test_verbose_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--verbose", "run", "--data-dir", "/data", "--output", "/out"])
        assert args.verbose is True

    def test_reliability_flags_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--data-dir",
                "/data",
                "--output",
                "/out",
                "--min-time",
                "1.5",
                "--min-batches",
                "3",
            ],
        )
        assert args.min_time == pytest.approx(1.5)
        assert args.min_batches == 3

    def test_device_flag_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out", "--device", "cuda"])

        assert args.device == "cuda"

    def test_refresh_requirements_is_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out"])
        assert args.refresh_requirements is True

    def test_no_refresh_requirements_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/data", "--output", "/out", "--no-refresh-requirements"])
        assert args.refresh_requirements is False

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
                "--gcp-venv-cache-uri",
                "gs://b/cache",
                "--gcp-force-venv-cache-rebuild",
            ],
        )
        assert args.gcp_gcs_data_uri == "gs://b/d"
        assert args.gcp_gcs_results_uri == "gs://b/r"
        assert args.gcp_disk_size_gb == 200
        assert args.gcp_dry_run is True
        assert args.gcp_venv_cache_uri == "gs://b/cache"
        assert args.gcp_force_venv_cache_rebuild is True
        assert args.gcp_attached is False
        assert args.gcp_preemptible is False

        preemptible_args = parser.parse_args(
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
                "--gcp-preemptible",
            ],
        )
        assert preemptible_args.gcp_preemptible is True

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
        assert argv[argv.index("--media") + 1] == "video"
        assert argv[argv.index("--mode") + 1] == "pipeline"
        assert argv[argv.index("--batch-size") + 1] == "4"
        assert argv[argv.index("--workers") + 1] == "2"
        assert argv[argv.index("--clip-length") + 1] == "16"

    def test_builds_no_refresh_requirements_argv(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run", "--data-dir", "/ignored", "--output", "/ignored", "--no-refresh-requirements"],
        )
        argv = build_gcp_benchmark_cli_argv(
            args,
            data_dir="/remote/data",
            output="/remote/out",
            repo_root=tmp_path,
        )
        assert "--no-refresh-requirements" in argv

    def test_builds_slow_skip_argv(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--data-dir",
                "/ignored",
                "--output",
                "/ignored",
                "--slow-threshold-sec-per-item",
                "0.2",
                "--slow-preflight-items",
                "5",
                "--disable-slow-skip",
            ],
        )
        argv = build_gcp_benchmark_cli_argv(
            args,
            data_dir="/remote/data",
            output="/remote/out",
            repo_root=tmp_path,
        )
        assert argv[argv.index("--slow-threshold-sec-per-item") + 1] == "0.2"
        assert argv[argv.index("--slow-preflight-items") + 1] == "5"
        assert "--disable-slow-skip" in argv

    def test_builds_device_argv(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--data-dir", "/ignored", "--output", "/ignored", "--device", "cuda"])

        argv = build_gcp_benchmark_cli_argv(
            args,
            data_dir="/remote/data",
            output="/remote/out",
            repo_root=tmp_path,
        )

        assert argv[argv.index("--device") + 1] == "cuda"

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


def test_micro_output_file_includes_device_suffix(tmp_path: Path) -> None:
    assert micro_output_file(tmp_path, "kornia", device="cuda").name == "kornia_micro_dev-cuda_results.json"


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
            patch("benchmark.cloud.instance.GCPInstanceConfig") as mock_config,
            patch("benchmark.cloud.gcp.new_run_id", return_value="testrunid"),
        ):
            from benchmark.cli import _cmd_run_gcp

            _cmd_run_gcp(args, tmp_path, tmp_path)

        assert mock_config.call_args.kwargs["preemptible"] is False
        mock_runner.run_detached.assert_called_once()
        _, kwargs = mock_runner.run_detached.call_args
        assert kwargs["dry_run"] is True
        assert kwargs["job"]["venv_cache_uri"] == "gs://b/augmentation-cache"
        mock_runner.create_instance.assert_not_called()

    def test_g2_machine_uses_gpu_image_without_explicit_accelerator(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = self._base_args(
            parser,
            [
                "--gcp-machine-type",
                "g2-standard-16",
                "--gcp-gcs-data-uri",
                "gs://b/d.tar",
                "--gcp-gcs-results-uri",
                "gs://b/r",
                "--gcp-dry-run",
            ],
        )
        mock_runner = MagicMock()
        mock_runner.run_detached.return_value = "gs://b/r/dryrunid"

        with (
            patch("benchmark.cloud.gcp.GCPRunner", return_value=mock_runner),
            patch("benchmark.cloud.instance.GCPInstanceConfig") as mock_config,
            patch("benchmark.cloud.gcp.new_run_id", return_value="testrunid"),
        ):
            from benchmark.cli import _cmd_run_gcp

            _cmd_run_gcp(args, tmp_path, tmp_path)

        assert mock_config.call_args.kwargs["accelerator_type"] is None
        assert mock_config.call_args.kwargs["accelerator_count"] == 0
        assert mock_config.call_args.kwargs["image_project"] == "deeplearning-platform-release"
        assert mock_config.call_args.kwargs["image_family"] == "pytorch-2-9-cu129-ubuntu-2404-nvidia-580"

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

    def test_detached_typed_job_payload_uses_vm_paths(self, tmp_path: Path) -> None:
        config = load_run_config(Path("configs/paper/gcp_g2_rgb_gpu_smoke.yaml"))
        args = config_to_namespace(config)
        args.gcp_dry_run = True
        mock_runner = MagicMock()
        mock_runner.run_detached.return_value = "gs://b/runs/abc123"

        with (
            patch("benchmark.cloud.gcp.GCPRunner", return_value=mock_runner),
            patch("benchmark.cloud.instance.GCPInstanceConfig"),
            patch("benchmark.cloud.gcp.new_run_id", return_value="abc123"),
        ):
            from benchmark.cli import _cmd_run_gcp

            _cmd_run_gcp(args, tmp_path, tmp_path, run_config=config)

        _, kwargs = mock_runner.run_detached.call_args
        run_config = kwargs["job"]["run_config"]
        assert run_config["data"]["data_dir"] == "/root/benchmark-data/val"
        assert run_config["output"]["output_dir"] == "/root/benchmark-work/results"
        assert "cloud" not in run_config

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


def test_gcp_job_dict_can_carry_typed_config() -> None:
    from benchmark.cloud.gcp import build_gcp_job_dict

    job = build_gcp_job_dict(
        run_id="abc",
        gcs_data_uri="gs://bucket/data.tar",
        benchmark_cli_args=["--scenario", "image-rgb"],
        run_config={"selection": {"scenario": "image-rgb"}},
        cloud_config={"provider": "gcp", "project": "p"},
        terminate_instance=True,
        keep_instance_on_failure=False,
        venv_cache_uri="",
        force_venv_cache_rebuild=False,
        submission={},
        instance_meta={},
    )

    assert job["run_config"] == {"selection": {"scenario": "image-rgb"}}
    assert job["cloud_config"] == {"provider": "gcp", "project": "p"}


def test_scenario_run_builds_jobs_from_resolved_config(tmp_path: Path) -> None:
    from benchmark.cli import _cmd_run_scenario

    data = load_run_config(Path("configs/examples/local_rgb_micro_cpu.yaml")).model_dump()
    data["selection"]["libraries"] = ["kornia"]
    data["data"]["data_dir"] = str(tmp_path / "data")
    config = resolve_config_transform_set(BenchmarkRunConfig.model_validate(data), Path.cwd())

    with patch("benchmark.cli.execute_job") as execute:
        _cmd_run_scenario(run_config=config, repo_root=Path.cwd(), output_dir=tmp_path / "out", verbose=False)

    job = execute.call_args.args[0]
    assert job.library == "kornia"
    assert job.mode == "micro"
    assert job.data_dir == tmp_path / "data"
    assert "HorizontalFlip" in job.transforms_filter


def test_manual_micro_job_uses_typed_config_and_preserves_micro_semantics(tmp_path: Path) -> None:
    from benchmark.cli import _run_micro_job

    config = BenchmarkRunConfig.model_validate(
        {
            "selection": {
                "mode": "pipeline",
                "media": "image",
                "libraries": ["kornia"],
                "transforms": ["HorizontalFlip"],
            },
            "data": {"data_dir": str(tmp_path / "data"), "num_items": 7, "num_channels": 9},
            "execution": {"num_runs": 3, "workers": 2, "device": "none"},
            "output": {"output_dir": str(tmp_path / "out")},
        },
    )
    spec = tmp_path / "spec.py"
    spec.write_text('LIBRARY = "kornia"\n', encoding="utf-8")

    with patch("benchmark.cli.execute_job") as execute:
        _run_micro_job(
            library="kornia",
            spec_file=spec,
            data_dir=tmp_path / "data",
            output_file=tmp_path / "out.json",
            run_config=config,
            repo_root=Path.cwd(),
            verbose=True,
        )

    job = execute.call_args.args[0]
    assert job.mode == "micro"
    assert job.scenario == "image-manual"
    assert job.workers == 0
    assert job.num_items == 7
    assert job.num_runs == 3
    assert job.num_channels == 9
    assert job.transforms_filter == ("HorizontalFlip",)


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


class TestRequirementsCacheKey:
    def test_torch_libraries_share_image_env_group(self, tmp_path: Path) -> None:
        assert _library_env_group("torchvision", "image") == "torch_stack"
        assert _library_env_group("kornia", "image") == "torch_stack"
        assert _library_env_group("pillow", "image") == "torch_stack"

        reqs = _requirements_for_env_group("torch_stack", "image", tmp_path)
        assert [path.name for path in reqs] == ["requirements.txt", "torchvision.txt", "kornia.txt", "pillow.txt"]

    def test_cache_key_changes_when_requirements_change(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("numpy\n", encoding="utf-8")
        python = tmp_path / "python"

        first = _requirements_cache_key(
            python=python,
            requirements_paths=[req],
            env_group="pillow",
            media="image",
        )
        req.write_text("numpy\npillow\n", encoding="utf-8")
        second = _requirements_cache_key(
            python=python,
            requirements_paths=[req],
            env_group="pillow",
            media="image",
        )

        assert first != second

    def test_compile_keeps_existing_lock_when_platform_resolution_fails(self, tmp_path: Path) -> None:
        req = tmp_path / "dali-video.txt"
        req.write_text("nvidia-dali-cuda120\n", encoding="utf-8")
        req.with_suffix(".in").write_text("nvidia-dali-cuda120\n", encoding="utf-8")

        with (
            patch("benchmark.envs.subprocess.run", side_effect=RuntimeError("wrong exception")),
            pytest.raises(RuntimeError),
        ):
            _compile_requirements(tmp_path / "python", req)

        with patch("benchmark.envs.subprocess.run", side_effect=subprocess.CalledProcessError(1, "uv")):
            _compile_requirements(tmp_path / "python", req)

        assert req.read_text(encoding="utf-8") == "nvidia-dali-cuda120\n"
