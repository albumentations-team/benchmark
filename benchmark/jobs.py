from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


@dataclass(frozen=True)
class BenchmarkJob:
    library: str
    scenario: str
    mode: Literal["micro", "pipeline"]
    media: Literal["image", "video"]
    data_dir: Path
    output_file: Path
    num_items: int | None
    num_runs: int
    num_channels: int
    clip_length: int = 16
    spec_file: Path | None = None
    transforms_filter: tuple[str, ...] = ()
    pipeline_scope: str = "decode_dataloader_augment"
    batch_size: int = 32
    workers: int = 0
    min_time: float = 0.0
    min_batches: int = 1
    device: str = "none"
    thread_policy: str = "pipeline-default"
    refresh_requirements: bool = True
    slow_threshold_sec_per_item: float | None = None
    slow_preflight_items: int | None = None
    disable_slow_skip: bool = False
    backend: Literal["pyperf", "pipeline", "dali_pipeline"] = "pyperf"

    @classmethod
    def from_args(
        cls,
        *,
        library: str,
        scenario_name: str,
        mode: Literal["micro", "pipeline"],
        media: Literal["image", "video"],
        data_dir: Path,
        output_file: Path,
        args: argparse.Namespace,
        num_channels: int,
        clip_length: int,
        spec_file: Path | None,
        backend: Literal["pyperf", "pipeline", "dali_pipeline"] | None = None,
    ) -> BenchmarkJob:
        selected_backend = backend or ("pyperf" if mode == "micro" else "pipeline")
        return cls(
            library=library,
            scenario=scenario_name,
            mode=mode,
            media=media,
            data_dir=data_dir,
            output_file=output_file,
            num_items=args.num_items,
            num_runs=args.num_runs,
            num_channels=num_channels,
            clip_length=clip_length,
            spec_file=spec_file,
            transforms_filter=tuple(args.transforms or ()),
            pipeline_scope=args.pipeline_scope,
            batch_size=args.batch_size,
            workers=args.workers,
            min_time=args.min_time,
            min_batches=args.min_batches,
            device=args.device,
            thread_policy=args.thread_policy,
            refresh_requirements=args.refresh_requirements,
            slow_threshold_sec_per_item=args.slow_threshold_sec_per_item,
            slow_preflight_items=args.slow_preflight_items,
            disable_slow_skip=args.disable_slow_skip,
            backend=selected_backend,
        )

    def pyperf_output_file(self) -> Path:
        return self.output_file.with_suffix(".pyperf.json")

    def micro_command(self, python: Path) -> list[str]:
        if self.spec_file is None:
            msg = f"{self.library} micro job requires a spec file"
            raise ValueError(msg)
        cmd = [
            str(python),
            "-m",
            "benchmark.pyperf_micro_runner",
            "--specs-file",
            str(self.spec_file),
            "--data-dir",
            str(self.data_dir),
            "--json-output",
            str(self.output_file),
            "--output",
            str(self.pyperf_output_file()),
            "--media",
            self.media,
            "--scenario",
            self.scenario,
            "--num-channels",
            str(self.num_channels),
            "--processes",
            "1",
            "--values",
            str(self.num_runs),
        ]
        if self.num_items is not None:
            cmd += ["--num-items", str(self.num_items)]
        if self.transforms_filter:
            cmd += ["--transforms", ",".join(self.transforms_filter)]
        self._append_slow_skip_args(cmd)
        return cmd

    def pipeline_command(self, python: Path) -> list[str]:
        if self.spec_file is None:
            msg = f"{self.library} pipeline job requires a spec file"
            raise ValueError(msg)
        cmd = [
            str(python),
            "-m",
            "benchmark.pipeline_runner",
            "--specs-file",
            str(self.spec_file),
            "--data-dir",
            str(self.data_dir),
            "--output",
            str(self.output_file),
            "--media",
            self.media,
            "--scenario",
            self.scenario,
            "--num-runs",
            str(self.num_runs),
            "--batch-size",
            str(self.batch_size),
            "--workers",
            str(self.workers),
            "--min-time",
            str(self.min_time),
            "--min-batches",
            str(self.min_batches),
            "--num-channels",
            str(self.num_channels),
            "--clip-length",
            str(self.clip_length),
            "--pipeline-scope",
            self.pipeline_scope,
            "--device",
            self.device,
            "--thread-policy",
            self.thread_policy,
        ]
        if self.num_items is not None:
            cmd += ["--num-items", str(self.num_items)]
        self._append_slow_skip_args(cmd)
        return cmd

    def env_extra(self, *, verbose: bool = False) -> dict[str, str]:
        env_extra: dict[str, str] = {}
        if self.transforms_filter:
            env_extra["BENCHMARK_TRANSFORMS_FILTER"] = ",".join(self.transforms_filter)
        if verbose:
            env_extra["BENCHMARK_VERBOSE"] = "1"
        return env_extra

    def _append_slow_skip_args(self, cmd: list[str]) -> None:
        if self.slow_threshold_sec_per_item is not None:
            cmd += ["--slow-threshold-sec-per-item", str(self.slow_threshold_sec_per_item)]
        if self.slow_preflight_items is not None:
            cmd += ["--slow-preflight-items", str(self.slow_preflight_items)]
        if self.disable_slow_skip:
            cmd.append("--disable-slow-skip")
