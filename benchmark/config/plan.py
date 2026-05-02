from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.cloud.paths import VM_RESULTS, staged_data_dir_for_gcs_uri
from benchmark.matrix import IMAGE_SPECS, MULTICHANNEL_IMAGE_SPECS, VIDEO_SPECS, spec_map_for_scenario
from benchmark.output_naming import manual_micro_output_file, micro_output_file, pipeline_output_file
from benchmark.scenarios import get_scenario, resolve_decoders, resolve_libraries, resolve_mode

if TYPE_CHECKING:
    from benchmark.config.models import BenchmarkRunConfig


@dataclass(frozen=True)
class PlannedJob:
    scenario: str
    mode: str
    media: str
    library: str | None
    decoder: str | None
    backend: str
    spec_file: str | None
    output_file: str
    device: str
    data_dir: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class RunPlan:
    output_dir: str
    expected_outputs: tuple[str, ...]
    jobs: tuple[PlannedJob, ...]
    cloud: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "output_dir": self.output_dir,
            "expected_outputs": list(self.expected_outputs),
            "jobs": [job.to_dict() for job in self.jobs],
        }
        if self.cloud is not None:
            data["cloud"] = self.cloud
        return data


def _cloud_plan(config: BenchmarkRunConfig) -> dict[str, object] | None:
    if not config.cloud or not config.cloud.enabled:
        return None
    return {
        "provider": config.cloud.provider,
        "project": config.cloud.project,
        "zone": config.cloud.zone,
        "machine_type": config.cloud.machine_type,
        "gpu_type": config.cloud.gpu_type,
        "attached": config.cloud.attached,
        "dry_run": config.cloud.dry_run,
        "local_output_dir": config.output.output_dir,
        "execution_output_dir": str(_base_output_dir(config)),
        "execution_data_dir": _planned_data_dir(config),
        "gcs_data_uri": config.data.gcs_uri,
        "gcs_results_uri": config.output.gcs_results_uri,
        "remote_data_dir": config.data.remote_data_dir,
    }


def _base_output_dir(config: BenchmarkRunConfig) -> Path:
    if config.cloud and config.cloud.enabled and not config.cloud.attached:
        return Path(VM_RESULTS)
    output_dir = Path(config.output.output_dir or "output")
    is_local = not (config.cloud and config.cloud.enabled)
    if config.selection.multichannel and config.resolved_media() == "image" and is_local:
        return output_dir / "multichannel"
    return output_dir


def _planned_data_dir(config: BenchmarkRunConfig) -> str:
    if config.cloud and config.cloud.enabled:
        if config.cloud.attached:
            return config.data.remote_data_dir or "unknown"
        return staged_data_dir_for_gcs_uri(config.data.gcs_uri)
    return config.data.data_dir or config.data.gcs_uri or "unknown"


def _planned_job(
    *,
    config: BenchmarkRunConfig,
    scenario: str,
    mode: str,
    media: str,
    output_file: Path,
    data_dir: str,
    library: str | None = None,
    decoder: str | None = None,
    backend: str,
    spec_file: Path | None = None,
) -> PlannedJob:
    return PlannedJob(
        scenario=scenario,
        mode=mode,
        media=media,
        library=library,
        decoder=decoder,
        backend=backend,
        spec_file=str(spec_file) if spec_file is not None else None,
        output_file=str(output_file),
        device=config.execution.device,
        data_dir=data_dir,
    )


def _scenario_jobs(config: BenchmarkRunConfig, repo_root: Path, output_dir: Path, data_dir: str) -> list[PlannedJob]:
    if config.selection.scenario is None:
        return []

    scenario = get_scenario(config.selection.scenario)
    mode = resolve_mode(scenario, config.selection.mode)
    scenario_output_dir = output_dir / scenario.name / mode
    if mode == "decode":
        decoders = resolve_decoders(scenario, config.selection.decoders)
        decode_jobs = [
            _planned_job(
                config=config,
                scenario=scenario.name,
                mode=mode,
                media=scenario.media,
                output_file=scenario_output_dir / f"{decoder}_decode_results.json",
                data_dir=data_dir,
                decoder=decoder,
                backend="decode",
            )
            for decoder in decoders
        ]
        decode_jobs.append(
            _planned_job(
                config=config,
                scenario=scenario.name,
                mode=mode,
                media=scenario.media,
                output_file=scenario_output_dir / "video_decode_results.json",
                data_dir=data_dir,
                decoder=None,
                backend="decode",
            ),
        )
        return decode_jobs

    libraries = resolve_libraries(scenario, mode, config.selection.libraries)
    spec_map = spec_map_for_scenario(scenario.name, mode)
    jobs: list[PlannedJob] = []
    for library in libraries:
        backend = (
            "dali_pipeline"
            if mode == "pipeline" and library == "dali"
            else "pipeline"
            if mode == "pipeline"
            else "pyperf"
        )
        spec_file = None if backend == "dali_pipeline" else repo_root / spec_map[library]
        output_file = (
            pipeline_output_file(
                scenario_output_dir,
                library,
                pipeline_scope=config.execution.pipeline_scope,
                num_items=config.data.num_items,
                num_runs=config.execution.num_runs,
                workers=config.execution.workers,
                batch_size=config.execution.batch_size,
                device=config.execution.device,
            )
            if mode == "pipeline"
            else micro_output_file(scenario_output_dir, library, device=config.execution.device)
        )
        jobs.append(
            _planned_job(
                config=config,
                scenario=scenario.name,
                mode=mode,
                media=scenario.media,
                output_file=output_file,
                data_dir=data_dir,
                library=library,
                backend=backend,
                spec_file=spec_file,
            ),
        )
    return jobs


def _manual_spec_jobs(config: BenchmarkRunConfig, repo_root: Path, output_dir: Path, data_dir: str) -> list[PlannedJob]:
    if config.selection.spec is None:
        return []
    spec_file = Path(config.selection.spec)
    if not spec_file.is_absolute():
        spec_file = repo_root / spec_file
    return [
        _planned_job(
            config=config,
            scenario=f"{config.resolved_media()}-manual",
            mode="micro",
            media=config.resolved_media(),
            output_file=output_dir / f"{spec_file.stem}.json",
            data_dir=data_dir,
            library=None,
            backend="pyperf",
            spec_file=spec_file,
        ),
    ]


def _manual_library_jobs(
    config: BenchmarkRunConfig,
    repo_root: Path,
    output_dir: Path,
    data_dir: str,
) -> list[PlannedJob]:
    if config.selection.scenario or config.selection.spec:
        return []
    media = config.resolved_media()
    spec_map = (
        MULTICHANNEL_IMAGE_SPECS
        if config.selection.multichannel and media == "image"
        else VIDEO_SPECS
        if media == "video"
        else IMAGE_SPECS
    )
    libraries = config.selection.libraries or list(spec_map)
    return [
        _planned_job(
            config=config,
            scenario=f"{media}-manual",
            mode="micro",
            media=media,
            output_file=manual_micro_output_file(
                output_dir,
                library,
                media=media,
                device=config.execution.device,
            ),
            data_dir=data_dir,
            library=library,
            backend="pyperf",
            spec_file=repo_root / spec_map[library],
        )
        for library in libraries
    ]


def build_run_plan(config: BenchmarkRunConfig, repo_root: Path) -> RunPlan:
    output_dir = _base_output_dir(config)
    data_dir = _planned_data_dir(config)
    jobs = (
        _scenario_jobs(config, repo_root, output_dir, data_dir)
        or _manual_spec_jobs(config, repo_root, output_dir, data_dir)
        or _manual_library_jobs(config, repo_root, output_dir, data_dir)
    )
    return RunPlan(
        output_dir=str(output_dir),
        expected_outputs=tuple(job.output_file for job in jobs),
        jobs=tuple(jobs),
        cloud=_cloud_plan(config),
    )
