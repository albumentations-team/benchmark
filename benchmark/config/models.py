from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmark.devices import ensure_supported_device
from benchmark.scenarios import get_scenario, resolve_libraries, resolve_mode

BenchmarkMode = Literal["micro", "pipeline", "decode"]
DeviceOption = Literal["none", "cuda", "mps", "auto"]
PipelineScope = Literal[
    "memory_dataloader_augment",
    "decode_dataloader_augment",
    "decode_dataloader_augment_batch_copy",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SelectionConfig(StrictModel):
    scenario: str | None = None
    mode: BenchmarkMode | None = None
    media: Literal["image", "video"] = "image"
    libraries: list[str] | None = None
    transforms: list[str] | None = None
    transform_set: Literal["paper"] | None = None
    spec: str | None = None
    multichannel: bool = False
    decoders: list[str] | None = None


class DataConfig(StrictModel):
    data_dir: str | None = None
    gcs_uri: str | None = None
    remote_data_dir: str | None = None
    num_items: int | None = Field(default=None, ge=1)
    num_channels: int = Field(default=3, ge=1)
    clip_length: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_channels(self) -> DataConfig:
        if self.num_channels % 3 != 0:
            msg = f"num_channels must be a multiple of 3, got {self.num_channels}"
            raise ValueError(msg)
        return self


class ExecutionConfig(StrictModel):
    num_runs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=32, ge=1)
    workers: int = Field(default=0, ge=0)
    min_time: float = Field(default=0.0, ge=0.0)
    min_batches: int = Field(default=1, ge=1)
    pipeline_scope: PipelineScope = "decode_dataloader_augment"
    device: DeviceOption = "none"
    thread_policy: Literal["micro-single", "pipeline-default", "pipeline-single-worker"] | None = None
    refresh_requirements: bool = True
    slow_threshold_sec_per_item: float | None = Field(default=None, gt=0.0)
    slow_preflight_items: int | None = Field(default=None, ge=1)
    disable_slow_skip: bool = False


class OutputConfig(StrictModel):
    output_dir: str | None = None
    gcs_results_uri: str | None = None


class CloudConfig(StrictModel):
    provider: Literal["gcp"] | None = None
    project: str | None = None
    zone: str = "us-central1-a"
    machine_type: str = "n1-standard-8"
    gpu_type: str | None = None
    attached: bool = False
    dry_run: bool = False
    disk_size_gb: int = Field(default=100, ge=20)
    keep_instance: bool = False
    keep_on_failure: bool = False
    preemptible: bool = False
    remote_repo_dir: str = "~/benchmark"
    venv_cache_uri: str | None = None
    no_venv_cache: bool = False
    force_venv_cache_rebuild: bool = False

    @property
    def enabled(self) -> bool:
        return self.provider is not None


class BenchmarkRunConfig(StrictModel):
    version: int = 1
    selection: SelectionConfig
    data: DataConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig
    cloud: CloudConfig | None = None

    @model_validator(mode="after")
    def validate_run(self) -> BenchmarkRunConfig:
        mode = self._resolved_mode()
        media = self._resolved_media()
        libraries = self._resolved_libraries(mode)

        if mode == "micro" and self.execution.workers != 0:
            raise ValueError("micro benchmarks do not use DataLoader workers; set execution.workers to 0")

        if self.selection.multichannel and media != "image":
            raise ValueError("selection.multichannel is only valid for image benchmarks")

        if self.selection.spec is None:
            for library in libraries:
                ensure_supported_device(library, media, self.execution.device)

        if self.cloud and self.cloud.enabled:
            self._validate_cloud()
        elif not self.data.data_dir:
            raise ValueError("data.data_dir is required for local runs")

        if not self.output.output_dir:
            raise ValueError("output.output_dir is required")

        return self

    def _resolved_mode(self) -> BenchmarkMode:
        if self.selection.scenario:
            return resolve_mode(get_scenario(self.selection.scenario), self.selection.mode)
        return self.selection.mode or "micro"

    def _resolved_media(self) -> Literal["image", "video"]:
        if self.selection.scenario:
            return get_scenario(self.selection.scenario).media
        return self.selection.media

    def _resolved_libraries(self, mode: BenchmarkMode) -> list[str]:
        if self.selection.spec:
            return []
        if self.selection.scenario:
            return resolve_libraries(get_scenario(self.selection.scenario), mode, self.selection.libraries)
        return list(self.selection.libraries or [])

    def _validate_cloud(self) -> None:
        if self.cloud is None:
            return
        if self.cloud.provider != "gcp":
            raise ValueError("only GCP cloud execution is supported")
        if not self.cloud.project:
            raise ValueError("cloud.project is required for GCP runs")
        if self.cloud.attached:
            if not self.data.remote_data_dir:
                raise ValueError("data.remote_data_dir is required for attached GCP runs")
            return
        if not self.data.gcs_uri:
            raise ValueError("data.gcs_uri is required for detached GCP runs")
        if not self.output.gcs_results_uri:
            raise ValueError("output.gcs_results_uri is required for detached GCP runs")

    def to_legacy_args(self) -> dict[str, Any]:
        cloud = self.cloud or CloudConfig()
        return {
            "data_dir": self.data.data_dir or "unused",
            "output": self.output.output_dir or "output",
            "media": self._resolved_media(),
            "libraries": self.selection.libraries,
            "transforms": self.selection.transforms,
            "transform_set": self.selection.transform_set,
            "spec": self.selection.spec,
            "scenario": self.selection.scenario,
            "mode": self._resolved_mode(),
            "batch_size": self.execution.batch_size,
            "workers": self.execution.workers,
            "min_time": self.execution.min_time,
            "min_batches": self.execution.min_batches,
            "pipeline_scope": self.execution.pipeline_scope,
            "device": self.execution.device,
            "thread_policy": self.execution.thread_policy,
            "clip_length": self.data.clip_length,
            "decoders": self.selection.decoders,
            "cloud": cloud.provider,
            "gcp_project": cloud.project,
            "gcp_zone": cloud.zone,
            "gcp_machine_type": cloud.machine_type,
            "gcp_gpu_type": cloud.gpu_type,
            "gcp_remote_data_dir": self.data.remote_data_dir,
            "gcp_remote_repo_dir": cloud.remote_repo_dir,
            "gcp_gcs_data_uri": self.data.gcs_uri,
            "gcp_gcs_results_uri": self.output.gcs_results_uri,
            "gcp_attached": cloud.attached,
            "gcp_dry_run": cloud.dry_run,
            "gcp_disk_size_gb": cloud.disk_size_gb,
            "gcp_keep_instance": cloud.keep_instance,
            "gcp_keep_on_failure": cloud.keep_on_failure,
            "gcp_preemptible": cloud.preemptible,
            "gcp_venv_cache_uri": cloud.venv_cache_uri,
            "gcp_no_venv_cache": cloud.no_venv_cache,
            "gcp_force_venv_cache_rebuild": cloud.force_venv_cache_rebuild,
            "num_items": self.data.num_items,
            "num_runs": self.execution.num_runs,
            "slow_threshold_sec_per_item": self.execution.slow_threshold_sec_per_item,
            "slow_preflight_items": self.execution.slow_preflight_items,
            "disable_slow_skip": self.execution.disable_slow_skip,
            "refresh_requirements": self.execution.refresh_requirements,
            "num_channels": self.data.num_channels,
            "multichannel": self.selection.multichannel,
        }
