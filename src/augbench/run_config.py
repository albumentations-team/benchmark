"""The small, explicit contract for one benchmark family."""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_RGB_CHANNELS = 3


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OutputConfig(_ConfigModel):
    device: Literal["cuda"] = "cuda"
    channels: int = Field(ge=1)
    height: int = Field(ge=1)
    width: int = Field(ge=1)
    dtype: Literal["float16"] = "float16"
    layout: Literal["BCHW"] = "BCHW"

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)


class DatasetConfig(_ConfigModel):
    archive_uri: str = Field(pattern=r"^gs://")
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_member_prefix: str = Field(min_length=1)
    archive_member_suffix: str = Field(min_length=1)
    item_count: int = Field(ge=1)
    selection: Literal["first-sorted"]
    access_order: Literal["seeded-random-without-replacement"]


class ExecutionConfig(_ConfigModel):
    batch_size: int = Field(ge=1)
    warmup_batches: int = Field(ge=0)
    measured_batches: int = Field(ge=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    num_workers: int = Field(ge=0)
    prefetch_factor: int | None = Field(default=None, ge=1)
    persistent_workers: bool = True
    prewarm_dataset: bool

    @model_validator(mode="after")
    def validate_workers(self) -> ExecutionConfig:
        if self.num_workers == 0 and (self.prefetch_factor is not None or self.persistent_workers):
            raise ValueError("prefetch_factor and persistent_workers require num_workers > 0")
        return self

    @property
    def required_items(self) -> int:
        return self.batch_size * (self.warmup_batches + self.measured_batches)


class FamilyRunConfig(_ConfigModel):
    schema_version: Literal[1] = 1
    family: Literal["rgb", "image9ch", "video", "volume"]
    output: OutputConfig
    normalization: Literal["gpu"]
    dataset: DatasetConfig
    recipes: pathlib.Path
    coverage_catalog: pathlib.Path
    cloud: pathlib.Path
    execution: ExecutionConfig
    implementations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_runnable_family(self) -> FamilyRunConfig:
        if self.family != "rgb":
            raise ValueError("only the RGB family is runnable until its dataset and adapters are added")
        if self.output.channels != _RGB_CHANNELS:
            raise ValueError("the active RGB contract requires three channels")
        if self.normalization != "gpu":
            raise ValueError("normalization is always applied to the collated CUDA batch")
        if self.dataset.item_count < self.execution.required_items:
            raise ValueError("dataset is shorter than the warmup plus measured window")
        return self


def load_family_config(path: pathlib.Path) -> FamilyRunConfig:
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return FamilyRunConfig.model_validate(payload)


class GcpRunConfig(_ConfigModel):
    schema_version: Literal[1] = 1
    project: str = Field(min_length=1)
    gcs_base_uri: str = Field(pattern=r"^gs://")
    zones: Literal["all"]
    machine_type: Literal["g2-standard-16"]
    accelerator: Literal["nvidia-l4"]
    image: str = Field(min_length=1)
    service_account: str = Field(min_length=1)
    python_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    provisioning_model: Literal["STANDARD"]
    boot_disk_size_gib: int = Field(ge=50)
    provisioning_deadline_seconds: int = Field(ge=60)
    run_deadline_seconds: int = Field(ge=60)


def load_gcp_config(path: pathlib.Path) -> GcpRunConfig:
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return GcpRunConfig.model_validate(payload)


def source_order(*, item_count: int, required_items: int, seed: int) -> tuple[int, ...]:
    """Return the per-seed random-access sequence without allowing rollover.

    The files are read once sequentially before all cells to prewarm the cache.
    Timed cells use this deterministic permutation, which is shared by every
    implementation for the same seed.
    """
    if item_count < 1:
        raise ValueError("dataset item_count must be positive")
    if required_items < 1:
        raise ValueError("required_items must be positive")
    if required_items > item_count:
        raise ValueError("dataset is shorter than the requested measurement window")
    generator = np.random.Generator(np.random.PCG64(seed))
    return tuple(int(index) for index in generator.permutation(item_count)[:required_items])
