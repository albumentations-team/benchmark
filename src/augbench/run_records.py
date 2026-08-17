"""Immutable run identity and the only result record written by production."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RunInputs(_Record):
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    code_archive_sha256: Sha256
    dataset_archive_sha256: Sha256
    recipe_catalog_sha256: Sha256
    environment_lock_sha256: dict[str, Sha256] = Field(min_length=1)


class RunRecord(_Record):
    schema_version: Literal[1] = 1
    run_id: Sha256
    family_config: dict[str, Any]
    inputs: RunInputs
    hardware: dict[str, str] = Field(min_length=1)
    access_order: Literal["seeded-random-without-replacement"] = "seeded-random-without-replacement"


def build_run_record(
    *,
    family_config: dict[str, Any],
    git_commit: str,
    code_archive_sha256: str,
    dataset_archive_sha256: str,
    recipe_catalog_sha256: str,
    environment_lock_sha256: dict[str, str],
    hardware: dict[str, str],
) -> RunRecord:
    inputs = RunInputs(
        git_commit=git_commit,
        code_archive_sha256=code_archive_sha256,
        dataset_archive_sha256=dataset_archive_sha256,
        recipe_catalog_sha256=recipe_catalog_sha256,
        environment_lock_sha256=environment_lock_sha256,
    )
    identity = {
        "schema_version": 1,
        "family_config": family_config,
        "inputs": inputs.model_dump(mode="json"),
        "hardware": hardware,
        "access_order": "seeded-random-without-replacement",
    }
    return RunRecord(run_id=_digest(identity), family_config=family_config, inputs=inputs, hardware=hardware)


class CellKey(_Record):
    run_id: Sha256
    family: Literal["rgb"]
    implementation: str = Field(min_length=1)
    recipe_id: str = Field(min_length=1)
    seed: int

    @property
    def cell_id(self) -> Sha256:
        return _digest(self.model_dump(exclude={"cell_id"}))


class Throughput(_Record):
    unit: Literal["images/s"] = "images/s"
    completed_items: int = Field(ge=1)
    duration_seconds: float = Field(gt=0)

    @property
    def value(self) -> float:
        return self.completed_items / self.duration_seconds


class GpuMemory(_Record):
    measurement: Literal["nvml_process_memory"] = "nvml_process_memory"
    peak_mib: float = Field(gt=0)
    poll_interval_ms: int = Field(gt=0)
    valid_samples: int = Field(ge=1)


class OutputObservation(_Record):
    device: Literal["cuda"] = "cuda"
    dtype: Literal["float16"] = "float16"
    layout: Literal["BCHW"] = "BCHW"
    shape: tuple[int, int, int, int]

    @model_validator(mode="after")
    def validate_model_ready_shape(self) -> OutputObservation:
        if self.shape[0] < 1 or any(value < 1 for value in self.shape[1:]):
            raise ValueError("output shape must have positive dimensions")
        return self


class ResultRecord(_Record):
    schema_version: Literal[1] = 1
    run_id: Sha256
    cell: CellKey
    status: Literal["ok"] = "ok"
    throughput: Throughput
    gpu_memory: GpuMemory
    output: OutputObservation
    runtime: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> ResultRecord:
        if self.run_id != self.cell.run_id:
            raise ValueError("result and cell run IDs differ")
        return self

    @property
    def cell_id(self) -> str:
        return self.cell.cell_id
