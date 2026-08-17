"""The complete, immutable request consumed by one benchmark VM."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from augbench.run_records import RunRecord  # noqa: TC001 - Pydantic resolves this nested model at runtime.

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class GuestRequest(BaseModel):
    """Describe one resumed run without mutable controller checkpoints.

    The controller derives ``pending_cell_ids`` from immutable GCS cells before
    creating the VM.  The guest may safely be started again with a smaller
    tuple after an interruption; it never needs an attempt record or shard.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    run: RunRecord
    gcs_base_uri: str = Field(pattern=r"^gs://")
    code_archive_uri: str = Field(pattern=r"^gs://")
    dataset_archive_uri: str = Field(pattern=r"^gs://")
    dataset_archive_sha256: Sha256
    environment_cache_uri: str = Field(pattern=r"^gs://")
    environment_lock_path: str = Field(min_length=1)
    environment_lock_sha256: Sha256
    environment_python_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    pending_cell_ids: tuple[Sha256, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inputs(self) -> GuestRequest:
        if self.dataset_archive_sha256 != self.run.inputs.dataset_archive_sha256:
            raise ValueError("guest dataset archive checksum differs from the immutable run")
        if self.environment_lock_sha256 not in self.run.inputs.environment_lock_sha256.values():
            raise ValueError("guest environment lock checksum differs from the immutable run")
        if len(set(self.pending_cell_ids)) != len(self.pending_cell_ids):
            raise ValueError("guest request contains duplicate cell IDs")
        return self
