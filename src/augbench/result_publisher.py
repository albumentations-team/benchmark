"""Publish a validated result through GCS conditional object creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from augbench.result_store import ResultConflictError, encode_result

if TYPE_CHECKING:
    from augbench.run_records import ResultRecord


class ConditionalObjectStore(Protocol):
    def create_if_absent(self, key: str, payload: bytes) -> bool: ...

    def read(self, key: str) -> bytes: ...


class ResultPublisher:
    def __init__(self, remote: ConditionalObjectStore) -> None:
        self._remote = remote

    def publish(self, result: ResultRecord) -> bool:
        payload = encode_result(result)
        key = f"runs/{result.run_id}/cells/{result.cell_id}.json"
        created = self._remote.create_if_absent(key, payload)
        if not created and self._remote.read(key) != payload:
            raise ResultConflictError(f"remote immutable result conflicts for {result.cell_id}")
        return created
