"""Local immutable storage for validated production cells."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from augbench.run_records import ResultRecord

_SHA256_HEX_LENGTH = 64


class ResultConflictError(RuntimeError):
    """A completed cell ID already refers to different bytes."""


class ImmutableResultStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def put(self, result: ResultRecord) -> bool:
        payload = encode_result(result)
        destination = self.path_for(result.cell_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".cell-", delete=False) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as error:
                if destination.read_bytes() != payload:
                    raise ResultConflictError(f"immutable result conflicts for {result.cell_id}") from error
                return False
        finally:
            temporary_path.unlink(missing_ok=True)
        return True

    def get(self, cell_id: str) -> ResultRecord:
        return decode_result(self.path_for(cell_id).read_bytes(), expected_cell_id=cell_id)

    def has(self, cell_id: str) -> bool:
        try:
            self.get(cell_id)
        except FileNotFoundError:
            return False
        return True

    def path_for(self, cell_id: str) -> Path:
        if len(cell_id) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in cell_id):
            raise ValueError(f"invalid cell ID {cell_id!r}")
        return self._root / "cells" / f"{cell_id}.json"


def result_payload(result: ResultRecord) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    payload["cell_id"] = result.cell_id
    payload["throughput"]["value"] = result.throughput.value
    return payload


def encode_result(result: ResultRecord) -> bytes:
    return f"{json.dumps(result_payload(result), sort_keys=True, separators=(',', ':'))}\n".encode()


def decode_result(payload: bytes, *, expected_cell_id: str) -> ResultRecord:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise TypeError("result JSON must be an object")
    cell_id = decoded.pop("cell_id", None)
    throughput = decoded.get("throughput")
    if isinstance(throughput, dict):
        throughput.pop("value", None)
    result = ResultRecord.model_validate(decoded)
    if cell_id != expected_cell_id or result.cell_id != expected_cell_id:
        raise ValueError("result cell ID does not match its storage key")
    return result
