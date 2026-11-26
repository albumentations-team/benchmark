"""Atomic GCS claims that let a fleet of VMs share one run's (implementation, seed) jobs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Collection


class ClaimStore(Protocol):
    def create_if_absent(self, key: str, payload: bytes) -> bool: ...


class ClaimReaderStore(Protocol):
    def read(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> tuple[str, ...]: ...

    def delete(self, key: str) -> None: ...


def claims_prefix(run_id: str) -> str:
    return f"runs/{run_id}/claims"


def claim_key(run_id: str, implementation: str, seed: int) -> str:
    return f"{claims_prefix(run_id)}/{implementation}-{seed}.json"


def claim_job(*, store: ClaimStore, run_id: str, implementation: str, seed: int, instance: str) -> bool:
    """Return True only for the one VM whose conditional create wins the job."""
    payload = json.dumps({"implementation": implementation, "instance": instance, "seed": seed}, sort_keys=True)
    return store.create_if_absent(claim_key(run_id, implementation, seed), f"{payload}\n".encode())


def release_stale_claims(*, store: ClaimReaderStore, run_id: str, active_instances: Collection[str]) -> int:
    """Delete claims whose VM is gone so the next launch can rerun their missing cells."""
    released = 0
    for key in store.list_keys(claims_prefix(run_id)):
        if json.loads(store.read(key))["instance"] not in active_instances:
            store.delete(key)
            released += 1
    return released
