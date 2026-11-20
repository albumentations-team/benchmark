from __future__ import annotations

import json
import socket
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from augbench.gcp_controller import GcpControllerStore
    from augbench.guest_request import GuestRequest
    from augbench.run_records import CellKey


def claim_jobs(
    *, request: GuestRequest, cells: Iterable[CellKey], remote: GcpControllerStore
) -> Iterator[tuple[CellKey, ...]]:
    if request.pool_id is None:
        raise ValueError("job claims require a pool ID")
    jobs: dict[tuple[str, int], list[CellKey]] = defaultdict(list)
    for cell in cells:
        jobs[cell.implementation, cell.seed].append(cell)
    for (implementation, seed), job in jobs.items():
        key = f"runs/{request.run.run_id}/pools/{request.pool_id}/claims/{implementation}-{seed}.json"
        payload = json.dumps({"worker": socket.gethostname(), "cell_ids": [cell.cell_id for cell in job]}).encode()
        if remote.create_if_absent(key, payload):
            yield tuple(job)
