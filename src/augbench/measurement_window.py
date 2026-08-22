"""The exact steady-state timing boundary used by every production cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from augbench.run_records import Throughput

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True)
class MeasurementWindow:
    warmup_batches: int
    measured_batches: int


def measure_window[Batch](
    *,
    batches: Iterator[Batch],
    consume: Callable[[Batch], int],
    synchronize: Callable[[], None],
    window: MeasurementWindow,
    clock: Callable[[], float],
) -> Throughput:
    """Warm up, synchronize, time the fixed window, and synchronize again."""
    if window.warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")
    if window.measured_batches < 1:
        raise ValueError("measured_batches must be positive")
    for _ in range(window.warmup_batches):
        consume(_next(batches, phase="warmup"))
    synchronize()
    started = clock()
    completed_items = sum(consume(_next(batches, phase="measurement")) for _ in range(window.measured_batches))
    synchronize()
    duration_seconds = clock() - started
    return Throughput(completed_items=completed_items, duration_seconds=duration_seconds)


def _next[Batch](batches: Iterator[Batch], *, phase: str) -> Batch:
    try:
        return next(batches)
    except StopIteration as error:
        raise RuntimeError(f"source exhausted during {phase}; no epoch rollover is allowed") from error
