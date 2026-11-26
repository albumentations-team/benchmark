"""The exact steady-state timing boundary used by every production cell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augbench.run_records import Throughput

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True)
class MeasurementWindow:
    warmup_batches: int
    measured_batches: int
    waits: list[float] | None = field(default=None, compare=False)


def measure_window[Batch](
    *,
    batches: Iterator[Batch],
    consume: Callable[[Batch], int],
    synchronize: Callable[[], None],
    window: MeasurementWindow,
    clock: Callable[[], float],
) -> Throughput:
    """Warm up, synchronize, time the fixed window, and synchronize again.

    When ``window.waits`` is a list, it receives the seconds spent pulling each measured batch.
    """
    if window.warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")
    if window.measured_batches < 1:
        raise ValueError("measured_batches must be positive")
    for _ in range(window.warmup_batches):
        consume(_next(batches, phase="warmup"))
    synchronize()
    started = clock()
    completed_items = 0
    for _ in range(window.measured_batches):
        pulled = clock() if window.waits is not None else 0.0
        batch = _next(batches, phase="measurement")
        if window.waits is not None:
            window.waits.append(clock() - pulled)
        completed_items += consume(batch)
    synchronize()
    duration_seconds = clock() - started
    return Throughput(completed_items=completed_items, duration_seconds=duration_seconds)


def _next[Batch](batches: Iterator[Batch], *, phase: str) -> Batch:
    try:
        return next(batches)
    except StopIteration as error:
        raise RuntimeError(f"source exhausted during {phase}; no epoch rollover is allowed") from error
