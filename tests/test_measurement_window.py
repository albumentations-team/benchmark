from collections.abc import Iterator

from augbench.measurement_window import MeasurementWindow, measure_window


def test_only_the_post_warmup_batches_contribute_to_throughput() -> None:
    batches = iter((1, 2, 3))
    ticks = iter((10.0, 14.0))
    consumed: list[int] = []
    synchronized = 0

    def consume(batch: int) -> int:
        consumed.append(batch)
        return 2

    def synchronize() -> None:
        nonlocal synchronized
        synchronized += 1

    measurement = measure_window(
        batches=batches,
        consume=consume,
        synchronize=synchronize,
        window=MeasurementWindow(warmup_batches=1, measured_batches=2),
        clock=lambda: next(ticks),
    )

    assert consumed == [1, 2, 3]
    assert synchronized == 2
    assert measurement.completed_items == 4
    assert measurement.duration_seconds == 4.0


def test_waits_record_only_the_time_spent_pulling_measured_batches() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def source() -> Iterator[int]:
        nonlocal now
        for pull_seconds in (5.0, 2.0, 3.0):
            now += pull_seconds
            yield 1

    waits: list[float] = []
    measure_window(
        batches=source(),
        consume=lambda _batch: 1,
        synchronize=lambda: None,
        window=MeasurementWindow(warmup_batches=1, measured_batches=2, waits=waits),
        clock=clock,
    )

    assert waits == [2.0, 3.0]
