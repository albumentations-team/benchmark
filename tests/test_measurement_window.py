from augbench.measurement_window import measure_window


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
        warmup_batches=1,
        measured_batches=2,
        clock=lambda: next(ticks),
    )

    assert consumed == [1, 2, 3]
    assert synchronized == 2
    assert measurement.completed_items == 4
    assert measurement.duration_seconds == 4.0
