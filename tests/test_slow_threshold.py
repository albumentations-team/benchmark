from benchmark.slow_threshold import (
    is_slow_time_per_item,
    slow_marker_from_threshold,
    slow_threshold_info,
    slow_threshold_reason,
)


def test_slow_threshold_is_inclusive() -> None:
    assert is_slow_time_per_item(0.1, 0.1)
    assert is_slow_time_per_item(0.2, 0.1)
    assert not is_slow_time_per_item(0.099, 0.1)


def test_slow_threshold_info_formats_throughput_floor() -> None:
    assert slow_threshold_info(0.1, "img/s") == {
        "slow_threshold_sec_per_item": 0.1,
        "slow_threshold_throughput": 10.0,
        "slow_threshold_unit": "img/s",
        "slow_marker": "≤10 img/s",
    }


def test_slow_marker_from_threshold_handles_zero() -> None:
    assert slow_marker_from_threshold(0.0, "img/s") == "slow-skipped"


def test_slow_threshold_reason_uses_shared_comparator() -> None:
    reason = slow_threshold_reason("Elastic", 0.1234, 0.1, "image")

    assert reason == "Elastic slower than threshold: 0.123 sec/image >= 0.100"
