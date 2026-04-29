"""Shared slow-threshold semantics for benchmark runners and reporting."""

from __future__ import annotations

from typing import TypedDict

SLOW_THRESHOLD_COMPARATOR = ">="


class SlowThresholdInfo(TypedDict):
    slow_threshold_sec_per_item: float
    slow_threshold_throughput: float
    slow_threshold_unit: str
    slow_marker: str


def is_slow_time_per_item(time_per_item: float, threshold_sec_per_item: float) -> bool:
    """Return True when a timing should be slow-skipped."""
    return time_per_item >= threshold_sec_per_item


def slow_threshold_throughput(threshold_sec_per_item: float) -> float:
    return 1.0 / threshold_sec_per_item if threshold_sec_per_item > 0 else 0.0


def slow_marker_from_threshold(threshold_throughput: float, unit: str) -> str:
    return f"≤{threshold_throughput:.0f} {unit}" if threshold_throughput > 0 else "slow-skipped"


def slow_threshold_info(threshold_sec_per_item: float, unit: str) -> SlowThresholdInfo:
    threshold_throughput = slow_threshold_throughput(threshold_sec_per_item)
    return {
        "slow_threshold_sec_per_item": threshold_sec_per_item,
        "slow_threshold_throughput": threshold_throughput,
        "slow_threshold_unit": unit,
        "slow_marker": slow_marker_from_threshold(threshold_throughput, unit),
    }


def slow_threshold_reason(transform_name: str, time_per_item: float, threshold_sec_per_item: float, item: str) -> str:
    return (
        f"{transform_name} slower than threshold: {time_per_item:.3f} sec/{item} "
        f"{SLOW_THRESHOLD_COMPARATOR} {threshold_sec_per_item:.3f}"
    )
