# ruff: noqa: INP001
"""Exact, distribution-free intervals for seed-level throughput.

Each cell contributes one throughput value per seed, and seeds run on different VMs, so the
values carry both input-order and machine variation. The intervals below assume only that
the seeds are exchangeable. They never use a normal approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import comb
from statistics import median
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIDENCE = 0.95
Verdict = Literal["faster", "indistinguishable", "slower"]


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    confidence: float


@dataclass(frozen=True)
class RatioInterval:
    """Interval for the throughput ratio of two cells; ``ratio`` is the ratio of their medians."""

    ratio: float
    low: float
    high: float
    confidence: float

    @property
    def verdict(self) -> Verdict:
        if self.low > 1:
            return "faster"
        if self.high < 1:
            return "slower"
        return "indistinguishable"


def median_interval(values: Sequence[float], confidence: float = CONFIDENCE) -> Interval:
    """Order-statistic interval ``[x(r), x(n+1-r)]`` for the median.

    The largest ``r`` whose coverage reaches ``confidence`` is used. When even ``r = 1`` falls short,
    the sample range is returned with its exact, lower coverage: 93.75% for five seeds.
    """
    ordered = sorted(values)
    count = len(ordered)
    tail = (1 - confidence) / 2
    rank = 1
    cumulative = 0.0
    for candidate in range(1, count // 2 + 1):
        cumulative += comb(count, candidate - 1) / 2**count
        if cumulative > tail:
            break
        rank = candidate
    achieved = 1 - 2 * sum(comb(count, index) for index in range(rank)) / 2**count
    return Interval(ordered[rank - 1], ordered[count - rank], achieved)


def ratio_interval(
    numerator: Sequence[float], denominator: Sequence[float], confidence: float = CONFIDENCE
) -> RatioInterval:
    """Hodges-Lehmann interval for a ratio of locations from the exact Mann-Whitney distribution.

    Order the ``n * m`` pairwise ratios. The interval ``[r(k+1), r(nm-k)]`` has coverage
    ``1 - 2 P(U <= k)`` under exchangeability. The largest ``k`` that reaches ``confidence`` is
    used. Small samples may fall short, and the achieved coverage is reported.
    """
    size, other = len(numerator), len(denominator)
    counts = _mann_whitney_counts(size, other)
    total = comb(size + other, size)
    tail = (1 - confidence) / 2
    cutoff = 0
    cumulative = 0
    for statistic, count in enumerate(counts):
        cumulative += count
        if cumulative / total > tail:
            break
        cutoff = statistic
    achieved = 1 - 2 * sum(counts[: cutoff + 1]) / total
    ratios = sorted(a / b for a in numerator for b in denominator)
    return RatioInterval(
        median(numerator) / median(denominator), ratios[cutoff], ratios[size * other - 1 - cutoff], achieved
    )


@cache
def _mann_whitney_counts(size: int, other: int) -> tuple[int, ...]:
    """Number of rank arrangements for each U, the count of pairs where a first-sample value is larger."""
    if size == 0 or other == 0:
        return (1,)
    # The largest value belongs to the first sample (it beats all `other` values) or to the second.
    first = _mann_whitney_counts(size - 1, other)
    second = _mann_whitney_counts(size, other - 1)
    counts = [0] * (size * other + 1)
    for statistic, count in enumerate(first):
        counts[statistic + other] += count
    for statistic, count in enumerate(second):
        counts[statistic] += count
    return tuple(counts)
