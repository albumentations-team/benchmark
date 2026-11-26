from math import comb

import pytest

from paper.intervals import _mann_whitney_counts, median_interval, ratio_interval


@pytest.mark.parametrize(("size", "other"), [(3, 3), (5, 5), (5, 3), (1, 4)])
def test_mann_whitney_counts_cover_every_rank_arrangement(size: int, other: int) -> None:
    counts = _mann_whitney_counts(size, other)

    assert sum(counts) == comb(size + other, size)
    assert len(counts) == size * other + 1
    assert counts == counts[::-1]


def test_five_seeds_give_the_exact_sample_range_for_the_median() -> None:
    interval = median_interval([5.0, 1.0, 3.0, 2.0, 4.0])

    assert (interval.low, interval.high) == (1.0, 5.0)
    assert interval.confidence == pytest.approx(0.9375)


def test_more_seeds_trim_the_median_interval() -> None:
    interval = median_interval(list(range(1, 21)))

    assert (interval.low, interval.high) == (6, 15)
    assert interval.confidence >= 0.95


def test_complete_separation_of_five_seeds_is_a_clear_difference() -> None:
    result = ratio_interval([200, 201, 202, 203, 204], [100, 101, 102, 103, 104])

    assert result.verdict == "faster"
    assert result.confidence == pytest.approx(1 - 2 * 4 / 252)
    assert result.low > 1


def test_overlapping_seeds_are_indistinguishable() -> None:
    result = ratio_interval([100, 110, 90, 105, 95], [102, 98, 108, 92, 100])

    assert result.verdict == "indistinguishable"
    assert result.low < 1 < result.high


def test_slower_cell_is_detected_symmetrically() -> None:
    assert ratio_interval([100, 101, 102, 103, 104], [200, 201, 202, 203, 204]).verdict == "slower"


def test_three_seeds_report_the_coverage_they_actually_reach() -> None:
    result = ratio_interval([200, 201, 202], [100, 101, 102])

    assert result.confidence == pytest.approx(0.9)
    assert result.verdict == "faster"
