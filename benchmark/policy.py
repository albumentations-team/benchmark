from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

MediaName = Literal["image", "video"]


@dataclass(frozen=True)
class SlowSkipPolicy:
    threshold_sec_per_item: float
    preflight_items: int
    max_preflight_secs: float


@dataclass(frozen=True)
class MediaBenchmarkPolicy:
    media: MediaName
    num_items: int
    max_warmup_iterations: int
    warmup_subset_size: int
    slow_skip: SlowSkipPolicy
    min_iterations_before_stopping: int
    item_label: str
    item_label_singular: str
    throughput_unit: str


MEDIA_POLICIES: dict[MediaName, MediaBenchmarkPolicy] = {
    "image": MediaBenchmarkPolicy(
        media="image",
        num_items=1000,
        max_warmup_iterations=1000,
        warmup_subset_size=10,
        slow_skip=SlowSkipPolicy(
            threshold_sec_per_item=0.1,
            preflight_items=10,
            max_preflight_secs=60.0,
        ),
        min_iterations_before_stopping=10,
        item_label="images",
        item_label_singular="image",
        throughput_unit="img/s",
    ),
    "video": MediaBenchmarkPolicy(
        media="video",
        num_items=50,
        max_warmup_iterations=100,
        warmup_subset_size=3,
        slow_skip=SlowSkipPolicy(
            threshold_sec_per_item=2.0,
            preflight_items=3,
            max_preflight_secs=120.0,
        ),
        min_iterations_before_stopping=5,
        item_label="videos",
        item_label_singular="video",
        throughput_unit="vid/s",
    ),
}


def media_name(media: str | Any) -> MediaName:
    value = getattr(media, "value", media)
    if value not in MEDIA_POLICIES:
        msg = f"Unknown media policy {value!r}"
        raise ValueError(msg)
    return cast("MediaName", value)


def media_policy(media: str | Any) -> MediaBenchmarkPolicy:
    return MEDIA_POLICIES[media_name(media)]


def slow_skip_config(
    media: str | Any,
    *,
    threshold_sec_per_item: float | None = None,
    preflight_items: int | None = None,
) -> tuple[float, int, float]:
    policy = media_policy(media).slow_skip
    return (
        policy.threshold_sec_per_item if threshold_sec_per_item is None else threshold_sec_per_item,
        policy.preflight_items if preflight_items is None else preflight_items,
        policy.max_preflight_secs,
    )
