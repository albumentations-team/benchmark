"""Compatibility helpers shared by Albumentations-based benchmark specs."""

from __future__ import annotations

from typing import Any

import numpy as np


class ConstrainedCoarseDropoutWrapper:
    """Injects a dummy mask so ConstrainedCoarseDropout actually runs.

    Albumentations skips this transform and emits a warning when no bboxes or
    mask are provided. The benchmark specs only pass image data, so wrapping at
    construction time keeps the hot ``__call__`` path unchanged.
    """

    def __init__(self, transform: Any) -> None:
        self._inner = transform
        if getattr(transform, "mask_indices", None) is None and getattr(transform, "bbox_labels", None) is None:
            transform.mask_indices = [1]

    def __call__(self, **data: Any) -> dict[str, Any]:
        data = dict(data)
        if "images" in data:
            data.setdefault("mask", np.ones((*data["images"].shape[1:3], 1), dtype=np.uint8))
        elif "image" in data:
            data.setdefault("mask", np.ones((*data["image"].shape[:2], 1), dtype=np.uint8))
        return self._inner(**data)
