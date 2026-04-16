"""Compatibility helpers for Albumentations MIT across spec files.

Keeping these in one place means a single fix propagates to image, video,
and multichannel specs automatically.
"""

from __future__ import annotations

from typing import Any


class ConstrainedCoarseDropoutWrapper:
    """Injects a dummy bbox so MIT 2.0.8 ConstrainedCoarseDropout actually runs.

    MIT 2.0.8 silently skips the transform and emits a warning when no bboxes
    or mask is provided. Wrapping at construction time keeps ``__call__`` clean.
    Works for both single-image (``image=``) and batch (``images=``) call patterns.
    """

    def __init__(self, transform: Any) -> None:
        self._inner = transform

    def __call__(self, **data: Any) -> dict[str, Any]:
        data = dict(data)
        if "images" in data:
            n = len(data["images"])
            data.setdefault("bboxes", [[(0.25, 0.25, 0.5, 0.5)] for _ in range(n)])
        elif "image" in data:
            data.setdefault("bboxes", [(0.25, 0.25, 0.5, 0.5)])
        return self._inner(**data)
