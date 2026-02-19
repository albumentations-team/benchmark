"""Central transform registry.

Each entry in TRANSFORMS maps a canonical transform name to per-library factory
functions for both image and video modes. Libraries that don't support a given
transform return None from their factory, and those entries are automatically
excluded from that library's TRANSFORMS list.

Adding a new transform:
    Add one TransformDef entry here — no other files need changing.

Adding a new library:
    1. Add a key to TransformDef.image_impls / video_impls for each transform it supports.
    2. Create benchmark/transforms/{library}_impl.py defining LIBRARY, __call__, and TRANSFORMS
       (built by calling build_transforms(LIBRARY, MediaType.IMAGE) from this module).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.transforms.specs import TRANSFORM_SPECS


@dataclass
class TransformDef:
    """Canonical definition of a single benchmark transform.

    image_impls and video_impls map library names to zero-argument callables
    that construct the transform object. Return None to signal "not supported".
    """

    name: str
    params: dict[str, Any]
    image_impls: dict[str, Any] = field(default_factory=dict)
    video_impls: dict[str, Any] = field(default_factory=dict)


def _build_defs() -> list[TransformDef]:
    """Lazily construct TransformDef list from existing create_transform functions.

    This bridges the old per-library factory pattern to the new registry without
    requiring every library to be imported at registry-import time (they live in
    separate venvs).  Each library's _impl.py registers itself by calling
    register_library() below.
    """
    return [TransformDef(name=spec.name, params=spec.params) for spec in TRANSFORM_SPECS]


# Global registry — populated by register_library() calls from *_impl.py files
TRANSFORMS: list[TransformDef] = _build_defs()

# Maps spec.name → TransformDef for O(1) lookup
_BY_NAME: dict[str, TransformDef] = {t.name: t for t in TRANSFORMS}


def register_library(
    library: str,
    create_image_fn: Any | None = None,
    create_video_fn: Any | None = None,
) -> None:
    """Register per-library factory functions against the global registry.

    Called from each *_impl.py at module load time.

    Args:
        library: Library name string (e.g. "albumentationsx")
        create_image_fn: A callable(spec) -> transform | None for images.
        create_video_fn: A callable(spec) -> transform | None for videos.
    """
    for spec in TRANSFORM_SPECS:
        td = _BY_NAME.get(spec.name)
        if td is None:
            continue
        if create_image_fn is not None:
            try:
                td.image_impls[library] = create_image_fn(spec)
            except Exception:
                td.image_impls[library] = None
        if create_video_fn is not None:
            try:
                td.video_impls[library] = create_video_fn(spec)
            except Exception:
                td.video_impls[library] = None


def build_transforms(library: str, media: str = "image") -> list[dict[str, Any]]:
    """Return the list of {"name": ..., "transform": ...} dicts for a library.

    Entries where the factory returned None are excluded (transform not supported).

    Args:
        library: Library name (e.g. "albumentationsx")
        media: "image" or "video"
    """
    result: list[dict[str, Any]] = []
    for td in TRANSFORMS:
        impls = td.image_impls if media == "image" else td.video_impls
        transform = impls.get(library)
        if transform is not None:
            result.append({"name": td.name, "transform": transform})
    return result


__all__ = ["TRANSFORMS", "TransformDef", "build_transforms", "register_library"]
