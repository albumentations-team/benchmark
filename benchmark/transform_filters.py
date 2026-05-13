from __future__ import annotations

from typing import Any

KORNIA_GPU_IMAGE_EXCLUDED_NAMES = frozenset({"Shear"})
KORNIA_GPU_IMAGE_EXCLUDED_RECIPES = frozenset({"RandomCrop224+Shear+Normalize+ToTensor"})
KORNIA_GPU_9CH_IMAGE_EXCLUDED_NAMES = frozenset({"MedianBlur"})
KORNIA_GPU_9CH_IMAGE_EXCLUDED_RECIPES = frozenset(
    {
        "RandomCrop224+MedianBlur+Normalize+ToTensor",
    },
)
TORCHVISION_GPU_IMAGE_EXCLUDED_NAMES = frozenset({"JpegCompression"})
TORCHVISION_GPU_IMAGE_EXCLUDED_RECIPES = frozenset({"RandomCrop224+JpegCompression+Normalize+ToTensor"})
KORNIA_CPU_VIDEO_MICRO_EXCLUDED_NAMES = frozenset({"Elastic", "Rotate"})


def filter_transforms_for_library_device(
    transforms: tuple[str, ...],
    *,
    scenario: str | None = None,
    mode: str | None = None,
    library: str,
    media: str,
    device: str,
) -> tuple[str, ...]:
    """Apply library/device-specific exclusions after global transform-set expansion."""
    if not transforms:
        return transforms
    if scenario == "video-16f" and mode == "micro" and library == "kornia" and device == "none":
        return tuple(transform for transform in transforms if transform not in KORNIA_CPU_VIDEO_MICRO_EXCLUDED_NAMES)
    if media != "image" or device == "none":
        return transforms

    if library == "kornia":
        excluded = KORNIA_GPU_IMAGE_EXCLUDED_NAMES | KORNIA_GPU_IMAGE_EXCLUDED_RECIPES
        if scenario == "image-9ch":
            excluded = excluded | KORNIA_GPU_9CH_IMAGE_EXCLUDED_NAMES | KORNIA_GPU_9CH_IMAGE_EXCLUDED_RECIPES
    elif library == "torchvision":
        excluded = TORCHVISION_GPU_IMAGE_EXCLUDED_NAMES | TORCHVISION_GPU_IMAGE_EXCLUDED_RECIPES
    else:
        return transforms
    return tuple(transform for transform in transforms if transform not in excluded)


def filter_transform_dicts_for_library_device(
    transforms: list[dict[str, Any]],
    *,
    scenario: str | None = None,
    mode: str | None = None,
    library: str,
    media: str,
    device: str,
) -> list[dict[str, Any]]:
    names = tuple(str(transform["name"]) for transform in transforms)
    filtered_names = set(
        filter_transforms_for_library_device(
            names,
            scenario=scenario,
            mode=mode,
            library=library,
            media=media,
            device=device,
        ),
    )
    if len(filtered_names) == len(names):
        return transforms
    return [transform for transform in transforms if str(transform["name"]) in filtered_names]
