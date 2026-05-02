from __future__ import annotations

KORNIA_GPU_IMAGE_EXCLUDED_NAMES = frozenset({"Shear"})
KORNIA_GPU_IMAGE_EXCLUDED_RECIPES = frozenset({"RandomCrop224+Shear+Normalize+ToTensor"})


def filter_transforms_for_library_device(
    transforms: tuple[str, ...],
    *,
    library: str,
    media: str,
    device: str,
) -> tuple[str, ...]:
    """Apply library/device-specific exclusions after global transform-set expansion."""
    if not transforms:
        return transforms
    if media != "image" or library != "kornia" or device == "none":
        return transforms

    excluded = KORNIA_GPU_IMAGE_EXCLUDED_NAMES | KORNIA_GPU_IMAGE_EXCLUDED_RECIPES
    return tuple(transform for transform in transforms if transform not in excluded)
