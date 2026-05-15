from __future__ import annotations

from benchmark.transforms.kornia_unstable import KORNIA_BENCHMARK_EXCLUDED_NAMES
from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

_CROP_RECIPE_NAMES = {"RandomCrop224", "RandomResizedCrop"}
_RECIPE_EXCLUDED_NAMES = {"Normalize"}
_MIN_RECIPE_LIBRARY_SUPPORT = 2

_ALBUMENTATIONSX_VIDEO_PIPELINE_EXCLUDED_NAMES = {
    "Colorize",
    "ConstrainedCoarseDropout",
    "Normalize",
}

_TORCHVISION_VIDEO_PIPELINE_SUPPORTED_NAMES = {
    "Affine",
    "AutoContrast",
    "Brightness",
    "ChannelShuffle",
    "ColorJiggle",
    "ColorJitter",
    "Contrast",
    "Elastic",
    "Equalize",
    "Erasing",
    "GaussianBlur",
    "Grayscale",
    "HorizontalFlip",
    "Invert",
    "JpegCompression",
    "Pad",
    "Perspective",
    "Posterize",
    "RandomCrop224",
    "RandomResizedCrop",
    "Resize",
    "Rotate",
    "Sharpen",
    "Solarize",
    "VerticalFlip",
}

_KORNIA_VIDEO_PIPELINE_SUPPORTED_NAMES: set[str] = {
    "Affine",
    "AutoContrast",
    "Blur",
    "Brightness",
    "CLAHE",
    "ChannelDropout",
    "ChannelShuffle",
    "ColorJiggle",
    "ColorJitter",
    "Contrast",
    "CornerIllumination",
    "Elastic",
    "Equalize",
    "Erasing",
    "GaussianBlur",
    "GaussianIllumination",
    "GaussianNoise",
    "Grayscale",
    "HorizontalFlip",
    "Hue",
    "Invert",
    "JpegCompression",
    "LinearIllumination",
    "MedianBlur",
    "MotionBlur",
    "OpticalDistortion",
    "PlankianJitter",
    "PlasmaBrightness",
    "PlasmaContrast",
    "PlasmaShadow",
    "RGBShift",
    "Rain",
    "RandomCrop224",
    "RandomGamma",
    "RandomJigsaw",
    "RandomResizedCrop",
    "RandomRotate90",
    "Resize",
    "Rotate",
    "SaltAndPepper",
    "Saturation",
    "Sharpen",
    "Shear",
    "Snow",
    "Solarize",
    "ThinPlateSpline",
    "VerticalFlip",
}
_KORNIA_VIDEO_PIPELINE_SUPPORTED_NAMES.difference_update(KORNIA_BENCHMARK_EXCLUDED_NAMES)

_VIDEO_PIPELINE_SUPPORT_BY_LIBRARY = {
    "albumentationsx": {spec.name for spec in TRANSFORM_SPECS} - _ALBUMENTATIONSX_VIDEO_PIPELINE_EXCLUDED_NAMES,
    "torchvision": _TORCHVISION_VIDEO_PIPELINE_SUPPORTED_NAMES,
    "kornia": _KORNIA_VIDEO_PIPELINE_SUPPORTED_NAMES,
}


def repeated_stats() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return NORMALIZE_MEAN, NORMALIZE_STD


def spec_by_name(name: str) -> TransformSpec:
    for spec in TRANSFORM_SPECS:
        if spec.name == name:
            return spec
    msg = f"Unknown transform spec {name!r}"
    raise ValueError(msg)


def is_crop_recipe_spec(spec: TransformSpec) -> bool:
    return spec.name in _CROP_RECIPE_NAMES


def recipe_name(spec: TransformSpec) -> str:
    prefix = spec.name if is_crop_recipe_spec(spec) else f"RandomCrop224+{spec.name}"
    return f"{prefix}+Normalize+ToTensor"


def _support_sets() -> tuple[set[str], ...]:
    return tuple(_VIDEO_PIPELINE_SUPPORT_BY_LIBRARY.values())


def is_supported_by_library(spec: TransformSpec, library: str) -> bool:
    return spec.name in _VIDEO_PIPELINE_SUPPORT_BY_LIBRARY.get(library, set())


def _supported_by_enough_libraries(spec: TransformSpec, support_sets: tuple[set[str], ...]) -> bool:
    return sum(spec.name in supported_names for supported_names in support_sets) >= _MIN_RECIPE_LIBRARY_SUPPORT


def recipe_augmentation_specs() -> list[TransformSpec]:
    support_sets = _support_sets()
    return [
        spec
        for spec in TRANSFORM_SPECS
        if spec.name not in _RECIPE_EXCLUDED_NAMES and _supported_by_enough_libraries(spec, support_sets)
    ]
