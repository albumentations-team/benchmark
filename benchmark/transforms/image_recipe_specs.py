from __future__ import annotations

from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

_CROP_RECIPE_NAMES = {"RandomCrop224", "RandomResizedCrop", "CenterCrop224"}
_RECIPE_EXCLUDED_NAMES = {"Normalize"}
_MIN_RECIPE_LIBRARY_SUPPORT = 2

_ALBUMENTATIONSX_PIPELINE_EXCLUDED_NAMES = {
    "Colorize",
    "ConstrainedCoarseDropout",
    "Normalize",
}

_ALBUMENTATIONSX_MULTICHANNEL_EXCLUDED_NAMES = _ALBUMENTATIONSX_PIPELINE_EXCLUDED_NAMES | {
    "CLAHE",
    "ColorJiggle",
    "ColorJitter",
    "Equalize",
    "Hue",
    "PhotoMetricDistort",
    "PlankianJitter",
    "RGBShift",
    "Rain",
    "Saturation",
    "Snow",
}

_TORCHVISION_PIPELINE_SUPPORTED_NAMES = {
    "Affine",
    "AutoContrast",
    "Brightness",
    "CenterCrop224",
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
    "PhotoMetricDistort",
    "Posterize",
    "RandomCrop224",
    "RandomResizedCrop",
    "Resize",
    "Rotate",
    "Sharpen",
    "Solarize",
    "VerticalFlip",
}

_KORNIA_PIPELINE_SUPPORTED_NAMES = {
    "Affine",
    "AutoContrast",
    "Blur",
    "Brightness",
    "CLAHE",
    "CenterCrop224",
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
    "LongestMaxSize",
    "MedianBlur",
    "MotionBlur",
    "OpticalDistortion",
    "Perspective",
    "PlankianJitter",
    "PlasmaBrightness",
    "PlasmaContrast",
    "PlasmaShadow",
    "Posterize",
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
    "SmallestMaxSize",
    "Snow",
    "Solarize",
    "ThinPlateSpline",
    "VerticalFlip",
}

_KORNIA_MULTICHANNEL_EXCLUDED_NAMES = {
    "CLAHE",
    "ColorJiggle",
    "ColorJitter",
    "Equalize",
    "Hue",
    "PlankianJitter",
    "RGBShift",
    "Rain",
    "SaltAndPepper",
    "Saturation",
    "Snow",
}

_PILLOW_PIPELINE_SUPPORTED_NAMES = {
    "Affine",
    "AutoContrast",
    "Blur",
    "Brightness",
    "CenterCrop224",
    "Contrast",
    "Dithering",
    "EnhanceDetail",
    "EnhanceEdge",
    "Equalize",
    "GaussianBlur",
    "Grayscale",
    "HorizontalFlip",
    "Invert",
    "JpegCompression",
    "MedianBlur",
    "Pad",
    "Posterize",
    "RandomCrop224",
    "RandomResizedCrop",
    "Resize",
    "Rotate",
    "Saturation",
    "Shear",
    "Solarize",
    "Transpose",
    "UnsharpMask",
    "VerticalFlip",
}


def repeated_stats(num_channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    repeats = num_channels // 3
    return NORMALIZE_MEAN * repeats, NORMALIZE_STD * repeats


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


def _support_sets(num_channels: int) -> tuple[set[str], ...]:
    all_names = {spec.name for spec in TRANSFORM_SPECS}
    albumentationsx = all_names - _ALBUMENTATIONSX_PIPELINE_EXCLUDED_NAMES
    if num_channels == 3:
        return (
            albumentationsx,
            _TORCHVISION_PIPELINE_SUPPORTED_NAMES,
            _KORNIA_PIPELINE_SUPPORTED_NAMES,
            _PILLOW_PIPELINE_SUPPORTED_NAMES,
        )
    return (
        albumentationsx - _ALBUMENTATIONSX_MULTICHANNEL_EXCLUDED_NAMES,
        _TORCHVISION_PIPELINE_SUPPORTED_NAMES,
        _KORNIA_PIPELINE_SUPPORTED_NAMES - _KORNIA_MULTICHANNEL_EXCLUDED_NAMES,
    )


def _supported_by_enough_libraries(spec: TransformSpec, support_sets: tuple[set[str], ...]) -> bool:
    return sum(spec.name in supported_names for supported_names in support_sets) >= _MIN_RECIPE_LIBRARY_SUPPORT


def recipe_augmentation_specs(num_channels: int) -> list[TransformSpec]:
    support_sets = _support_sets(num_channels)
    return [
        spec
        for spec in TRANSFORM_SPECS
        if spec.name not in _RECIPE_EXCLUDED_NAMES and _supported_by_enough_libraries(spec, support_sets)
    ]
