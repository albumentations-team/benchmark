"""Pillow (PIL) implementations for transforms Pillow directly supports.

The Pillow benchmark intentionally does not recreate Albumentations-style
augmentations with custom NumPy/ImageDraw glue. If Pillow has no direct analogue
for a transform, the transform is reported as unsupported for Pillow.
"""

from __future__ import annotations

import io
import random
from math import ceil, exp, sqrt
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from augbench.implementations.geometry import (
    aspect_size,
    centered_padding,
    emboss_kernel,
    sample_inverse_affine,
    sample_perspective_coefficients,
    x_shear_matrix,
)
from augbench.implementations.randaugment import SIGNED, RandAugmentSpec
from augbench.implementations.specs import odd_kernel_sizes, parameter_range

if TYPE_CHECKING:
    from collections.abc import Callable

    from augbench.implementations.specs import TransformSpec

LIBRARY = "pillow"


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    out = transform(image)
    if isinstance(out, Image.Image):
        out.load()
    return out


def _pil_interp(name: str) -> int:
    return Image.Resampling.BILINEAR if name == "bilinear" else Image.Resampling.NEAREST


def create_transform(spec: TransformSpec) -> Any | None:
    """Create a Pillow callable, or None when Pillow has no direct analogue."""

    builder = _TRANSFORM_BUILDERS.get(spec.name)
    return builder(spec.params) if builder is not None else None


def _build_resize(params: dict[str, Any]) -> Any | None:
    size = params["target_size"]
    return build_resize_transform((size, size), params["interpolation"], antialias=params.get("antialias", True))


def build_resize_transform(
    size: tuple[int, int], interpolation: str, *, antialias: bool
) -> Callable[[Image.Image], Image.Image]:
    resample = _pil_interp(interpolation)
    if antialias or resample != Image.Resampling.BILINEAR:
        return lambda image: image.resize(size, resample)

    width, height = size

    def resize_bilinear(image: Image.Image) -> Image.Image:
        return image.transform(
            size, Image.Transform.AFFINE, (image.width / width, 0, 0, 0, image.height / height, 0), resample=resample
        )

    return resize_bilinear


def _build_horizontal_flip(_params: dict[str, Any]) -> Any | None:
    return lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def _build_vertical_flip(_params: dict[str, Any]) -> Any | None:
    return lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def _build_transpose(_params: dict[str, Any]) -> Any | None:
    return lambda img: img.transpose(Image.Transpose.TRANSPOSE)


def _build_rotate(params: dict[str, Any]) -> Any | None:
    angle_lo, angle_hi = params["angle_range"]
    fill = params["fill"]
    interp = _pil_interp(params["interpolation"])
    return lambda img: img.rotate(-random.uniform(angle_lo, angle_hi), resample=interp, fillcolor=fill)


def _build_pad(params: dict[str, Any]) -> Any | None:
    return lambda img: ImageOps.expand(img, border=params["padding"], fill=params["fill"])


def _build_perspective(params: dict[str, Any]) -> Any:
    scale = tuple(params["scale"])
    fill = params["fill"]

    def warp(image: Image.Image) -> Image.Image:
        a, b, c, d, e, f, g, h = sample_perspective_coefficients(image.height, image.width, scale)
        return ImageOps.expand(image, border=1, fill=fill).transform(
            image.size,
            Image.Transform.PERSPECTIVE,
            (a + g, b + h, c + 1, d + g, e + h, f + 1, g, h),
            resample=Image.Resampling.BILINEAR,
            fillcolor=fill,
        )

    return warp


def _build_shear(params: dict[str, Any]) -> Any | None:
    angle_range = parameter_range(params["shear"])

    def shear(img: Image.Image) -> Image.Image:
        a, b, tx, d, e, ty = x_shear_matrix(-random.uniform(*angle_range), img.height / 2)
        # Padding supplies the zero-valued neighbors used by the other samplers.
        return ImageOps.expand(img, border=1, fill=0).transform(
            img.size,
            Image.Transform.AFFINE,
            (a, b, tx + 1, d, e, ty + 1),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    return shear


def _build_brightness(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["brightness_limit"])
    return lambda img: ImageEnhance.Brightness(img).enhance(1 + random.uniform(*bounds))


def _build_contrast(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["contrast_limit"])
    return lambda img: ImageEnhance.Contrast(img).enhance(1 + random.uniform(*bounds))


def _build_saturation(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["saturation_factor"])
    return lambda img: ImageEnhance.Color(img).enhance(1 + random.uniform(*bounds))


def _build_auto_contrast(_params: dict[str, Any]) -> Any | None:
    return ImageOps.autocontrast


def _build_equalize(_params: dict[str, Any]) -> Any | None:
    return ImageOps.equalize


def _build_grayscale(params: dict[str, Any]) -> Any | None:
    if params["num_output_channels"] == 1:
        return ImageOps.grayscale
    return lambda img: ImageOps.grayscale(img).convert("RGB")


def _build_invert(_params: dict[str, Any]) -> Any | None:
    return ImageOps.invert


def _build_posterize(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["bits"])
    return lambda img: ImageOps.posterize(img, random.randint(*bounds))


def _build_solarize(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["threshold"])
    return lambda img: ImageOps.solarize(img, int(random.uniform(*bounds) * 255))


def _build_gaussian_blur(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["sigma"])
    size = params["kernel_size"][0]
    radius = size // 2

    def blur(image: Image.Image) -> Image.Image:
        sigma = random.uniform(*bounds)
        weights = [exp(-0.5 * (offset / sigma) ** 2) for offset in range(-radius, radius + 1)]
        kernel = ImageFilter.Kernel((size, size), [y * x for y in weights for x in weights])
        return _filter_with_reflection(image, kernel, radius)

    return blur


def _filter_with_reflection(image: Image.Image, filter_: ImageFilter.Filter, radius: int) -> Image.Image:
    width, height = image.size
    horizontal = Image.new(image.mode, (width + 2 * radius, height))
    horizontal.paste(image, (radius, 0))
    horizontal.paste(ImageOps.mirror(image.crop((1, 0, 1 + radius, height))), (0, 0))
    horizontal.paste(ImageOps.mirror(image.crop((width - radius - 1, 0, width - 1, height))), (width + radius, 0))
    padded_width = horizontal.width
    padded = Image.new(image.mode, (padded_width, height + 2 * radius))
    padded.paste(horizontal, (0, radius))
    padded.paste(ImageOps.flip(horizontal.crop((0, 1, padded_width, 1 + radius))), (0, 0))
    bottom = horizontal.crop((0, height - radius - 1, padded_width, height - 1))
    padded.paste(ImageOps.flip(bottom), (0, height + radius))
    return padded.filter(filter_).crop((radius, radius, width + radius, height + radius))


def _build_median_blur(params: dict[str, Any]) -> Any:
    sizes = odd_kernel_sizes(params["blur_limit"])
    return lambda img: img.filter(ImageFilter.MedianFilter(size=random.choice(sizes)))


def _build_blur(params: dict[str, Any]) -> Any:
    sizes = odd_kernel_sizes(params["kernel_size"])

    def blur(image: Image.Image) -> Image.Image:
        radius = random.choice(sizes) // 2
        return _filter_with_reflection(image, ImageFilter.BoxBlur(radius), radius)

    return blur


def _build_center_crop(params: dict[str, Any]) -> Any:
    height, width = params["height"], params["width"]

    def crop(image: Image.Image) -> Image.Image:
        left = (image.width - width + int(image.width < width)) // 2
        top = (image.height - height + int(image.height < height)) // 2
        return image.crop((left, top, left + width, top + height))

    return crop


def _build_crop(params: dict[str, Any]) -> Any:
    left, top = params["left"], params["top"]
    box = (left, top, left + params["width"], top + params["height"])
    return lambda image: image.crop(box)


def _build_crop_and_pad(params: dict[str, Any]) -> Any:
    top, right, bottom, left = params["sides"]
    return lambda image: image.crop((-left, -top, image.width + right, image.height + bottom))


def _resize_without_antialias(image: Image.Image, height: int, width: int, interpolation: str) -> Image.Image:
    return build_resize_transform((width, height), interpolation, antialias=False)(image)


def _build_resize_by_side(params: dict[str, Any], *, cover: bool) -> Any:
    def resize(image: Image.Image) -> Image.Image:
        height, width = aspect_size(image.height, image.width, params["max_size"], cover=cover)
        return _resize_without_antialias(image, height, width, params["interpolation"])

    return resize


def _build_letter_box(params: dict[str, Any]) -> Any:
    def letter_box(image: Image.Image) -> Image.Image:
        height, width = aspect_size(image.height, image.width, params["height"])
        resized = _resize_without_antialias(image, height, width, "bilinear")
        return ImageOps.expand(resized, centered_padding(height, width, params["height"], params["width"]), fill=0)

    return letter_box


def _build_pad_if_needed(params: dict[str, Any]) -> Any:
    def pad(image: Image.Image) -> Image.Image:
        right = max(params["min_width"] - image.width, 0)
        bottom = max(params["min_height"] - image.height, 0)
        return ImageOps.expand(image, (0, 0, right, bottom), fill=0) if right or bottom else image

    return pad


def _build_channel_swap(params: dict[str, Any]) -> Any:
    order = tuple(params["channel_order"])

    def swap(image: Image.Image) -> Image.Image:
        channels = image.split()
        return Image.merge("RGB", tuple(channels[index] for index in order))

    return swap


def _build_channel_shuffle(_params: dict[str, Any]) -> Any:
    def shuffle(image: Image.Image) -> Image.Image:
        channels = list(image.split())
        random.shuffle(channels)
        return Image.merge("RGB", channels)

    return shuffle


def _build_channel_dropout(_params: dict[str, Any]) -> Any:
    def dropout(image: Image.Image) -> Image.Image:
        channels = list(image.split())
        channels[random.randrange(3)] = Image.new("L", image.size)
        return Image.merge("RGB", channels)

    return dropout


def _build_rgb_shift(params: dict[str, Any]) -> Any:
    limit = params["pixel_shift"]

    def shift(image: Image.Image) -> Image.Image:
        offsets = [random.uniform(-limit, limit) for _ in range(3)]
        table = [min(255, max(0, int(pixel + offset))) for offset in offsets for pixel in range(256)]
        return image.point(table)

    return shift


def _build_random_gamma(params: dict[str, Any]) -> Any:
    bounds = parameter_range(params["gamma"])
    levels = tuple(value / 255 for value in range(256))

    def gamma(image: Image.Image) -> Image.Image:
        exponent = random.uniform(*bounds) / 100
        table = [int(255 * level**exponent) for level in levels]
        return image.point(table * 3)

    return gamma


def _build_colorize(params: dict[str, Any]) -> Any:
    return lambda image: ImageOps.colorize(image, tuple(params["black"]), tuple(params["white"]))


def _build_dithering(_params: dict[str, Any]) -> Any:
    return lambda image: image.convert("L").convert("1", dither=Image.Dither.FLOYDSTEINBERG).convert("RGB")


def _build_random_scale(params: dict[str, Any]) -> Any:
    def resize(image: Image.Image) -> Image.Image:
        scale = random.uniform(*params["scale"])
        return _resize_without_antialias(image, int(image.height * scale), int(image.width * scale), "bilinear")

    return resize


def _build_random_rotate90(params: dict[str, Any]) -> Any:
    rotations = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)

    def rotate(image: Image.Image) -> Image.Image:
        rotation = rotations[random.randint(*params["times"])]
        return image if rotation is None else image.transpose(rotation)

    return rotate


def _build_random_sized_crop(params: dict[str, Any]) -> Any:
    size = params["crop_size"]
    height, width = params["size"]
    resize = build_resize_transform((width, height), "bilinear", antialias=False)

    def crop(image: Image.Image) -> Image.Image:
        left = random.randint(0, image.width - size)
        top = random.randint(0, image.height - size)
        return resize(image.crop((left, top, left + size, top + size)))

    return crop


def _build_coarse_dropout(params: dict[str, Any]) -> Any:
    size = params["hole_size"]

    def erase(image: Image.Image) -> Image.Image:
        left = random.randint(0, image.width - size)
        top = random.randint(0, image.height - size)
        image.paste(0, (left, top, left + size, top + size))
        return image

    return erase


def _build_emboss(params: dict[str, Any]) -> Any:
    alpha = parameter_range(params["alpha"])
    strength = parameter_range(params["strength"])

    def emboss(image: Image.Image) -> Image.Image:
        kernel = emboss_kernel(random.uniform(*alpha), random.uniform(*strength))
        return image.filter(ImageFilter.Kernel((3, 3), kernel[6:] + kernel[3:6] + kernel[:3], scale=1))

    return emboss


def _build_affine(params: dict[str, Any]) -> Any:
    def warp(image: Image.Image) -> Image.Image:
        center = (image.width / 2, image.height / 2)
        a, b, tx, d, e, ty = sample_inverse_affine(center, params)
        return ImageOps.expand(image, border=1, fill=0).transform(
            image.size,
            Image.Transform.AFFINE,
            (a, b, tx + 1, d, e, ty + 1),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    return warp


def _build_unsharp_mask(params: dict[str, Any]) -> Any | None:
    sizes = odd_kernel_sizes(params["blur_limit"])
    alpha = parameter_range(params["alpha"])
    # Match the 3-tap step response. For 5 and 7 taps, use OpenCV's sigma=0 kernel deviations.
    radii = {3: 0.85, 5: 1.0, 7: sqrt(1.875)}

    def _unsharp_mask(img: Image.Image) -> Image.Image:
        radius = radii[random.choice(sizes)]
        percent = round(random.uniform(*alpha) * 100)
        filter_ = ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=params["threshold"])
        return _filter_with_reflection(img, filter_, radius=5)

    return _unsharp_mask


def _build_enhance(params: dict[str, Any]) -> Any:
    preset = ImageFilter.EDGE_ENHANCE if params["mode"] == "edge" else ImageFilter.DETAIL
    size, scale, offset, weights = preset.filterargs
    center = len(weights) // 2

    def enhance(image: Image.Image) -> Image.Image:
        alpha = random.uniform(*params["alpha_range"])
        blended = [alpha * weight for weight in weights]
        blended[center] += (1 - alpha) * scale
        kernel = ImageFilter.Kernel(size, blended, scale=scale, offset=offset)
        return _filter_with_reflection(image, kernel, radius=1)

    return enhance


def _build_jpeg_compression(params: dict[str, Any]) -> Any | None:
    low, high = params["quality_range"]

    def _jpeg(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        quality = random.randint(low, high)
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    return _jpeg


class _RandAugment:
    """RandAugment from Pillow primitives: each call draws its own operations, with replacement."""

    def __init__(self, spec: RandAugmentSpec) -> None:
        self._spec = spec
        self._fill = (int(spec.fill),) * 3
        self._operations = {
            "Identity": lambda image, _sign: image,
            "ShearX": lambda image, sign: self._affine(image, (1, sign * spec.shear, 0, 0, 1, 0)),
            "ShearY": lambda image, sign: self._affine(image, (1, 0, 0, sign * spec.shear, 1, 0)),
            "TranslateX": lambda image, sign: self._affine(
                image, (1, 0, -sign * int(spec.translate_fraction * image.width), 0, 1, 0)
            ),
            "TranslateY": lambda image, sign: self._affine(
                image, (1, 0, 0, 0, 1, -sign * int(spec.translate_fraction * image.height))
            ),
            "Rotate": lambda image, sign: image.rotate(
                sign * spec.rotate_degrees, resample=Image.Resampling.BILINEAR, fillcolor=self._fill
            ),
            "Brightness": lambda image, sign: ImageEnhance.Brightness(image).enhance(1 + sign * spec.enhance),
            "Color": lambda image, sign: ImageEnhance.Color(image).enhance(1 + sign * spec.enhance),
            "Contrast": lambda image, sign: ImageEnhance.Contrast(image).enhance(1 + sign * spec.enhance),
            "Sharpness": lambda image, sign: ImageEnhance.Sharpness(image).enhance(1 + sign * spec.enhance),
            "Posterize": lambda image, _sign: ImageOps.posterize(image, spec.posterize_bits),
            "Solarize": lambda image, _sign: ImageOps.solarize(image, threshold=ceil(255 * spec.solarize_fraction)),
            "AutoContrast": lambda image, _sign: ImageOps.autocontrast(image),
            "Equalize": lambda image, _sign: ImageOps.equalize(image),
        }
        self._names = tuple(self._operations)

    def _affine(self, image: Image.Image, coefficients: tuple[float, ...]) -> Image.Image:
        return image.transform(
            image.size, Image.Transform.AFFINE, coefficients, resample=Image.Resampling.BILINEAR, fillcolor=self._fill
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        for _ in range(self._spec.num_ops):
            name = random.choice(self._names)
            sign = random.choice((-1.0, 1.0)) if name in SIGNED else 1.0
            image = self._operations[name](image, sign)
        return image


def _build_rand_augment(params: dict[str, Any]) -> Any:
    return _RandAugment(RandAugmentSpec.from_params(params))


_TRANSFORM_BUILDERS = {
    "RandAugment": _build_rand_augment,
    "Resize": _build_resize,
    "CenterCrop": _build_center_crop,
    "Crop": _build_crop,
    "CropAndPad": _build_crop_and_pad,
    "LetterBox": _build_letter_box,
    "PadIfNeeded": _build_pad_if_needed,
    "ChannelSwap": _build_channel_swap,
    "ChannelShuffle": _build_channel_shuffle,
    "ChannelDropout": _build_channel_dropout,
    "RGBShift": _build_rgb_shift,
    "LongestMaxSize": lambda params: _build_resize_by_side(params, cover=False),
    "SmallestMaxSize": lambda params: _build_resize_by_side(params, cover=True),
    "RandomGamma": _build_random_gamma,
    "Colorize": _build_colorize,
    "Dithering": _build_dithering,
    "RandomScale": _build_random_scale,
    "RandomSizedCrop": _build_random_sized_crop,
    "CoarseDropout": _build_coarse_dropout,
    "Emboss": _build_emboss,
    "ShiftScaleRotate": _build_affine,
    "RandomRotate90": _build_random_rotate90,
    "HorizontalFlip": _build_horizontal_flip,
    "VerticalFlip": _build_vertical_flip,
    "Transpose": _build_transpose,
    "Rotate": _build_rotate,
    "Pad": _build_pad,
    "Affine": _build_affine,
    "Perspective": _build_perspective,
    "Shear": _build_shear,
    "Brightness": _build_brightness,
    "Contrast": _build_contrast,
    "Saturation": _build_saturation,
    "AutoContrast": _build_auto_contrast,
    "Equalize": _build_equalize,
    "Grayscale": _build_grayscale,
    "Invert": _build_invert,
    "Posterize": _build_posterize,
    "Solarize": _build_solarize,
    "GaussianBlur": _build_gaussian_blur,
    "MedianBlur": _build_median_blur,
    "Blur": _build_blur,
    "UnsharpMask": _build_unsharp_mask,
    "EnhanceEdge": _build_enhance,
    "EnhanceDetail": _build_enhance,
    "JpegCompression": _build_jpeg_compression,
}
