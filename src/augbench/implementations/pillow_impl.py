"""Pillow (PIL) implementations for transforms Pillow directly supports.

The Pillow benchmark intentionally does not recreate Albumentations-style
augmentations with custom NumPy/ImageDraw glue. If Pillow has no direct analogue
for a transform, the transform is reported as unsupported for Pillow.
"""

from __future__ import annotations

import io
import random
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from augbench.implementations.geometry import aspect_size, centered_inverse_affine, centered_padding, emboss_kernel
from augbench.implementations.shear import x_shear_matrix

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


def _affine_coeffs(angle_deg: float, tx: float, ty: float, scale: float, shear_deg: float) -> tuple[float, ...]:
    """Build PIL AFFINE inverse-mapping coefficients (input = M * output)."""
    import math

    angle = math.radians(angle_deg)
    shear = math.tan(math.radians(shear_deg))
    cos_angle, sin_angle = math.cos(angle) / scale, math.sin(angle) / scale
    return (
        cos_angle,
        sin_angle + shear * cos_angle,
        -tx / scale,
        -sin_angle,
        cos_angle + shear * sin_angle,
        -ty / scale,
    )


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


def _build_affine(params: dict[str, Any]) -> Any | None:
    tx, ty = params["shift"]
    shear = params["shear"][0] if isinstance(params["shear"], tuple | list) else params["shear"]
    coeffs = _affine_coeffs(params["angle"], tx, ty, params["scale"], shear)
    return lambda img: img.transform(
        img.size,
        Image.Transform.AFFINE,
        coeffs,
        resample=_pil_interp(params["interpolation"]),
        fillcolor=params["fill"],
    )


def _build_shear(params: dict[str, Any]) -> Any | None:
    angle = float(params["shear"])

    def shear(img: Image.Image) -> Image.Image:
        a, b, tx, d, e, ty = x_shear_matrix(-angle, img.height / 2)
        # Padding supplies the zero-valued neighbors used by the other samplers.
        return ImageOps.expand(img, border=1, fill=0).transform(
            img.size,
            Image.Transform.AFFINE,
            (a, b, tx + 1, d, e, ty + 1),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    return shear


def _build_brightness(params: dict[str, Any]) -> Any | None:
    limit = params["brightness_limit"]
    factor = 1.0 + float(limit[0] if isinstance(limit, (list, tuple)) else limit)
    return lambda img: ImageEnhance.Brightness(img).enhance(factor)


def _build_contrast(params: dict[str, Any]) -> Any | None:
    limit = params["contrast_limit"]
    factor = 1.0 + float(limit[0] if isinstance(limit, (list, tuple)) else limit)
    return lambda img: ImageEnhance.Contrast(img).enhance(factor)


def _build_saturation(params: dict[str, Any]) -> Any | None:
    return lambda img: ImageEnhance.Color(img).enhance(1.0 + params["saturation_factor"])


def _build_auto_contrast(_params: dict[str, Any]) -> Any | None:
    return ImageOps.autocontrast


def _build_equalize(_params: dict[str, Any]) -> Any | None:
    return ImageOps.equalize


def _build_grayscale(_params: dict[str, Any]) -> Any | None:
    return lambda img: ImageOps.grayscale(img).convert("RGB")


def _build_invert(_params: dict[str, Any]) -> Any | None:
    return ImageOps.invert


def _build_posterize(params: dict[str, Any]) -> Any | None:
    return lambda img: ImageOps.posterize(img, int(params["bits"]))


def _build_solarize(params: dict[str, Any]) -> Any | None:
    return lambda img: ImageOps.solarize(img, int(params["threshold"] * 255))


def _build_gaussian_blur(params: dict[str, Any]) -> Any | None:
    return lambda img: img.filter(ImageFilter.GaussianBlur(radius=params["sigma"]))


def _build_median_blur(params: dict[str, Any]) -> Any | None:
    size = params["blur_limit"]
    size = size if size % 2 == 1 else size + 1
    return lambda img: img.filter(ImageFilter.MedianFilter(size=size))


def _build_blur(params: dict[str, Any]) -> Any | None:
    blur = ImageFilter.BoxBlur(radius=(params["kernel_size"] - 1) / 2)
    return lambda img: img.filter(blur)


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


def _build_channel_swap(_params: dict[str, Any]) -> Any:
    def swap(image: Image.Image) -> Image.Image:
        red, green, blue = image.split()
        return Image.merge("RGB", (blue, green, red))

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
    gamma = params["gamma"] / 100
    table = [int(255 * (value / 255) ** gamma) for value in range(256)] * 3
    return lambda image: image.point(table)


def _build_colorize(params: dict[str, Any]) -> Any:
    return lambda image: ImageOps.colorize(image.convert("L"), tuple(params["black"]), tuple(params["white"]))


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
    kernel = emboss_kernel(params["alpha"], params["strength"])
    pillow_kernel = ImageFilter.Kernel((3, 3), kernel[6:] + kernel[3:6] + kernel[:3], scale=1)
    return lambda image: image.filter(pillow_kernel)


def _build_shift_scale_rotate(params: dict[str, Any]) -> Any:
    center = params["input_size"] / 2
    a, b, tx, d, e, ty = centered_inverse_affine(
        (center, center), params["angle"], params["scale"], tuple(params["shift"])
    )

    def warp(image: Image.Image) -> Image.Image:
        return ImageOps.expand(image, border=1, fill=0).transform(
            image.size,
            Image.Transform.AFFINE,
            (a, b, tx + 1, d, e, ty + 1),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    return warp


def _build_unsharp_mask(params: dict[str, Any]) -> Any | None:
    blur_lo, blur_hi = params["blur_limit"]
    alpha_lo, alpha_hi = params["alpha"]

    def _unsharp_mask(img: Image.Image) -> Image.Image:
        radius = random.uniform(blur_lo / 2.0, blur_hi / 2.0)
        percent = int(100 + random.uniform(alpha_lo, alpha_hi) * 200)
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=params["threshold"]))

    return _unsharp_mask


def _build_enhance_edge(_params: dict[str, Any]) -> Any | None:
    return lambda img: img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def _build_enhance_detail(_params: dict[str, Any]) -> Any | None:
    return lambda img: img.filter(ImageFilter.DETAIL)


def _build_jpeg_compression(params: dict[str, Any]) -> Any | None:
    quality = params["quality"]

    def _jpeg(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    return _jpeg


_TRANSFORM_BUILDERS = {
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
    "ShiftScaleRotate": _build_shift_scale_rotate,
    "RandomRotate90": _build_random_rotate90,
    "HorizontalFlip": _build_horizontal_flip,
    "VerticalFlip": _build_vertical_flip,
    "Transpose": _build_transpose,
    "Rotate": _build_rotate,
    "Pad": _build_pad,
    "Affine": _build_affine,
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
    "EnhanceEdge": _build_enhance_edge,
    "EnhanceDetail": _build_enhance_detail,
    "JpegCompression": _build_jpeg_compression,
}
