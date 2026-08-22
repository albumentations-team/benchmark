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

if TYPE_CHECKING:
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

    # Geometry: direct Image methods only. Random crop/resize composites are unsupported.
    builder = _TRANSFORM_BUILDERS.get(spec.name)
    return builder(spec.params) if builder is not None else None


def _build_resize(params: dict[str, Any]) -> Any | None:
    size = params["target_size"]
    return lambda img: img.resize((size, size), _pil_interp(params["interpolation"]))


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
    import math

    shear = math.tan(math.radians(params["shear"]))
    return lambda img: img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )


# Color and point operations with direct ImageOps/ImageEnhance equivalents.


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


# Direct ImageFilter equivalents.


def _build_gaussian_blur(params: dict[str, Any]) -> Any | None:
    return lambda img: img.filter(ImageFilter.GaussianBlur(radius=params["sigma"]))


def _build_median_blur(params: dict[str, Any]) -> Any | None:
    size = params["blur_limit"]
    size = size if size % 2 == 1 else size + 1
    return lambda img: img.filter(ImageFilter.MedianFilter(size=size))


def _build_blur(params: dict[str, Any]) -> Any | None:
    return lambda img: img.filter(ImageFilter.BoxBlur(radius=params["radius"]))


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
