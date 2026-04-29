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

from benchmark.transforms.registry import build_transforms, register_library

if TYPE_CHECKING:
    from benchmark.transforms.specs import TransformSpec

LIBRARY = "pillow"


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    out = transform(image)
    if isinstance(out, Image.Image):
        out.load()
    return out


def _pil_interp(name: str) -> int:
    return Image.BILINEAR if name == "bilinear" else Image.NEAREST


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
    params = spec.params

    # Geometry: direct Image methods only. Random crop/resize composites are unsupported.
    if spec.name == "Resize":
        size = params["target_size"]
        return lambda img: img.resize((size, size), _pil_interp(params["interpolation"]))

    if spec.name == "HorizontalFlip":
        return lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)

    if spec.name == "VerticalFlip":
        return lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)

    if spec.name == "Transpose":
        return lambda img: img.transpose(Image.TRANSPOSE)

    if spec.name == "Rotate":
        angle_lo, angle_hi = params["angle_range"]
        fill = params["fill"]
        interp = _pil_interp(params["interpolation"])
        return lambda img: img.rotate(-random.uniform(angle_lo, angle_hi), resample=interp, fillcolor=fill)

    if spec.name == "Pad":
        return lambda img: ImageOps.expand(img, border=params["padding"], fill=params["fill"])

    if spec.name == "Affine":
        tx, ty = params["shift"]
        shear = params["shear"][0] if isinstance(params["shear"], tuple | list) else params["shear"]
        coeffs = _affine_coeffs(params["angle"], tx, ty, params["scale"], shear)
        return lambda img: img.transform(
            img.size,
            Image.AFFINE,
            coeffs,
            resample=_pil_interp(params["interpolation"]),
            fillcolor=params["fill"],
        )

    if spec.name == "Shear":
        import math

        shear = math.tan(math.radians(params["shear"]))
        return lambda img: img.transform(
            img.size,
            Image.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.BILINEAR,
            fillcolor=0,
        )

    # Color and point operations with direct ImageOps/ImageEnhance equivalents.
    if spec.name == "Brightness":
        limit = params["brightness_limit"]
        factor = 1.0 + float(limit[0] if isinstance(limit, (list, tuple)) else limit)
        return lambda img: ImageEnhance.Brightness(img).enhance(factor)

    if spec.name == "Contrast":
        limit = params["contrast_limit"]
        factor = 1.0 + float(limit[0] if isinstance(limit, (list, tuple)) else limit)
        return lambda img: ImageEnhance.Contrast(img).enhance(factor)

    if spec.name == "Saturation":
        return lambda img: ImageEnhance.Color(img).enhance(1.0 + params["saturation_factor"])

    if spec.name == "AutoContrast":
        return ImageOps.autocontrast

    if spec.name == "Equalize":
        return ImageOps.equalize

    if spec.name == "Grayscale":
        return lambda img: ImageOps.grayscale(img).convert("RGB")

    if spec.name == "Invert":
        return ImageOps.invert

    if spec.name == "Posterize":
        return lambda img: ImageOps.posterize(img, int(params["bits"]))

    if spec.name == "Solarize":
        return lambda img: ImageOps.solarize(img, int(params["threshold"] * 255))

    if spec.name == "Colorize":
        black = params["black_range"][0]
        white = params["white_range"][0]
        mid = params["mid_range"][0]
        midpoint = params["mid_value_range"][0]
        return lambda img: ImageOps.colorize(
            ImageOps.grayscale(img),
            black=black,
            white=white,
            mid=mid,
            midpoint=midpoint,
        )

    if spec.name == "Dithering":
        return lambda img: img.convert("P", dither=Image.FLOYDSTEINBERG).convert("RGB")

    # Direct ImageFilter equivalents.
    if spec.name == "GaussianBlur":
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius=params["sigma"]))

    if spec.name == "MedianBlur":
        size = params["blur_limit"]
        size = size if size % 2 == 1 else size + 1
        return lambda img: img.filter(ImageFilter.MedianFilter(size=size))

    if spec.name == "Blur":
        return lambda img: img.filter(ImageFilter.BoxBlur(radius=params["radius"]))

    if spec.name == "ModeFilter":
        kernel_lo, kernel_hi = params["kernel_range"]
        return lambda img: img.filter(ImageFilter.ModeFilter(size=random.randrange(kernel_lo | 1, kernel_hi + 1, 2)))

    if spec.name == "UnsharpMask":
        blur_lo, blur_hi = params["blur_limit"]
        alpha_lo, alpha_hi = params["alpha"]

        def _unsharp_mask(img: Image.Image) -> Image.Image:
            radius = random.uniform(blur_lo / 2.0, blur_hi / 2.0)
            percent = int(100 + random.uniform(alpha_lo, alpha_hi) * 200)
            return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=params["threshold"]))

        return _unsharp_mask

    if spec.name == "EnhanceEdge":
        return lambda img: img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    if spec.name == "EnhanceDetail":
        return lambda img: img.filter(ImageFilter.DETAIL)

    if spec.name == "JpegCompression":
        quality = params["quality"]

        def _jpeg(img: Image.Image) -> Image.Image:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).copy()

        return _jpeg

    return None


register_library(LIBRARY, create_image_fn=create_transform)

TRANSFORMS = build_transforms(LIBRARY, media="image")
