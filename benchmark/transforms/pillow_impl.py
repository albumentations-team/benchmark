"""Pillow (PIL) implementations of transforms for the image benchmark.

PIL operates on PIL.Image.Image objects in RGB mode.  Where PIL has no native
path (noise generation, HSV manipulation, vignette gradients) we use numpy for
the arithmetic, but the numpy<->PIL conversion cost is intentionally included in
the timing — that is real PIL overhead for those operations.

No pre-built 256-element LUT tables are constructed outside the callable.
image.point() is used with a plain Python callable where appropriate — PIL
builds its own C-level LUT from the callable internally, which is PIL's own
native mechanism.
"""

import io
import math
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from benchmark.transforms.registry import build_transforms, register_library
from benchmark.transforms.specs import TransformSpec

LIBRARY = "pillow"


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pil_interp(name: str) -> int:
    return Image.BILINEAR if name == "bilinear" else Image.NEAREST


def _find_perspective_coeffs(source: list[tuple[float, float]], target: list[tuple[float, float]]) -> list[float]:
    """8-coefficient perspective transform: maps target (output) coords -> source (input) coords."""
    matrix = []
    for s, t in zip(source, target, strict=False):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    mat = np.array(matrix, dtype=np.float64)
    b = np.array([c for s in source for c in s], dtype=np.float64)
    return np.linalg.solve(mat, b).tolist()


def _pad_to_min_size(img: Image.Image, min_width: int, min_height: int, fill: int = 0) -> Image.Image:
    iw, ih = img.size
    pad_w = max(0, min_width - iw)
    pad_h = max(0, min_height - ih)
    if pad_w == 0 and pad_h == 0:
        return img
    left = pad_w // 2
    top = pad_h // 2
    return ImageOps.expand(img, border=(left, top, pad_w - left, pad_h - top), fill=fill)


def _hsv_shift(arr: np.ndarray, h_shift: float, s_scale: float, v_scale: float) -> np.ndarray:
    """Vectorised RGB->HSV shift->RGB.  arr is float32 in [0, 1], shape HxWx3."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))
    diff = maxc - minc

    v = maxc
    s = np.where(maxc > 0, diff / maxc, 0.0)

    safe_diff = np.where(diff > 0, diff, 1.0)
    rc = np.where(diff > 0, (maxc - r) / safe_diff, 0.0)
    gc = np.where(diff > 0, (maxc - g) / safe_diff, 0.0)
    bc = np.where(diff > 0, (maxc - b) / safe_diff, 0.0)

    h = np.where(maxc == r, bc - gc, np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = np.where(diff == 0, 0.0, h)

    h = (h + h_shift) % 1.0
    s = np.clip(s * s_scale, 0.0, 1.0)
    v = np.clip(v * v_scale, 0.0, 1.0)

    i = (h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i6 = i % 6

    r_o = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4], [v, q, p, p, t], default=v)
    g_o = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4], [t, v, v, q, p], default=p)
    b_o = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4], [p, p, t, v, v], default=q)

    return np.stack([r_o, g_o, b_o], axis=-1)


def _affine_coeffs(angle_deg: float, tx: float, ty: float, scale: float, shear_deg: float) -> tuple[float, ...]:
    """Build PIL AFFINE inverse-mapping coefficients (input = M * output)."""
    a = math.radians(angle_deg)
    shx = math.tan(math.radians(shear_deg))
    cos_a, sin_a = math.cos(a) / scale, math.sin(a) / scale
    # Forward: out = scale * [[cos,-sin+shx*cos],[sin,cos+shx*sin]] * in + [tx,ty]
    # Inverse row0: (cos_a,  sin_a + shx*cos_a, -tx/scale)
    #         row1: (-sin_a, cos_a + shx*sin_a, -ty/scale)  (approximation ignoring shear coupling)
    return (cos_a, sin_a + shx * cos_a, -tx / scale, -sin_a, cos_a + shx * sin_a, -ty / scale)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_transform(spec: TransformSpec) -> Any | None:
    """Create a PIL-based callable from a TransformSpec, or None if unsupported."""
    params = spec.params

    # ------------------------------------------------------------------
    # Geometric
    # ------------------------------------------------------------------
    if spec.name == "Resize":
        size = params["target_size"]
        interp = _pil_interp(params["interpolation"])
        return lambda img: img.resize((size, size), interp)

    if spec.name == "RandomCrop128":
        h, w = params["height"], params["width"]

        def _random_crop(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            x = random.randint(0, max(0, iw - w))
            y = random.randint(0, max(0, ih - h))
            return img.crop((x, y, x + w, y + h))

        return _random_crop

    if spec.name == "CenterCrop128":
        h, w = params["height"], params["width"]

        def _center_crop(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            x = (iw - w) // 2
            y = (ih - h) // 2
            return img.crop((x, y, x + w, y + h))

        return _center_crop

    if spec.name == "RandomResizedCrop":
        th, tw = params["size"]
        scale_lo, scale_hi = params["scale"]
        ratio_lo, ratio_hi = params["ratio"]
        interp = _pil_interp(params["interpolation"])

        def _rrc(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            area = iw * ih
            for _ in range(10):
                target_area = random.uniform(scale_lo, scale_hi) * area
                aspect = math.exp(random.uniform(math.log(ratio_lo), math.log(ratio_hi)))
                cw = round(math.sqrt(target_area * aspect))
                ch = round(math.sqrt(target_area / aspect))
                if 0 < cw <= iw and 0 < ch <= ih:
                    x = random.randint(0, iw - cw)
                    y = random.randint(0, ih - ch)
                    return img.crop((x, y, x + cw, y + ch)).resize((tw, th), interp)
            m = min(iw, ih)
            x, y = (iw - m) // 2, (ih - m) // 2
            return img.crop((x, y, x + m, y + m)).resize((tw, th), interp)

        return _rrc

    if spec.name == "HorizontalFlip":
        return lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)

    if spec.name == "VerticalFlip":
        return lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)

    if spec.name == "Transpose":
        return lambda img: img.transpose(Image.TRANSPOSE)

    if spec.name == "SquareSymmetry":
        _ops = [
            Image.ROTATE_90,
            Image.ROTATE_180,
            Image.ROTATE_270,
            Image.FLIP_LEFT_RIGHT,
            Image.FLIP_TOP_BOTTOM,
            Image.TRANSPOSE,
            Image.TRANSVERSE,
        ]
        return lambda img: img.transpose(random.choice(_ops))

    if spec.name == "RandomRotate90":
        _ops90 = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        return lambda img: img if (op := random.choice(_ops90)) is None else img.transpose(op)

    if spec.name == "Rotate":
        angle = params["angle"]
        fill = params["fill"]
        return lambda img: img.rotate(-angle, resample=Image.NEAREST, fillcolor=fill)

    if spec.name == "SafeRotate":
        limit = params["limit"]
        fill = params["fill"]

        def _safe_rotate(img: Image.Image) -> Image.Image:
            angle = random.uniform(-limit, limit)
            iw, ih = img.size
            rotated = img.rotate(-angle, resample=Image.BILINEAR, expand=True, fillcolor=fill)
            return rotated.resize((iw, ih), Image.BILINEAR)

        return _safe_rotate

    if spec.name == "Pad":
        padding = params["padding"]
        fill = params["fill"]
        return lambda img: ImageOps.expand(img, border=padding, fill=fill)

    if spec.name == "PadIfNeeded":
        min_h = params["min_height"]
        min_w = params["min_width"]
        fill = params["fill"]

        def _pad_if_needed(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            pad_w = max(0, min_w - iw)
            pad_h = max(0, min_h - ih)
            if pad_w == 0 and pad_h == 0:
                return img
            left, top = pad_w // 2, pad_h // 2
            return ImageOps.expand(img, border=(left, top, pad_w - left, pad_h - top), fill=fill)

        return _pad_if_needed

    if spec.name == "Affine":
        tx, ty = params["shift"]
        coeffs = _affine_coeffs(params["angle"], tx, ty, params["scale"], params["shear"][0])
        return lambda img: img.transform(img.size, Image.AFFINE, coeffs, resample=Image.BILINEAR, fillcolor=0)

    if spec.name == "Shear":
        shear_rad = math.tan(math.radians(params["shear"]))
        _shear_data = (1, shear_rad, 0, 0, 1, 0)
        return lambda img: img.transform(img.size, Image.AFFINE, _shear_data, resample=Image.BILINEAR, fillcolor=0)

    if spec.name == "ShiftScaleRotate":
        shift_lim = params["shift_limit"]
        scale_lim = params["scale_limit"]
        rotate_lim = params["rotate_limit"]
        fill = params["fill"]

        def _ssr(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            angle = math.radians(random.uniform(-rotate_lim, rotate_lim))
            scale = 1.0 + random.uniform(-scale_lim, scale_lim)
            tx = random.uniform(-shift_lim, shift_lim) * iw
            ty = random.uniform(-shift_lim, shift_lim) * ih
            cos_a, sin_a = math.cos(angle) / scale, math.sin(angle) / scale
            cx, cy = iw / 2, ih / 2
            c = cx - cos_a * cx - sin_a * cy - tx / scale
            f = cy + sin_a * cx - cos_a * cy - ty / scale
            return img.transform(
                img.size,
                Image.AFFINE,
                (cos_a, sin_a, c, -sin_a, cos_a, f),
                resample=Image.BILINEAR,
                fillcolor=fill,
            )

        return _ssr

    if spec.name == "Perspective":
        scale_max = params["scale"][1]
        fill = params["fill"]

        def _perspective(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            dx, dy = int(iw * scale_max), int(ih * scale_max)
            src = [(0, 0), (iw, 0), (iw, ih), (0, ih)]
            dst = [
                (random.randint(0, dx), random.randint(0, dy)),
                (iw - random.randint(0, dx), random.randint(0, dy)),
                (iw - random.randint(0, dx), ih - random.randint(0, dy)),
                (random.randint(0, dx), ih - random.randint(0, dy)),
            ]
            coeffs = _find_perspective_coeffs(src, dst)
            return img.transform((iw, ih), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=fill)

        return _perspective

    if spec.name == "RandomScale":
        scale_lo, scale_hi = params["scale_limit"]

        def _random_scale(img: Image.Image) -> Image.Image:
            factor = 1.0 + random.uniform(scale_lo, scale_hi)
            iw, ih = img.size
            return img.resize((max(1, int(iw * factor)), max(1, int(ih * factor))), Image.BILINEAR)

        return _random_scale

    if spec.name == "Downscale":
        scale = params["scale_range"][0]
        down_resample = _pil_interp(params["interpolation_pair"][0])
        up_resample = _pil_interp(params["interpolation_pair"][1])

        def _downscale(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            small = img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))), down_resample)
            return small.resize((iw, ih), up_resample)

        return _downscale

    if spec.name == "LongestMaxSize":
        max_size = params["max_size"]

        def _longest(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            longest = max(iw, ih)
            if longest <= max_size:
                return img
            s = max_size / longest
            return img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.BILINEAR)

        return _longest

    if spec.name == "SmallestMaxSize":
        max_size = params["max_size"]

        def _smallest(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            smallest = min(iw, ih)
            if smallest >= max_size:
                return img
            s = max_size / smallest
            return img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.BILINEAR)

        return _smallest

    if spec.name == "CropAndPad":
        # px = (top, right, bottom, left): negative -> pad, positive -> crop
        top_px, right_px, bottom_px, left_px = params["px"]
        fill = params["fill"]

        def _crop_and_pad(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            crop_l = max(0, left_px)
            crop_r = max(0, right_px)
            crop_t = max(0, top_px)
            crop_b = max(0, bottom_px)
            img = img.crop((crop_l, crop_t, iw - crop_r if crop_r else iw, ih - crop_b if crop_b else ih))
            pad_l = -left_px if left_px < 0 else 0
            pad_r = -right_px if right_px < 0 else 0
            pad_t = -top_px if top_px < 0 else 0
            pad_b = -bottom_px if bottom_px < 0 else 0
            if any((pad_l, pad_r, pad_t, pad_b)):
                img = ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=fill)
            return img

        return _crop_and_pad

    if spec.name == "RandomSizedCrop":
        min_h, max_h = params["min_max_height"]
        out_h, out_w = params["size"]
        interp = _pil_interp(params["interpolation"])

        def _random_sized_crop(img: Image.Image) -> Image.Image:
            img = _pad_to_min_size(img, min_width=min_h, min_height=min_h)
            iw, ih = img.size
            h = random.randint(min_h, min(max_h, ih, iw))
            w = h
            x = random.randint(0, max(0, iw - w))
            y = random.randint(0, max(0, ih - h))
            return img.crop((x, y, x + w, y + h)).resize((out_w, out_h), interp)

        return _random_sized_crop

    # ------------------------------------------------------------------
    # Color — ImageEnhance + ImageOps
    # ------------------------------------------------------------------
    if spec.name == "Brightness":
        limit = params["brightness_limit"]
        offset = float(limit[0] if isinstance(limit, (list, tuple)) else limit)
        factor = 1.0 + offset
        return lambda img: ImageEnhance.Brightness(img).enhance(factor)

    if spec.name == "Contrast":
        limit = params["contrast_limit"]
        offset = float(limit[0] if isinstance(limit, (list, tuple)) else limit)
        factor = 1.0 + offset
        return lambda img: ImageEnhance.Contrast(img).enhance(factor)

    if spec.name == "Saturation":
        factor = 1.0 + params["saturation_factor"]
        return lambda img: ImageEnhance.Color(img).enhance(factor)

    if spec.name == "Sharpen":
        alpha_lo, alpha_hi = params["alpha"]

        def _sharpen(img: Image.Image) -> Image.Image:
            alpha = random.uniform(alpha_lo, alpha_hi)
            sharpened = img.filter(ImageFilter.SHARPEN)
            return Image.blend(img, sharpened, alpha)

        return _sharpen

    if spec.name == "AutoContrast":
        return ImageOps.autocontrast

    if spec.name == "Equalize":
        return ImageOps.equalize

    if spec.name == "Grayscale":
        return lambda img: ImageOps.grayscale(img).convert("RGB")

    if spec.name == "Invert":
        return ImageOps.invert

    if spec.name == "Posterize":
        bits = params["bits"]
        return lambda img: ImageOps.posterize(img, int(bits))

    if spec.name == "Solarize":
        # spec threshold is in [0,1]; PIL solarize threshold is [0,255]
        threshold = int(params["threshold"] * 255)
        return lambda img: ImageOps.solarize(img, threshold)

    if spec.name == "ColorJitter":
        brightness = params["brightness"]
        contrast = params["contrast"]
        saturation = params["saturation"]
        # Hue is skipped — no native PIL hue shift without numpy roundtrip

        def _color_jitter(img: Image.Image) -> Image.Image:
            bf = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
            cf = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
            sf = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
            img = ImageEnhance.Brightness(img).enhance(bf)
            img = ImageEnhance.Contrast(img).enhance(cf)
            return ImageEnhance.Color(img).enhance(sf)

        return _color_jitter

    if spec.name == "ColorJiggle":
        brightness_lo, brightness_hi = params["brightness"]
        contrast_lo, contrast_hi = params["contrast"]
        saturation_lo, saturation_hi = params["saturation"]

        def _color_jiggle(img: Image.Image) -> Image.Image:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(brightness_lo, brightness_hi))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(contrast_lo, contrast_hi))
            return ImageEnhance.Color(img).enhance(random.uniform(saturation_lo, saturation_hi))

        return _color_jiggle

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

    if spec.name == "EnhanceEdge":
        return lambda img: img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    if spec.name == "EnhanceDetail":
        return lambda img: img.filter(ImageFilter.DETAIL)

    if spec.name == "ToSepia":
        _sepia = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], dtype=np.float32)

        def _to_sepia(img: Image.Image) -> Image.Image:
            arr = np.array(img, dtype=np.float32)
            return Image.fromarray(np.clip(arr @ _sepia.T, 0, 255).astype(np.uint8))

        return _to_sepia

    if spec.name == "Dithering":
        return lambda img: img.convert("P", dither=Image.FLOYDSTEINBERG).convert("RGB")

    if spec.name == "RandomGamma":
        gamma = params["gamma"] / 100.0
        return lambda img: img.point(lambda p: min(255, int((p / 255.0) ** gamma * 255)))

    if spec.name == "RandomToneCurve":
        scale = params["scale"]

        def _tone_curve(img: Image.Image) -> Image.Image:
            intensity = random.uniform(-scale, scale)
            return img.point(
                lambda p: max(0, min(255, int((p / 255.0 + intensity * (p / 255.0) * (1.0 - p / 255.0) * 4.0) * 255))),
            )

        return _tone_curve

    # ------------------------------------------------------------------
    # Filters — ImageFilter
    # ------------------------------------------------------------------
    if spec.name == "GaussianBlur":
        radius = params["sigma"]
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius=radius))

    if spec.name == "MedianBlur":
        size = params["blur_limit"]
        # MedianFilter requires odd size
        size = size if size % 2 == 1 else size + 1
        return lambda img: img.filter(ImageFilter.MedianFilter(size=size))

    if spec.name == "Blur":
        radius = params["radius"]
        return lambda img: img.filter(ImageFilter.BoxBlur(radius=radius))

    if spec.name == "ModeFilter":
        kernel_lo, kernel_hi = params["kernel_range"]

        def _mode_filter(img: Image.Image) -> Image.Image:
            size = random.randrange(kernel_lo | 1, kernel_hi + 1, 2)
            return img.filter(ImageFilter.ModeFilter(size=size))

        return _mode_filter

    if spec.name == "UnsharpMask":
        blur_lo, blur_hi = params["blur_limit"]
        threshold = params["threshold"]
        alpha_lo, alpha_hi = params["alpha"]

        def _unsharp_mask(img: Image.Image) -> Image.Image:
            radius = random.uniform(blur_lo / 2.0, blur_hi / 2.0)
            alpha = random.uniform(alpha_lo, alpha_hi)
            percent = int(100 + alpha * 200)
            return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

        return _unsharp_mask

    if spec.name == "Emboss":
        alpha_lo, alpha_hi = params["alpha"]

        def _emboss(img: Image.Image) -> Image.Image:
            embossed = img.filter(ImageFilter.EMBOSS)
            alpha = random.uniform(alpha_lo, alpha_hi)
            return Image.blend(img, embossed, alpha)

        return _emboss

    if spec.name == "Morphological":
        scale_lo, scale_hi = params["scale"]
        operation = params["operation"]

        def _morphological(img: Image.Image) -> Image.Image:
            size = random.randint(int(scale_lo), int(scale_hi))
            size = size if size % 2 == 1 else size + 1
            filt = ImageFilter.MaxFilter(size) if operation == "dilation" else ImageFilter.MinFilter(size)
            return img.filter(filt)

        return _morphological

    if spec.name == "Defocus":
        radius_lo, radius_hi = params["radius"]

        def _defocus(img: Image.Image) -> Image.Image:
            radius = random.uniform(radius_lo, radius_hi)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        return _defocus

    if spec.name == "JpegCompression":
        quality = params["quality"]

        def _jpeg(img: Image.Image) -> Image.Image:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).copy()

        return _jpeg

    # ------------------------------------------------------------------
    # Pixel ops — PIL native where possible, numpy where unavoidable
    # ------------------------------------------------------------------
    if spec.name == "ChannelShuffle":

        def _channel_shuffle(img: Image.Image) -> Image.Image:
            channels = list(img.split())
            random.shuffle(channels)
            return Image.merge("RGB", channels)

        return _channel_shuffle

    if spec.name == "ChannelSwap":

        def _channel_swap(img: Image.Image) -> Image.Image:
            channels = list(img.split())
            random.shuffle(channels)
            return Image.merge("RGB", channels)

        return _channel_swap

    if spec.name == "ChannelDropout":

        def _channel_dropout(img: Image.Image) -> Image.Image:
            channels = list(img.split())
            idx = random.randint(0, 2)
            channels[idx] = Image.new("L", img.size, 0)
            return Image.merge("RGB", channels)

        return _channel_dropout

    if spec.name == "Erasing":
        scale_lo, scale_hi = params["scale"]
        ratio_lo, ratio_hi = params["ratio"]
        fill = params["fill"]

        def _erasing(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            area = iw * ih
            for _ in range(10):
                erase_area = random.uniform(scale_lo, scale_hi) * area
                aspect = random.uniform(ratio_lo, ratio_hi)
                eh = int(math.sqrt(erase_area / aspect))
                ew = int(math.sqrt(erase_area * aspect))
                if 0 < eh <= ih and 0 < ew <= iw:
                    x = random.randint(0, iw - ew)
                    y = random.randint(0, ih - eh)
                    out = img.copy()
                    ImageDraw.Draw(out).rectangle([x, y, x + ew, y + eh], fill=fill)
                    return out
            return img

        return _erasing

    if spec.name == "CoarseDropout":
        h_lo, h_hi = params["hole_height_range"]
        w_lo, w_hi = params["hole_width_range"]
        n_lo, n_hi = params["num_holes_range"]

        def _coarse_dropout(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            out = img.copy()
            draw = ImageDraw.Draw(out)
            n = random.randint(int(n_lo), int(n_hi))
            for _ in range(n):
                hh = int(random.uniform(h_lo, h_hi) * ih)
                hw = int(random.uniform(w_lo, w_hi) * iw)
                x = random.randint(0, max(0, iw - hw))
                y = random.randint(0, max(0, ih - hh))
                draw.rectangle([x, y, x + hw, y + hh], fill=0)
            return out

        return _coarse_dropout

    if spec.name == "RGBShift":
        pixel_shift = params["pixel_shift"]

        def _rgb_shift(img: Image.Image) -> Image.Image:
            arr = np.array(img, dtype=np.int32)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + random.randint(-pixel_shift, pixel_shift), 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + random.randint(-pixel_shift, pixel_shift), 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + random.randint(-pixel_shift, pixel_shift), 0, 255)
            return Image.fromarray(arr.astype(np.uint8))

        return _rgb_shift

    if spec.name == "GaussianNoise":
        mean = params["mean"]
        std = params["std"]

        def _gaussian_noise(img: Image.Image) -> Image.Image:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(mean * 255, std * 255, arr.shape).astype(np.float32)
            return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

        return _gaussian_noise

    if spec.name == "SaltAndPepper":
        amount_lo, amount_hi = params["amount"]
        svp_lo, svp_hi = params["salt_vs_pepper"]

        def _salt_and_pepper(img: Image.Image) -> Image.Image:
            arr = np.array(img)
            h, w = arr.shape[:2]
            amount = random.uniform(amount_lo, amount_hi)
            svp = random.uniform(svp_lo, svp_hi)
            n_pixels = int(amount * h * w)
            n_salt = int(n_pixels * svp)
            n_pepper = n_pixels - n_salt
            rs = np.random.randint(0, h, n_salt)
            cs = np.random.randint(0, w, n_salt)
            arr[rs, cs] = 255
            rp = np.random.randint(0, h, n_pepper)
            cp = np.random.randint(0, w, n_pepper)
            arr[rp, cp] = 0
            return Image.fromarray(arr)

        return _salt_and_pepper

    if spec.name == "Normalize":
        mean = np.array(params["mean"], dtype=np.float32)
        std = np.array(params["std"], dtype=np.float32)

        def _normalize(img: Image.Image) -> Image.Image:
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            # Map back to uint8 for a returnable PIL image
            arr = np.clip(arr * 64.0 + 127.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        return _normalize

    if spec.name == "PixelDropout":
        dropout_prob = params["dropout_prob"]
        drop_value = params["drop_value"]

        def _pixel_dropout(img: Image.Image) -> Image.Image:
            arr = np.array(img)
            mask = np.random.random((arr.shape[0], arr.shape[1])) < dropout_prob
            arr[mask] = drop_value
            return Image.fromarray(arr)

        return _pixel_dropout

    if spec.name == "Vignetting":
        intensity_lo, intensity_hi = params["intensity_range"]

        def _vignetting(img: Image.Image) -> Image.Image:
            iw, ih = img.size
            intensity = random.uniform(intensity_lo, intensity_hi)
            arr = np.array(img, dtype=np.float32)
            cx, cy = iw / 2.0, ih / 2.0
            yy, xx = np.mgrid[0:ih, 0:iw]
            dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            mask = np.clip(1.0 - intensity * dist, 0.0, 1.0)[:, :, np.newaxis]
            return Image.fromarray(np.clip(arr * mask, 0, 255).astype(np.uint8))

        return _vignetting

    if spec.name == "FilmGrain":
        intensity_lo, intensity_hi = params["intensity_range"]

        def _film_grain(img: Image.Image) -> Image.Image:
            arr = np.array(img, dtype=np.float32)
            intensity = random.uniform(intensity_lo, intensity_hi)
            grain = np.random.normal(0, intensity * 255, arr.shape).astype(np.float32)
            return Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8))

        return _film_grain

    if spec.name == "HSV":
        hue_gain = params["hue"]
        sat_gain = params["saturation"]
        val_gain = params["value"]

        def _hsv(img: Image.Image) -> Image.Image:
            arr = np.array(img, dtype=np.float32) / 255.0
            h_shift = random.uniform(-hue_gain, hue_gain)
            s_scale = 1.0 + random.uniform(-sat_gain, sat_gain)
            v_scale = 1.0 + random.uniform(-val_gain, val_gain)
            arr = _hsv_shift(arr, h_shift, s_scale, v_scale)
            return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

        return _hsv

    if spec.name == "Hue":
        hue_deg = params["hue"]
        h_shift = hue_deg / 360.0

        def _hue(img: Image.Image) -> Image.Image:
            arr = np.array(img, dtype=np.float32) / 255.0
            shift = random.uniform(-h_shift, h_shift)
            arr = _hsv_shift(arr, shift, 1.0, 1.0)
            return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

        return _hue

    return None


register_library(LIBRARY, create_image_fn=create_transform)

TRANSFORMS = build_transforms(LIBRARY, media="image")
