"""Kornia transform factories for the active RGB recipe catalog."""

import random
from collections.abc import Callable
from typing import Any

import kornia
import kornia.augmentation as kornia_augmentation
import torch
import torch.nn.functional as torch_functional

from augbench.implementations.geometry import aspect_size, centered_padding, emboss_kernel
from augbench.implementations.shear import x_shear_matrix
from augbench.implementations.specs import TransformSpec

LIBRARY = "kornia"


class _TensorOperation(torch.nn.Module):
    def __init__(self, operation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self._operation = operation

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._operation(image)


class _Morphological(torch.nn.Module):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self._operation = kornia.morphology.dilation if operation == "dilation" else kornia.morphology.erosion
        self.register_buffer("kernel", torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._operation(image, self.kernel.to(image), engine="convolution")


def _build_center_crop(params: dict[str, Any]) -> Any:
    height, width = params["height"], params["width"]
    crop = kornia.augmentation.CenterCrop(size=(height, width), cropping_mode="slice", p=1)

    def apply(image: torch.Tensor) -> torch.Tensor:
        in_height, in_width = image.shape[-2:]
        left, top, right, bottom = centered_padding(in_height, in_width, max(in_height, height), max(in_width, width))
        if left or top or right or bottom:
            image = torch_functional.pad(image, (left, right, top, bottom))
        return crop(image)

    return _TensorOperation(apply)


def _build_crop(params: dict[str, Any]) -> Any:
    top, left = params["top"], params["left"]
    height, width = params["height"], params["width"]
    bottom, right = top + height - 1, left + width - 1
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))

    def crop(image: torch.Tensor) -> torch.Tensor:
        boxes = image.new_tensor([corners]).expand(image.shape[0], -1, -1)
        return kornia.geometry.transform.crop_by_indices(image, boxes, size=(height, width))

    return _TensorOperation(crop)


def _build_crop_and_pad(params: dict[str, Any]) -> Any:
    top, right, bottom, left = params["sides"]
    return _TensorOperation(lambda image: torch_functional.pad(image, (left, right, top, bottom)))


def _build_pad(params: dict[str, Any]) -> Any:
    return _TensorOperation(lambda image: torch_functional.pad(image, (params["padding"],) * 4))


def _build_pad_if_needed(params: dict[str, Any]) -> Any:
    def pad(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        right, bottom = max(params["min_width"] - width, 0), max(params["min_height"] - height, 0)
        return torch_functional.pad(image, (0, right, 0, bottom)) if right or bottom else image

    return _TensorOperation(pad)


def _build_letter_box(params: dict[str, Any]) -> Any:
    def letter_box(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        height, width = aspect_size(height, width, params["height"])
        resized = kornia.geometry.transform.resize(image, (height, width), align_corners=False, antialias=False)
        left, top, right, bottom = centered_padding(height, width, params["height"], params["width"])
        return torch_functional.pad(resized, (left, right, top, bottom))

    return _TensorOperation(letter_box)


def _build_random_scale(params: dict[str, Any]) -> Any:
    def resize(image: torch.Tensor) -> torch.Tensor:
        factor = random.uniform(*params["scale"])  # noqa: S311 - augmentation RNG.
        height, width = image.shape[-2:]
        size = (int(height * factor), int(width * factor))
        return kornia.geometry.transform.resize(image, size, align_corners=False, antialias=False)

    return _TensorOperation(resize)


def _build_random_sized_crop(params: dict[str, Any]) -> Any:
    area = (params["crop_size"] / params["input_size"]) ** 2
    return kornia_augmentation.RandomResizedCrop(
        tuple(params["size"]), scale=(area, area), ratio=(1.0, 1.0), cropping_mode="slice", align_corners=False, p=1
    )


def _build_coarse_dropout(params: dict[str, Any]) -> Any:
    area = (params["hole_size"] / params["input_size"]) ** 2
    return kornia_augmentation.RandomErasing(scale=(area, area), ratio=(1.0, 1.0), value=0, p=1)


def _build_colorize(params: dict[str, Any]) -> Any:
    anchors = list(zip(params["black"], params["white"], strict=True))
    palette = [tuple((black + (white - black) * value / 255) / 255 for black, white in anchors) for value in range(256)]
    colormap = kornia.color.ColorMap(palette, num_colors=256)
    return torch.nn.Sequential(kornia.color.RgbToGrayscale(), kornia.color.ApplyColorMap(colormap))


class _Emboss(torch.nn.Module):
    def __init__(self, alpha: float, strength: float) -> None:
        super().__init__()
        self.register_buffer("kernel", torch.tensor(emboss_kernel(alpha, strength)).reshape(1, 3, 3))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return kornia.filters.filter2d(image, self.kernel.to(image)).clamp_(0, 1)


class _RandomJigsawWithPad(torch.nn.Module):
    def __init__(self, grid: tuple[int, int]) -> None:
        super().__init__()
        self.grid = grid
        self.jigsaw = kornia_augmentation.RandomJigsaw(grid=grid, p=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        grid_h, grid_w = self.grid
        pad_height = (-height) % grid_h
        pad_width = (-width) % grid_w
        if pad_height or pad_width:
            image = torch_functional.pad(image, (0, pad_width, 0, pad_height))
        return self.jigsaw(image)[..., :height, :width]


class _FixedAffine(torch.nn.Module):
    def __init__(
        self,
        *,
        angle_degrees: float,
        translation: tuple[float, float],
        scale_factor: float,
        shear: tuple[float, float],
    ) -> None:
        super().__init__()
        self.register_buffer("angle", torch.tensor([angle_degrees], dtype=torch.float32))
        self.register_buffer("translation", torch.tensor([translation], dtype=torch.float32))
        self.register_buffer("scale_factor", torch.tensor([[scale_factor, scale_factor]], dtype=torch.float32))
        self.register_buffer("shear", torch.tensor([shear], dtype=torch.float32))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = int(image.shape[0])
        angle = self.angle.to(image)
        translation = self.translation.to(image)
        scale_factor = self.scale_factor.to(image)
        shear = self.shear.to(image)
        transform = kornia.geometry.transform.Affine(
            angle=angle.expand(batch_size),
            translation=translation.expand(batch_size, -1),
            scale_factor=scale_factor.expand(batch_size, -1),
            shear=shear.expand(batch_size, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return transform(image)


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image.unsqueeze(0)).squeeze(0)


def create_transform(spec: TransformSpec) -> Any | None:
    builder = _TRANSFORM_BUILDERS.get(spec.name)
    return builder(spec.params) if builder is not None else None


def _build_color_jitter(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.ColorJitter(
        brightness=params["brightness"],
        contrast=params["contrast"],
        saturation=params["saturation"],
        hue=params["hue"],
        p=1,
        same_on_batch=False,
    )


def _build_color_jiggle(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.ColorJiggle(
        brightness=params["brightness"],
        contrast=params["contrast"],
        saturation=params["saturation"],
        hue=params["hue"],
        p=1,
        same_on_batch=False,
    )


def _build_auto_contrast(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomAutoContrast(p=1)


def _build_blur(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomBoxBlur(
        p=1,
        kernel_size=(params["kernel_size"], params["kernel_size"]),
        border_type=params["border_mode"],
    )


def _build_brightness(params: dict[str, Any]) -> Any | None:
    # Kornia brightness is multiplicative (1.0=no change). Spec limit is additive offset,
    # so +0.2 additive -> multiplicative factor (1.2, 1.2).
    limit = params["brightness_limit"]
    offset = float(limit[0] if isinstance(limit, (list, tuple)) else limit)
    return kornia_augmentation.RandomBrightness(
        brightness=(1.0 + offset, 1.0 + offset),
        p=1,
    )


def _build_channel_dropout(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomChannelDropout(p=1)


def _build_channel_shuffle(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomChannelShuffle(p=1)


def _build_clahe(params: dict[str, Any]) -> Any | None:
    # Pass float clip_limit to avoid Kornia's internal Long-dtype sampling bug
    clip = params["clip_limit"]
    clip_float = (float(clip[0]), float(clip[1])) if isinstance(clip, (list, tuple)) else (float(clip), float(clip))
    return kornia_augmentation.RandomClahe(
        p=1,
        clip_limit=clip_float,
        grid_size=tuple(params["tile_grid_size"]),
    )


def _build_contrast(params: dict[str, Any]) -> Any | None:
    # Kornia contrast is multiplicative (1.0=no change). Convert additive offset to factor.
    limit = params["contrast_limit"]
    offset = float(limit[0] if isinstance(limit, (list, tuple)) else limit)
    return kornia_augmentation.RandomContrast(
        contrast=(1.0 + offset, 1.0 + offset),
        p=1,
    )


def _build_equalize(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomEqualize(p=1)


def _build_random_gamma(params: dict[str, Any]) -> Any | None:
    gamma = params["gamma"] / 100
    return kornia_augmentation.RandomGamma(
        gamma=(gamma, gamma),
        p=1,
    )


def _build_gaussian_blur(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomGaussianBlur(
        kernel_size=tuple(params["kernel_size"]),
        sigma=(params["sigma"], params["sigma"]),
        p=1,
    )


def _build_linear_illumination(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomLinearIllumination(
        gain=tuple(params["gain"]),
        p=1,
    )


def _build_corner_illumination(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomLinearCornerIllumination(
        gain=tuple(params["gain"]),
        p=1,
    )


def _build_gaussian_illumination(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomGaussianIllumination(
        gain=tuple(params["gain"]),
        p=1,
    )


def _build_gaussian_noise(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomGaussianNoise(
        mean=params["mean"],
        std=params["std"],
        p=1,
    )


def _build_grayscale(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomGrayscale(p=1)


def _build_hue(params: dict[str, Any]) -> Any | None:
    # Kornia expects hue in [-0.5, 0.5] (fraction of 360°). Spec is in degrees.
    hue_fraction = params["hue"] / 360.0
    return kornia_augmentation.RandomHue(
        hue=(-hue_fraction, hue_fraction),
        p=1,
    )


def _build_invert(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomInvert(p=1)


def _build_jpeg_compression(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomJPEG(
        jpeg_quality=params["quality"],
        p=1,
    )


def _build_median_blur(params: dict[str, Any]) -> Any | None:
    kernel_size = params["blur_limit"]
    return kornia_augmentation.RandomMedianBlur(
        kernel_size=(kernel_size, kernel_size),
        p=1,
    )


def _build_motion_blur(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomMotionBlur(
        kernel_size=params["kernel_size"],
        angle=tuple(params["angle_range"]),
        direction=tuple(params["direction_range"]),
        p=1,
    )


def _build_plankian_jitter(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlanckianJitter(
        mode=params["mode"],
        p=1,
    )


def _build_plasma_brightness(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlasmaBrightness(
        roughness=(params["roughness"], params["roughness"]),
        p=1,
    )


def _build_plasma_contrast(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlasmaContrast(
        roughness=(params["roughness"], params["roughness"]),
        p=1,
    )


def _build_plasma_shadow(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPlasmaShadow(
        roughness=(params["roughness"], params["roughness"]),
        p=1,
    )


def _build_rain(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomRain(
        drop_width=(params["drop_width"], params["drop_width"]),
        drop_height=(params["drop_height"], params["drop_height"]),
        p=1,
    )


def _build_rgbshift(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomRGBShift(
        r_shift_limit=params["pixel_shift"] / 255.0,
        g_shift_limit=params["pixel_shift"] / 255.0,
        b_shift_limit=params["pixel_shift"] / 255.0,
        p=1,
    )


def _build_salt_and_pepper(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomSaltAndPepperNoise(
        amount=tuple(params["amount"]),
        salt_vs_pepper=tuple(params["salt_vs_pepper"]),
        p=1,
    )


def _build_saturation(params: dict[str, Any]) -> Any | None:
    # Kornia saturation is multiplicative (1.0=no change, >1=more saturated).
    # Spec saturation_factor=0.5 is an additive-style offset -> factor (1.5, 1.5).
    factor = 1.0 + params["saturation_factor"]
    return kornia_augmentation.RandomSaturation(
        saturation=(factor, factor),
        p=1,
    )


def _build_sharpen(_params: dict[str, Any]) -> Any | None:
    # sharpness=2.0 produces noticeable sharpening (default 0.5 barely changes the image)
    return kornia_augmentation.RandomSharpness(
        sharpness=2.0,
        p=1,
    )


def _build_snow(params: dict[str, Any]) -> Any | None:
    # Pass explicit float brightness to avoid Kornia's internal Long-dtype sampling bug
    return kornia_augmentation.RandomSnow(
        snow_coefficient=tuple(params["snow_point_range"]),
        brightness=(2.0, 2.0),
        p=1,
    )


def _build_solarize(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomSolarize(
        thresholds=params["threshold"],
        p=1,
    )


def _build_affine(params: dict[str, Any]) -> Any | None:
    angle_degrees = float(params["angle"])
    # Translation in pixels (same as albumentations translate_px)
    tx, ty = float(params["shift"][0]), float(params["shift"][1])
    scale_factor = float(params["scale"])
    shear_value = float(params["shear"]) if isinstance(params["shear"], int | float) else 0.0

    return _FixedAffine(
        angle_degrees=angle_degrees,
        translation=(tx, ty),
        scale_factor=scale_factor,
        shear=(shear_value, shear_value),
    )


def _build_shift_scale_rotate(params: dict[str, Any]) -> Any:
    return _FixedAffine(
        angle_degrees=params["angle"],
        translation=tuple(params["shift"]),
        scale_factor=params["scale"],
        shear=(0, 0),
    )


def _build_random_crop224(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomCrop(
        size=(params["height"], params["width"]),
        pad_if_needed=True,
        p=1,
    )


def _build_elastic(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomElasticTransform(
        p=1,
        sigma=(params["sigma"], params["sigma"]),
        alpha=(params["kornia_alpha"], params["kornia_alpha"]),
    )


def _build_erasing(params: dict[str, Any]) -> Any | None:
    # Pass float value to avoid Kornia's internal Long-dtype sampling bug
    return kornia_augmentation.RandomErasing(
        p=1,
        scale=tuple(params["scale"]),
        ratio=tuple(params["ratio"]),
        value=float(params["fill"]),
    )


def _build_optical_distortion(_params: dict[str, Any]) -> Any | None:
    return kornia.augmentation.RandomFisheye(
        center_x=torch.tensor([-0.3, 0.3]),
        center_y=torch.tensor([-0.3, 0.3]),
        gamma=torch.tensor([0.9, 1.1]),
        p=1,
    )


def _build_horizontal_flip(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomHorizontalFlip(p=1)


def _build_perspective(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPerspective(
        distortion_scale=params["scale"][1],
        resample=params["interpolation"],
        p=1,
    )


def _build_random_resized_crop(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomResizedCrop(
        size=tuple(params["size"]),
        scale=tuple(params["scale"]),
        ratio=tuple(params["ratio"]),
        resample=params["interpolation"],
        align_corners=False,
        cropping_mode="slice",
        p=1,
    )


def _build_random_rotate90(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomRotation90(times=tuple(params["times"]), p=1)


def _build_random_jigsaw(params: dict[str, Any]) -> Any | None:
    return _RandomJigsawWithPad(grid=tuple(params["grid"]))


def _build_rotate(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomRotation(
        degrees=tuple(params["angle_range"]),
        p=1,
    )


def _build_shear(params: dict[str, Any]) -> Any | None:
    return _FixedXShear(float(params["shear"]))


class _FixedXShear(torch.nn.Module):
    def __init__(self, degrees: float) -> None:
        super().__init__()
        self._degrees = degrees

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        matrix = torch.tensor(
            x_shear_matrix(self._degrees, (height - 1) / 2),
            dtype=torch.float32,
            device=image.device,
        ).reshape(1, 2, 3)
        return kornia.geometry.transform.warp_affine(
            image.float(),
            matrix.expand(image.shape[0], -1, -1),
            (height, width),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(image.dtype)


def _build_thin_plate_spline(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomThinPlateSpline(
        scale=params["distortion_scale"],
        p=1,
    )


def _build_vertical_flip(_params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomVerticalFlip(p=1)


def _build_resize(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.Resize(
        size=(params["target_size"], params["target_size"]),
        resample=params["interpolation"],
        align_corners=False,
        antialias=params.get("antialias", False),
        p=1,
    )


def _build_normalize(_params: dict[str, Any]) -> Any | None:
    # Normalization is applied by GpuBatchNormalize after H2D, not by a
    # CPU Kornia recipe.
    return None


def _build_posterize(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.RandomPosterize(
        bits=float(params["bits"]),
        p=1,
    )


def _build_longest_max_size(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.LongestMaxSize(
        max_size=params["max_size"],
        resample=params["interpolation"],
        p=1,
    )


def _build_smallest_max_size(params: dict[str, Any]) -> Any | None:
    return kornia_augmentation.SmallestMaxSize(
        max_size=params["max_size"],
        resample=params["interpolation"],
        p=1,
    )


_TRANSFORM_BUILDERS = {
    "ColorJitter": _build_color_jitter,
    "ColorJiggle": _build_color_jiggle,
    "AutoContrast": _build_auto_contrast,
    "Blur": _build_blur,
    "Brightness": _build_brightness,
    "ChannelDropout": _build_channel_dropout,
    "ChannelShuffle": _build_channel_shuffle,
    "CLAHE": _build_clahe,
    "Contrast": _build_contrast,
    "Equalize": _build_equalize,
    "RandomGamma": _build_random_gamma,
    "GaussianBlur": _build_gaussian_blur,
    "LinearIllumination": _build_linear_illumination,
    "CornerIllumination": _build_corner_illumination,
    "GaussianIllumination": _build_gaussian_illumination,
    "GaussianNoise": _build_gaussian_noise,
    "Grayscale": _build_grayscale,
    "Hue": _build_hue,
    "Invert": _build_invert,
    "JpegCompression": _build_jpeg_compression,
    "MedianBlur": _build_median_blur,
    "MotionBlur": _build_motion_blur,
    "PlankianJitter": _build_plankian_jitter,
    "PlasmaBrightness": _build_plasma_brightness,
    "PlasmaContrast": _build_plasma_contrast,
    "PlasmaShadow": _build_plasma_shadow,
    "Rain": _build_rain,
    "RGBShift": _build_rgbshift,
    "SaltAndPepper": _build_salt_and_pepper,
    "Saturation": _build_saturation,
    "Sharpen": _build_sharpen,
    "Snow": _build_snow,
    "Solarize": _build_solarize,
    "Affine": _build_affine,
    "RandomCrop224": _build_random_crop224,
    "CenterCrop": _build_center_crop,
    "Crop": _build_crop,
    "CropAndPad": _build_crop_and_pad,
    "Pad": _build_pad,
    "PadIfNeeded": _build_pad_if_needed,
    "LetterBox": _build_letter_box,
    "RandomScale": _build_random_scale,
    "RandomSizedCrop": _build_random_sized_crop,
    "CoarseDropout": _build_coarse_dropout,
    "Colorize": _build_colorize,
    "Emboss": lambda params: _Emboss(params["alpha"], params["strength"]),
    "ShiftScaleRotate": _build_shift_scale_rotate,
    "ChannelSwap": lambda _: _TensorOperation(kornia.color.rgb_to_bgr),
    "Transpose": lambda _: _TensorOperation(lambda image: image.transpose(-2, -1)),
    "Morphological": lambda params: _Morphological(params["operation"]),
    "Elastic": _build_elastic,
    "Erasing": _build_erasing,
    "OpticalDistortion": _build_optical_distortion,
    "HorizontalFlip": _build_horizontal_flip,
    "Perspective": _build_perspective,
    "RandomResizedCrop": _build_random_resized_crop,
    "RandomRotate90": _build_random_rotate90,
    "RandomJigsaw": _build_random_jigsaw,
    "Rotate": _build_rotate,
    "Shear": _build_shear,
    "ThinPlateSpline": _build_thin_plate_spline,
    "VerticalFlip": _build_vertical_flip,
    "Resize": _build_resize,
    "Normalize": _build_normalize,
    "Posterize": _build_posterize,
    "LongestMaxSize": _build_longest_max_size,
    "SmallestMaxSize": _build_smallest_max_size,
}
