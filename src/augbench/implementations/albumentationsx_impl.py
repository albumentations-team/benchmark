"""AlbumentationsX transforms used by the active RGB recipe catalog."""

from typing import Any

import albumentations
import cv2

from augbench.implementations.specs import TransformSpec

# Required: Library name for dependency installation
LIBRARY = "albumentationsx"


def _range(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple | list):
        return (value[0], value[1])
    return (value, value)


def _symmetric_range(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple | list):
        return (value[0], value[1])
    return (-value, value)


def _affine_axis_ranges(value: Any) -> dict[str, tuple[Any, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple | list):
        return {"x": _range(value[0]), "y": _range(value[1])}
    return {"x": _range(value), "y": _range(value)}


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image=image)["image"]


def create_transform(spec: TransformSpec) -> Any:
    """Create an AlbumentationsX transform from a TransformSpec."""

    builder = _TRANSFORM_BUILDERS.get(spec.name)
    if builder is None:
        raise ValueError(f"Unknown transform: {spec.name}")
    return builder(spec.params)


def _build_resize(params: dict[str, Any]) -> Any | None:
    return albumentations.Resize(
        height=params["target_size"],
        width=params["target_size"],
        interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
        p=1,
    )


def _build_random_crop224(params: dict[str, Any]) -> Any | None:
    return albumentations.RandomCrop(
        height=params["height"],
        width=params["width"],
        pad_if_needed=True,
        p=1,
    )


def _build_random_resized_crop(params: dict[str, Any]) -> Any | None:
    return albumentations.RandomResizedCrop(
        size=params["size"],
        scale=params["scale"],
        ratio=params["ratio"],
        interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
        p=1,
    )


def _build_horizontal_flip(_params: dict[str, Any]) -> Any | None:
    return albumentations.HorizontalFlip(p=1)


def _build_vertical_flip(_params: dict[str, Any]) -> Any | None:
    return albumentations.VerticalFlip(p=1)


def _build_pad(params: dict[str, Any]) -> Any | None:
    return albumentations.Pad(
        padding=params["padding"],
        fill=params["fill"],
        border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
        p=1,
    )


def _build_rotate(params: dict[str, Any]) -> Any | None:
    return albumentations.Rotate(
        angle_range=params["angle_range"],
        interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT if params["mode"] == "constant" else cv2.BORDER_REFLECT,
        fill=params["fill"],
        p=1,
    )


def _build_affine(params: dict[str, Any]) -> Any | None:
    return albumentations.Affine(
        rotate=_symmetric_range(params["angle"]),
        translate_px=_affine_axis_ranges(params["shift"]),
        scale=_range(params["scale"]),
        shear=_affine_axis_ranges(params["shear"]),
        interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT if params["mode"] == "constant" else cv2.BORDER_REFLECT,
        fill=params["fill"],
        p=1,
    )


def _build_perspective(params: dict[str, Any]) -> Any | None:
    return albumentations.Perspective(
        scale=params["scale"],
        interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
        p=1,
    )


def _build_elastic(params: dict[str, Any]) -> Any | None:
    return albumentations.ElasticTransform(
        alpha=params["alpha"],
        sigma=params["sigma"],
        interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
        approximate=params["approximate"],
        same_dxdy=params["same_dxdy"],
        p=1,
    )


def _build_color_jitter(params: dict[str, Any]) -> Any | None:
    return albumentations.ColorJitter(
        brightness_range=_range(params["brightness"]),
        contrast_range=_range(params["contrast"]),
        saturation_range=_range(params["saturation"]),
        hue_range=_range(params["hue"]),
        p=1,
    )


def _build_color_jiggle(params: dict[str, Any]) -> Any | None:
    return albumentations.ColorJitter(
        brightness_range=params["brightness"],
        contrast_range=params["contrast"],
        saturation_range=params["saturation"],
        hue_range=params["hue"],
        p=1,
    )


def _build_channel_shuffle(_params: dict[str, Any]) -> Any | None:
    return albumentations.ChannelShuffle(p=1)


def _build_grayscale(params: dict[str, Any]) -> Any | None:
    return albumentations.ToGray(num_output_channels=params["num_output_channels"], p=1)


def _build_rgbshift(params: dict[str, Any]) -> Any | None:
    shift = params["pixel_shift"]
    return albumentations.RGBShift(
        r_shift_range=_symmetric_range(shift),
        g_shift_range=_symmetric_range(shift),
        b_shift_range=_symmetric_range(shift),
        p=1,
    )


def _build_gaussian_blur(params: dict[str, Any]) -> Any | None:
    return albumentations.GaussianBlur(
        blur_range=params["kernel_size"],
        sigma_range=_range(params["sigma"]),
        p=1,
    )


def _build_gaussian_noise(params: dict[str, Any]) -> Any | None:
    return albumentations.GaussNoise(
        std_range=(params["std"], params["std"]),
        mean_range=(params["mean"], params["mean"]),
        per_channel=params["per_channel"],
        p=1,
    )


def _build_invert(_params: dict[str, Any]) -> Any | None:
    return albumentations.InvertImg(p=1)


def _build_posterize(params: dict[str, Any]) -> Any | None:
    return albumentations.Posterize(
        num_bits=_range(params["bits"]),
        p=1,
    )


def _build_solarize(params: dict[str, Any]) -> Any | None:
    return albumentations.Solarize(
        threshold_range=(params["threshold"], params["threshold"]),
        p=1,
    )


def _build_sharpen(params: dict[str, Any]) -> Any | None:
    return albumentations.Sharpen(
        alpha_range=params["alpha"],
        lightness_range=params["lightness"],
        p=1,
    )


def _build_auto_contrast(_params: dict[str, Any]) -> Any | None:
    return albumentations.AutoContrast(p=1, method="pil")


def _build_equalize(params: dict[str, Any]) -> Any | None:
    return albumentations.Equalize(mode=params["mode"], p=1)


def _build_normalize(_params: dict[str, Any]) -> Any | None:
    # Normalization is a batch-level CUDA stage.  Returning no CPU
    # transform here makes it impossible for an RGB recipe factory to
    # accidentally put it back into a DataLoader worker.
    return None


def _build_erasing(params: dict[str, Any]) -> Any | None:
    return albumentations.Erasing(
        scale=params["scale"],
        ratio=params["ratio"],
        p=1,
    )


def _build_jpeg_compression(params: dict[str, Any]) -> Any | None:
    return albumentations.ImageCompression(
        quality_range=(params["quality"], params["quality"]),
        p=1,
    )


def _build_random_gamma(params: dict[str, Any]) -> Any | None:
    return albumentations.RandomGamma(
        gamma_range=_range(params["gamma"]),
        p=1,
    )


def _build_plankian_jitter(params: dict[str, Any]) -> Any | None:
    return albumentations.PlanckianJitter(
        mode=params["mode"],
        p=1,
    )


def _build_median_blur(params: dict[str, Any]) -> Any | None:
    return albumentations.MedianBlur(
        blur_range=_range(params["blur_limit"]),
        p=1,
    )


def _build_motion_blur(params: dict[str, Any]) -> Any | None:
    return albumentations.MotionBlur(
        blur_range=_range(params["kernel_size"]),
        angle_range=params["angle_range"],
        direction_range=params["direction_range"],
        p=1,
    )


def _build_clahe(params: dict[str, Any]) -> Any | None:
    return albumentations.CLAHE(
        clip_range=params["clip_limit"],
        tile_grid_size=params["tile_grid_size"],
        p=1,
    )


def _build_brightness(params: dict[str, Any]) -> Any | None:
    return albumentations.RandomBrightnessContrast(
        brightness_range=params["brightness_limit"],
        contrast_range=(0.0, 0.0),
        p=1,
    )


def _build_contrast(params: dict[str, Any]) -> Any | None:
    return albumentations.RandomBrightnessContrast(
        brightness_range=(0.0, 0.0),
        contrast_range=params["contrast_limit"],
        p=1,
    )


def _build_blur(params: dict[str, Any]) -> Any | None:
    return albumentations.Blur(
        blur_range=_range(params["radius"]),
        p=1,
    )


def _build_channel_dropout(_params: dict[str, Any]) -> Any | None:
    return albumentations.ChannelDropout(p=1)


def _build_linear_illumination(_params: dict[str, Any]) -> Any | None:
    return albumentations.Illumination(p=1, mode="linear", angle_range=(90, 90))


def _build_corner_illumination(_params: dict[str, Any]) -> Any | None:
    return albumentations.Illumination(p=1, mode="corner")


def _build_gaussian_illumination(_params: dict[str, Any]) -> Any | None:
    return albumentations.Illumination(p=1, mode="gaussian")


def _build_hue(params: dict[str, Any]) -> Any | None:
    return albumentations.HueSaturationValue(
        hue_shift_range=_symmetric_range(params["hue"]),
        sat_shift_range=(0, 0),
        val_shift_range=(0, 0),
        p=1,
    )


def _build_plasma_brightness(params: dict[str, Any]) -> Any | None:
    return albumentations.PlasmaBrightnessContrast(p=1, roughness=params["roughness"], contrast_range=(0.0, 0.0))


def _build_plasma_contrast(params: dict[str, Any]) -> Any | None:
    return albumentations.PlasmaBrightnessContrast(p=1, roughness=params["roughness"], brightness_range=(0.0, 0.0))


def _build_plasma_shadow(params: dict[str, Any]) -> Any | None:
    return albumentations.PlasmaShadow(p=1, roughness=params["roughness"])


def _build_rain(params: dict[str, Any]) -> Any | None:
    return albumentations.RandomRain(
        p=1,
        drop_width=params["drop_width"],
        brightness_coefficient=params["brightness_coefficient"],
    )


def _build_salt_and_pepper(params: dict[str, Any]) -> Any | None:
    return albumentations.SaltAndPepper(
        p=1, amount_range=params["amount"], salt_vs_pepper_range=params["salt_vs_pepper"]
    )


def _build_saturation(params: dict[str, Any]) -> Any | None:
    sat_shift_range = _symmetric_range(params["saturation_factor"] * 255)
    return albumentations.HueSaturationValue(
        p=1,
        hue_shift_range=(0, 0),
        sat_shift_range=sat_shift_range,
        val_shift_range=(0, 0),
    )


def _build_snow(params: dict[str, Any]) -> Any | None:
    return albumentations.RandomSnow(p=1, snow_point_range=params["snow_point_range"])


def _build_optical_distortion(params: dict[str, Any]) -> Any | None:
    return albumentations.OpticalDistortion(
        p=1, distort_range=_symmetric_range(params["distort_limit"]), mode=params["mode"]
    )


def _build_shear(params: dict[str, Any]) -> Any | None:
    return albumentations.Affine(
        p=1,
        shear=_range(params["shear"]),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
    )


def _build_thin_plate_spline(params: dict[str, Any]) -> Any | None:
    return albumentations.ThinPlateSpline(
        p=1,
        num_control_points=params["num_control_points"],
        scale_range=(params["distortion_scale"], params["distortion_scale"]),
    )


def _build_photo_metric_distort(params: dict[str, Any]) -> Any | None:
    return albumentations.PhotoMetricDistort(
        brightness_range=params["brightness_range"],
        contrast_range=params["contrast_range"],
        saturation_range=params["saturation_range"],
        hue_range=params["hue_range"],
        distort_p=1.0,
        p=1,
    )


def _build_longest_max_size(params: dict[str, Any]) -> Any | None:
    return albumentations.LongestMaxSize(
        max_size=params["max_size"],
        interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
        p=1,
    )


def _build_smallest_max_size(params: dict[str, Any]) -> Any | None:
    return albumentations.SmallestMaxSize(
        max_size=params["max_size"],
        interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
        p=1,
    )


def _build_transpose(_params: dict[str, Any]) -> Any | None:
    return albumentations.Transpose(p=1)


def _build_random_rotate90(_params: dict[str, Any]) -> Any | None:
    return albumentations.RandomRotate90(p=1)


def _build_random_jigsaw(params: dict[str, Any]) -> Any | None:
    return albumentations.RandomGridShuffle(grid=params["grid"], p=1)


def _build_enhance_edge(params: dict[str, Any]) -> Any | None:
    return albumentations.Enhance(mode=params["mode"], alpha_range=params["alpha_range"], p=1)


def _build_unsharp_mask(params: dict[str, Any]) -> Any | None:
    return albumentations.UnsharpMask(
        blur_range=params["blur_limit"],
        sigma_range=_range(params["sigma_limit"]),
        alpha_range=params["alpha"],
        threshold=params["threshold"],
        p=1,
    )


_TRANSFORM_BUILDERS = {
    "Resize": _build_resize,
    "RandomCrop224": _build_random_crop224,
    "RandomResizedCrop": _build_random_resized_crop,
    "HorizontalFlip": _build_horizontal_flip,
    "VerticalFlip": _build_vertical_flip,
    "Pad": _build_pad,
    "Rotate": _build_rotate,
    "Affine": _build_affine,
    "Perspective": _build_perspective,
    "Elastic": _build_elastic,
    "ColorJitter": _build_color_jitter,
    "ColorJiggle": _build_color_jiggle,
    "ChannelShuffle": _build_channel_shuffle,
    "Grayscale": _build_grayscale,
    "RGBShift": _build_rgbshift,
    "GaussianBlur": _build_gaussian_blur,
    "GaussianNoise": _build_gaussian_noise,
    "Invert": _build_invert,
    "Posterize": _build_posterize,
    "Solarize": _build_solarize,
    "Sharpen": _build_sharpen,
    "AutoContrast": _build_auto_contrast,
    "Equalize": _build_equalize,
    "Normalize": _build_normalize,
    "Erasing": _build_erasing,
    "JpegCompression": _build_jpeg_compression,
    "RandomGamma": _build_random_gamma,
    "PlankianJitter": _build_plankian_jitter,
    "MedianBlur": _build_median_blur,
    "MotionBlur": _build_motion_blur,
    "CLAHE": _build_clahe,
    "Brightness": _build_brightness,
    "Contrast": _build_contrast,
    "Blur": _build_blur,
    "ChannelDropout": _build_channel_dropout,
    "LinearIllumination": _build_linear_illumination,
    "CornerIllumination": _build_corner_illumination,
    "GaussianIllumination": _build_gaussian_illumination,
    "Hue": _build_hue,
    "PlasmaBrightness": _build_plasma_brightness,
    "PlasmaContrast": _build_plasma_contrast,
    "PlasmaShadow": _build_plasma_shadow,
    "Rain": _build_rain,
    "SaltAndPepper": _build_salt_and_pepper,
    "Saturation": _build_saturation,
    "Snow": _build_snow,
    "OpticalDistortion": _build_optical_distortion,
    "Shear": _build_shear,
    "ThinPlateSpline": _build_thin_plate_spline,
    "PhotoMetricDistort": _build_photo_metric_distort,
    "LongestMaxSize": _build_longest_max_size,
    "SmallestMaxSize": _build_smallest_max_size,
    "Transpose": _build_transpose,
    "RandomRotate90": _build_random_rotate90,
    "RandomJigsaw": _build_random_jigsaw,
    "EnhanceEdge": _build_enhance_edge,
    "EnhanceDetail": _build_enhance_edge,
    "UnsharpMask": _build_unsharp_mask,
}
