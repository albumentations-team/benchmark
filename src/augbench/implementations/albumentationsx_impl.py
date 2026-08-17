"""AlbumentationsX transforms used by the active RGB recipe catalog."""

from typing import Any

import albumentations as A
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
    params = spec.params

    if spec.name == "Resize":
        return A.Resize(
            height=params["target_size"],
            width=params["target_size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "RandomCrop224":
        return A.RandomCrop(
            height=params["height"],
            width=params["width"],
            pad_if_needed=True,
            p=1,
        )
    if spec.name == "RandomResizedCrop":
        return A.RandomResizedCrop(
            size=params["size"],
            scale=params["scale"],
            ratio=params["ratio"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "HorizontalFlip":
        return A.HorizontalFlip(p=1)
    if spec.name == "VerticalFlip":
        return A.VerticalFlip(p=1)
    if spec.name == "Pad":
        return A.Pad(
            padding=params["padding"],
            fill=params["fill"],
            border_mode=cv2.BORDER_CONSTANT if params["border_mode"] == "constant" else cv2.BORDER_REFLECT,
            p=1,
        )
    if spec.name == "Rotate":
        return A.Rotate(
            angle_range=params["angle_range"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT if params["mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "Affine":
        return A.Affine(
            rotate=_symmetric_range(params["angle"]),
            translate_px=_affine_axis_ranges(params["shift"]),
            scale=_range(params["scale"]),
            shear=_affine_axis_ranges(params["shear"]),
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT if params["mode"] == "constant" else cv2.BORDER_REFLECT,
            fill=params["fill"],
            p=1,
        )
    if spec.name == "Perspective":
        return A.Perspective(
            scale=params["scale"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "Elastic":
        return A.ElasticTransform(
            alpha=params["alpha"],
            sigma=params["sigma"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            approximate=params["approximate"],
            same_dxdy=params["same_dxdy"],
            p=1,
        )
    if spec.name == "ColorJitter":
        return A.ColorJitter(
            brightness_range=_range(params["brightness"]),
            contrast_range=_range(params["contrast"]),
            saturation_range=_range(params["saturation"]),
            hue_range=_range(params["hue"]),
            p=1,
        )
    if spec.name == "ColorJiggle":
        return A.ColorJitter(
            brightness_range=params["brightness"],
            contrast_range=params["contrast"],
            saturation_range=params["saturation"],
            hue_range=params["hue"],
            p=1,
        )
    if spec.name == "ChannelShuffle":
        return A.ChannelShuffle(p=1)
    if spec.name == "Grayscale":
        return A.ToGray(num_output_channels=params["num_output_channels"], p=1)
    if spec.name == "RGBShift":
        shift = params["pixel_shift"]
        return A.RGBShift(
            r_shift_range=_symmetric_range(shift),
            g_shift_range=_symmetric_range(shift),
            b_shift_range=_symmetric_range(shift),
            p=1,
        )
    if spec.name == "GaussianBlur":
        return A.GaussianBlur(
            blur_range=params["kernel_size"],
            sigma_range=_range(params["sigma"]),
            p=1,
        )
    if spec.name == "GaussianNoise":
        return A.GaussNoise(
            std_range=(params["std"], params["std"]),
            mean_range=(params["mean"], params["mean"]),
            per_channel=params["per_channel"],
            p=1,
        )
    if spec.name == "Invert":
        return A.InvertImg(p=1)
    if spec.name == "Posterize":
        return A.Posterize(
            num_bits=_range(params["bits"]),
            p=1,
        )
    if spec.name == "Solarize":
        return A.Solarize(
            threshold_range=(params["threshold"], params["threshold"]),
            p=1,
        )
    if spec.name == "Sharpen":
        return A.Sharpen(
            alpha_range=params["alpha"],
            lightness_range=params["lightness"],
            p=1,
        )
    if spec.name == "AutoContrast":
        return A.AutoContrast(p=1, method="pil")
    if spec.name == "Equalize":
        return A.Equalize(mode=params["mode"], p=1)
    if spec.name == "Normalize":
        # Normalization is a batch-level CUDA stage.  Returning no CPU
        # transform here makes it impossible for an RGB recipe factory to
        # accidentally put it back into a DataLoader worker.
        return None
    if spec.name == "Erasing":
        return A.Erasing(
            scale=params["scale"],
            ratio=params["ratio"],
            p=1,
        )
    if spec.name == "JpegCompression":
        return A.ImageCompression(
            quality_range=(params["quality"], params["quality"]),
            p=1,
        )
    if spec.name == "RandomGamma":
        return A.RandomGamma(
            gamma_range=_range(params["gamma"]),
            p=1,
        )
    if spec.name == "PlankianJitter":
        return A.PlanckianJitter(
            mode=params["mode"],
            p=1,
        )
    if spec.name == "MedianBlur":
        return A.MedianBlur(
            blur_range=_range(params["blur_limit"]),
            p=1,
        )
    if spec.name == "MotionBlur":
        return A.MotionBlur(
            blur_range=_range(params["kernel_size"]),
            angle_range=params["angle_range"],
            direction_range=params["direction_range"],
            p=1,
        )
    if spec.name == "CLAHE":
        return A.CLAHE(
            clip_range=params["clip_limit"],
            tile_grid_size=params["tile_grid_size"],
            p=1,
        )
    if spec.name == "Brightness":
        return A.RandomBrightnessContrast(
            brightness_range=params["brightness_limit"],
            contrast_range=(0.0, 0.0),
            p=1,
        )
    if spec.name == "Contrast":
        return A.RandomBrightnessContrast(
            brightness_range=(0.0, 0.0),
            contrast_range=params["contrast_limit"],
            p=1,
        )
    if spec.name == "Blur":
        return A.Blur(
            blur_range=_range(params["radius"]),
            p=1,
        )
    if spec.name == "ChannelDropout":
        return A.ChannelDropout(p=1)
    if spec.name == "LinearIllumination":
        return A.Illumination(p=1, mode="linear", angle_range=(90, 90))
    if spec.name == "CornerIllumination":
        return A.Illumination(p=1, mode="corner")
    if spec.name == "GaussianIllumination":
        return A.Illumination(p=1, mode="gaussian")
    if spec.name == "Hue":
        return A.HueSaturationValue(
            hue_shift_range=_symmetric_range(params["hue"]),
            sat_shift_range=(0, 0),
            val_shift_range=(0, 0),
            p=1,
        )
    if spec.name == "PlasmaBrightness":
        return A.PlasmaBrightnessContrast(p=1, roughness=params["roughness"], contrast_range=(0.0, 0.0))
    if spec.name == "PlasmaContrast":
        return A.PlasmaBrightnessContrast(p=1, roughness=params["roughness"], brightness_range=(0.0, 0.0))
    if spec.name == "PlasmaShadow":
        return A.PlasmaShadow(p=1, roughness=params["roughness"])
    if spec.name == "Rain":
        return A.RandomRain(
            p=1,
            drop_width=params["drop_width"],
            brightness_coefficient=params["brightness_coefficient"],
        )
    if spec.name == "SaltAndPepper":
        return A.SaltAndPepper(p=1, amount_range=params["amount"], salt_vs_pepper_range=params["salt_vs_pepper"])
    if spec.name == "Saturation":
        sat_shift_range = _symmetric_range(params["saturation_factor"] * 255)
        return A.HueSaturationValue(
            p=1,
            hue_shift_range=(0, 0),
            sat_shift_range=sat_shift_range,
            val_shift_range=(0, 0),
        )
    if spec.name == "Snow":
        return A.RandomSnow(p=1, snow_point_range=params["snow_point_range"])
    if spec.name == "OpticalDistortion":
        return A.OpticalDistortion(p=1, distort_range=_symmetric_range(params["distort_limit"]), mode=params["mode"])
    if spec.name == "Shear":
        return A.Affine(
            p=1,
            shear=_range(params["shear"]),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        )
    if spec.name == "ThinPlateSpline":
        return A.ThinPlateSpline(
            p=1,
            num_control_points=params["num_control_points"],
            scale_range=(params["distortion_scale"], params["distortion_scale"]),
        )
    if spec.name == "PhotoMetricDistort":
        return A.PhotoMetricDistort(
            brightness_range=params["brightness_range"],
            contrast_range=params["contrast_range"],
            saturation_range=params["saturation_range"],
            hue_range=params["hue_range"],
            distort_p=1.0,
            p=1,
        )
    if spec.name == "LongestMaxSize":
        return A.LongestMaxSize(
            max_size=params["max_size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "SmallestMaxSize":
        return A.SmallestMaxSize(
            max_size=params["max_size"],
            interpolation=cv2.INTER_LINEAR if params["interpolation"] == "bilinear" else cv2.INTER_NEAREST,
            p=1,
        )
    if spec.name == "Transpose":
        return A.Transpose(p=1)
    if spec.name == "RandomRotate90":
        return A.RandomRotate90(p=1)
    if spec.name == "RandomJigsaw":
        return A.RandomGridShuffle(grid=params["grid"], p=1)
    if spec.name in {"EnhanceEdge", "EnhanceDetail"}:
        return A.Enhance(mode=params["mode"], alpha_range=params["alpha_range"], p=1)
    if spec.name == "UnsharpMask":
        return A.UnsharpMask(
            blur_range=params["blur_limit"],
            sigma_range=_range(params["sigma_limit"]),
            alpha_range=params["alpha"],
            threshold=params["threshold"],
            p=1,
        )
    raise ValueError(f"Unknown transform: {spec.name}")
