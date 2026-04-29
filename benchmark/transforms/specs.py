"""Transform specifications shared across augmentation libraries.

PARAM CONVERSION RULES (library-specific):
When implementations reinterpret spec params, conversion rules live here. Keep in sync
when changing specs.

  Brightness: spec brightness_limit is additive offset (e.g. 0.2). Kornia/TorchVision
    use multiplicative factor -> (1.0 + offset, 1.0 + offset). AlbumentationsX uses
    additive directly.

  Contrast: same as Brightness; spec contrast_limit additive -> Kornia multiplicative.

  Saturation: spec saturation_factor is additive-style offset (0.5). Kornia uses
    multiplicative -> (1.0 + factor, 1.0 + factor). AlbumentationsX uses sat_shift_limit.

  Hue: spec hue in degrees (e.g. 20). Kornia expects fraction of 360° in [-0.5, 0.5]
    -> hue_degrees / 360.0. AlbumentationsX uses degrees directly.

  Affine: spec shift in pixels (for reference_size images). TorchVision translate is
    fraction of image size -> shift_px / reference_size. Kornia/Albumentations use
    pixels directly (translate_px). reference_size in spec controls the conversion.

  RandomGamma: spec gamma as int (e.g. 120). Kornia expects float -> gamma / 100.

  RGBShift: spec pixel_shift in [0,255]. Kornia expects normalized [0,1] -> /255.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TransformSpec:
    """Base class that defines exact parameters for each transform"""

    name: str
    params: dict[str, Any]

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"


# Define all transform specifications based on the original implementations
TRANSFORM_SPECS = [
    TransformSpec(
        "Resize",
        {
            "target_size": 512,
            "interpolation": "bilinear",
        },
    ),
    TransformSpec(
        "RandomCrop128",
        {
            "height": 128,
            "width": 128,
        },
    ),
    TransformSpec(
        "RandomResizedCrop",
        {
            "size": (512, 512),
            "width": 512,
            "scale": (0.08, 1.0),
            "ratio": (0.75, 1.3333333333333333),
            "interpolation": "bilinear",
        },
    ),
    TransformSpec(
        "CenterCrop128",
        {
            "height": 128,
            "width": 128,
        },
    ),
    TransformSpec(
        "HorizontalFlip",
        {},
    ),
    TransformSpec(
        "VerticalFlip",
        {},
    ),
    TransformSpec(
        "Pad",
        {
            "padding": 10,
            "fill": 0,
            "border_mode": "constant",
        },
    ),
    TransformSpec(
        "Rotate",
        {
            "angle_range": (-45, 45),
            "interpolation": "nearest",
            "mode": "constant",
            "fill": 0,
        },
    ),
    TransformSpec(
        "Affine",
        {
            "angle": 25.0,
            "shift": (20, 20),
            "scale": 2.0,
            "shear": (10.0, 15.0),
            "interpolation": "bilinear",
            "mode": "constant",
            "fill": 0,
            # reference_size: image size (px) for which shift is defined; used to convert
            # pixel shift -> fraction (e.g. torchvision). Set to your typical input size.
            "reference_size": 512,
        },
    ),
    TransformSpec(
        "Perspective",
        {
            "scale": (0.05, 0.1),
            "interpolation": "bilinear",
            "fill": 0,
        },
    ),
    TransformSpec(
        "Elastic",
        {
            "alpha": 50.0,
            "sigma": 5.0,
            "interpolation": "bilinear",
            "approximate": False,
            "same_dxdy": True,
        },
    ),
    TransformSpec(
        "ColorJitter",
        {
            "brightness": 0.5,
            "contrast": 1.5,
            "saturation": 1.5,
            "hue": 0.5,
        },
    ),
    TransformSpec(
        "ChannelShuffle",
        {},
    ),
    TransformSpec(
        "Grayscale",
        {
            "num_output_channels": 3,
        },
    ),
    TransformSpec(
        "RGBShift",
        {
            "pixel_shift": 100,
        },
    ),
    TransformSpec(
        "GaussianBlur",
        {
            "sigma": 2.0,
            "kernel_size": (5, 5),
        },
    ),
    TransformSpec(
        "GaussianNoise",
        {
            "mean": 0,
            "std": 0.44,
            "per_channel": True,
        },
    ),
    TransformSpec(
        "Invert",
        {},
    ),
    TransformSpec(
        "Posterize",
        {
            "bits": 4,
        },
    ),
    TransformSpec(
        "Solarize",
        {
            "threshold": 0.5,
        },
    ),
    TransformSpec(
        "Sharpen",
        {
            "alpha": (0.2, 0.5),
            "lightness": (0.5, 1),
        },
    ),
    TransformSpec(
        "AutoContrast",
        {},
    ),
    TransformSpec(
        "Equalize",
        {
            "mode": "pil",
        },
    ),
    TransformSpec(
        "Normalize",
        {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
    ),
    TransformSpec(
        "Erasing",
        {
            "scale": (0.02, 0.33),
            "ratio": (0.3, 3.3),
            "fill": 0,
        },
    ),
    TransformSpec(
        "JpegCompression",
        {
            "quality": 50,
        },
    ),
    TransformSpec(
        "RandomGamma",
        {
            "gamma": 120,
        },
    ),
    TransformSpec(
        "PlankianJitter",
        {
            "mode": "blackbody",
        },
    ),
    TransformSpec(
        "MedianBlur",
        {
            "blur_limit": 5,
        },
    ),
    TransformSpec(
        "MotionBlur",
        {
            "kernel_size": 5,
            "angle_range": (0, 360),
            "direction_range": (-1, 1),
        },
    ),
    TransformSpec(
        "CLAHE",
        {
            "clip_limit": (1, 4),
            "tile_grid_size": (8, 8),
        },
    ),
    TransformSpec(
        "Brightness",
        {
            "brightness_limit": (0.2, 0.2),
        },
    ),
    TransformSpec(
        "Contrast",
        {
            "contrast_limit": (0.2, 0.2),
        },
    ),
    TransformSpec(
        "CoarseDropout",
        {
            "hole_height_range": (0.1, 0.1),
            "hole_width_range": (0.1, 0.1),
            "num_holes_range": (4, 4),
        },
    ),
    TransformSpec(
        "Blur",
        {
            "radius": 5,
            "border_mode": "constant",
        },
    ),
    TransformSpec(
        "HSV",
        {
            "hue": 0.015,
            "saturation": 0.7,
            "value": 0.4,
        },
    ),
    TransformSpec(
        "ChannelDropout",
        {},
    ),
    TransformSpec(
        "LinearIllumination",
        {
            "gain": (0.01, 0.2),
        },
    ),
    TransformSpec(
        "CornerIllumination",
        {
            "gain": (0.01, 0.2),
        },
    ),
    TransformSpec(
        "GaussianIllumination",
        {
            "gain": (0.01, 0.2),
        },
    ),
    TransformSpec(
        "Hue",
        {
            "hue": 20,
        },
    ),
    TransformSpec(
        "PlasmaBrightness",
        {
            "roughness": 0.5,
        },
    ),
    TransformSpec(
        "PlasmaContrast",
        {
            "roughness": 0.5,
        },
    ),
    TransformSpec(
        "PlasmaShadow",
        {
            "roughness": 0.5,
        },
    ),
    TransformSpec(
        "Rain",
        {
            "drop_width": 5,
            "drop_height": 20,
            "brightness_coefficient": 1.0,
        },
    ),
    TransformSpec(
        "SaltAndPepper",
        {
            "amount": (0.01, 0.06),
            "salt_vs_pepper": (0.4, 0.6),
        },
    ),
    TransformSpec(
        "Saturation",
        {
            "saturation_factor": 0.5,
        },
    ),
    TransformSpec(
        "Snow",
        {"snow_point_range": (0.5, 0.5)},
    ),
    TransformSpec(
        "OpticalDistortion",
        {
            "distort_limit": 0.5,
            "mode": "fisheye",
        },
    ),
    TransformSpec(
        "Shear",
        {
            "shear": 10,
        },
    ),
    TransformSpec(
        "ThinPlateSpline",
        {
            "num_control_points": 2,
            "distortion_scale": 0.5,
        },
    ),
    TransformSpec(
        "PhotoMetricDistort",
        {
            "brightness_range": (0.875, 1.125),
            "contrast_range": (0.5, 1.5),
            "saturation_range": (0.5, 1.5),
            "hue_range": (-0.05, 0.05),
        },
    ),
    TransformSpec(
        "ColorJiggle",
        {
            "brightness": (0.875, 1.125),
            "contrast": (0.5, 1.5),
            "saturation": (0.5, 1.5),
            "hue": (-0.05, 0.05),
        },
    ),
    TransformSpec(
        "LongestMaxSize",
        {
            "max_size": 512,
            "interpolation": "bilinear",
        },
    ),
    TransformSpec(
        "SmallestMaxSize",
        {
            "max_size": 512,
            "interpolation": "bilinear",
        },
    ),
    # -----------------------------------------------------------------------
    # Additional transforms for AlbumentationsX vs Albumentations comparison
    # -----------------------------------------------------------------------
    # Blur
    TransformSpec(
        "AdvancedBlur",
        {
            "blur_limit": (3, 7),
            "sigma_x_limit": (0.2, 1.0),
            "sigma_y_limit": (0.2, 1.0),
        },
    ),
    TransformSpec(
        "Defocus",
        {
            "radius": (3, 7),
            "alias_blur": (0.1, 0.5),
        },
    ),
    TransformSpec(
        "ZoomBlur",
        {
            "max_factor": (1.05, 1.31),
        },
    ),
    TransformSpec(
        "GlassBlur",
        {
            "sigma": 0.7,
            "max_delta": 2,
            "iterations": 2,
        },
    ),
    TransformSpec(
        "ModeFilter",
        {
            "kernel_range": (3, 7),
        },
    ),
    # Geometric
    TransformSpec(
        "SquareSymmetry",
        {},
    ),
    TransformSpec(
        "Transpose",
        {},
    ),
    TransformSpec(
        "SafeRotate",
        {
            "limit": 45,
            "interpolation": "bilinear",
            "border_mode": "constant",
            "fill": 0,
        },
    ),
    TransformSpec(
        "RandomRotate90",
        {
            "times": (0, 3),
        },
    ),
    TransformSpec(
        "RandomScale",
        {
            "scale_limit": (-0.1, 0.1),
            "interpolation": "bilinear",
        },
    ),
    TransformSpec(
        "ShiftScaleRotate",
        {
            "shift_limit": 0.0625,
            "scale_limit": 0.1,
            "rotate_limit": 45,
            "interpolation": "bilinear",
            "border_mode": "constant",
            "fill": 0,
        },
    ),
    TransformSpec(
        "GridDistortion",
        {
            "num_steps": 5,
            "distort_limit": 0.3,
            "interpolation": "bilinear",
            "border_mode": "constant",
            "fill": 0,
        },
    ),
    TransformSpec(
        "PiecewiseAffine",
        {
            "scale": (0.03, 0.05),
            "nb_rows": 4,
            "nb_cols": 4,
        },
    ),
    TransformSpec(
        "RandomGridShuffle",
        {
            "grid": (3, 3),
        },
    ),
    TransformSpec(
        "RandomJigsaw",
        {
            "grid": (4, 4),
        },
    ),
    TransformSpec(
        "Morphological",
        {
            "scale": (2, 3),
            "operation": "dilation",
        },
    ),
    # Pixel-level
    TransformSpec(
        "Downscale",
        {
            "scale_range": (0.25, 0.25),
            "interpolation_pair": ("nearest", "nearest"),
        },
    ),
    TransformSpec(
        "Colorize",
        {
            "black_range": ((0, 0, 0), (0, 0, 0)),
            "white_range": ((255, 255, 255), (255, 255, 255)),
            "mid_range": ((96, 64, 32), (96, 64, 32)),
            "mid_value_range": (127, 127),
        },
    ),
    TransformSpec(
        "PixelSpread",
        {
            "radius": 2,
            "interpolation": "nearest",
            "border_mode": "reflect101",
            "fill": 0,
        },
    ),
    TransformSpec(
        "EnhanceEdge",
        {
            "mode": "edge",
            "alpha_range": (0.5, 1.0),
        },
    ),
    TransformSpec(
        "EnhanceDetail",
        {
            "mode": "detail",
            "alpha_range": (0.5, 1.0),
        },
    ),
    TransformSpec(
        "Emboss",
        {
            "alpha": (0.2, 0.5),
            "strength": (0.2, 0.7),
        },
    ),
    TransformSpec(
        "ChromaticAberration",
        {
            "primary_distortion_limit": (-0.02, 0.02),
            "secondary_distortion_limit": (-0.05, 0.05),
            "mode": "green_purple",
        },
    ),
    TransformSpec(
        "ISONoise",
        {
            "color_shift": (0.01, 0.05),
            "intensity": (0.1, 0.5),
        },
    ),
    TransformSpec(
        "ShotNoise",
        {
            "scale_range": (0.05, 0.15),
        },
    ),
    TransformSpec(
        "MultiplicativeNoise",
        {
            "multiplier": (0.9, 1.1),
            "per_channel": True,
        },
    ),
    TransformSpec(
        "AdditiveNoise",
        {
            "noise_type": "uniform",
            "spatial_mode": "per_pixel",
            "scale_range": (0.02, 0.1),
        },
    ),
    TransformSpec(
        "RandomFog",
        {
            "fog_coef_range": (0.3, 0.5),
            "alpha_coef": 0.08,
        },
    ),
    TransformSpec(
        "RandomShadow",
        {
            "num_shadows_limit": (1, 2),
            "shadow_dimension": 5,
        },
    ),
    TransformSpec(
        "RandomSunFlare",
        {
            "flare_roi": (0, 0, 1, 0.5),
            "num_flare_circles_range": (6, 10),
        },
    ),
    TransformSpec(
        "RandomToneCurve",
        {
            "scale": 0.1,
        },
    ),
    TransformSpec(
        "RingingOvershoot",
        {
            "blur_limit": (7, 15),
            "cutoff": (0.7854, 1.5708),
        },
    ),
    TransformSpec(
        "Spatter",
        {
            "mean": 0.65,
            "std": 0.3,
            "gauss_sigma": 2,
            "intensity": 0.6,
            "cutout_threshold": 0.68,
            "mode": "rain",
        },
    ),
    TransformSpec(
        "UnsharpMask",
        {
            "blur_limit": (3, 7),
            "sigma_limit": 0.0,
            "alpha": (0.2, 0.5),
            "threshold": 10,
        },
    ),
    TransformSpec(
        "FancyPCA",
        {
            "alpha": 0.1,
        },
    ),
    TransformSpec(
        "Superpixels",
        {
            "p_replace": (0.1, 0.1),
            "n_segments": (100, 100),
        },
    ),
    TransformSpec(
        "ToSepia",
        {},
    ),
    TransformSpec(
        "RandomGravel",
        {
            "gravel_roi": (0.1, 0.4, 0.9, 0.9),
            "number_of_patches": 2,
        },
    ),
    # Dropout
    TransformSpec(
        "GridDropout",
        {
            "ratio": 0.5,
            "unit_size_range": (2, 10),
            "holes_number_xy": None,
            "random_offset": False,
        },
    ),
    TransformSpec(
        "PixelDropout",
        {
            "dropout_prob": 0.01,
            "per_channel": False,
            "drop_value": 0,
        },
    ),
    TransformSpec(
        "ConstrainedCoarseDropout",
        {
            "num_holes_range": (1, 3),
            "hole_height_range": (0.1, 0.2),
            "hole_width_range": (0.1, 0.2),
        },
    ),
    # Pad / Crop
    TransformSpec(
        "PadIfNeeded",
        {
            "min_height": 1024,
            "min_width": 1024,
            "border_mode": "constant",
            "fill": 0,
        },
    ),
    TransformSpec(
        "CropAndPad",
        {
            "px": (-10, 20, -10, 20),
            "border_mode": "constant",
            "fill": 0,
            "keep_size": True,
            "interpolation": "bilinear",
        },
    ),
    TransformSpec(
        "RandomSizedCrop",
        {
            "min_max_height": (256, 480),
            "size": (512, 512),
            "interpolation": "bilinear",
        },
    ),
    # AlbumentationsX-only transforms (will fail gracefully for albumentations_mit)
    TransformSpec(
        "AtmosphericFog",
        {
            "density_range": (0.8, 1.8),
            "depth_mode": "linear",
        },
    ),
    TransformSpec(
        "Vignetting",
        {
            "intensity_range": (0.2, 0.5),
        },
    ),
    TransformSpec(
        "Dithering",
        {
            "method": "error_diffusion",
            "n_colors": 2,
            "color_mode": "grayscale",
            "error_diffusion_algorithm": "floyd_steinberg",
        },
    ),
    TransformSpec(
        "FilmGrain",
        {
            "intensity_range": (0.1, 0.3),
            "grain_size_range": (2, 4),
        },
    ),
    TransformSpec(
        "Halftone",
        {
            "dot_size_range": (4, 10),
            "blend_range": (0.0, 0.35),
        },
    ),
    TransformSpec(
        "LensFlare",
        {},
    ),
    TransformSpec(
        "ChannelSwap",
        {},
    ),
    TransformSpec(
        "GridMask",
        {
            "num_grid_range": (3, 7),
            "line_width_range": (0.2, 0.5),
            "rotation_range": (0, 0),
        },
    ),
    TransformSpec(
        "CopyAndPaste",
        {
            "min_visibility_after_paste": 0.05,
            "blend_mode": "hard",
            "blend_sigma_range": (1.0, 3.0),
            "scale_range": (1.0, 1.0),
            "min_paste_area": 1,
        },
    ),
    TransformSpec(
        "WaterRefraction",
        {},
    ),
]

# Use the same TRANSFORM_SPECS for both images and videos
# No need for separate VIDEO_TRANSFORM_SPECS
