# benchmark/transforms/imgaug_impl.py

from typing import Any

import cv2
import imgaug.augmenters as iaa
import numpy as np

from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec

# Ensure single thread
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Required: Library name for dependency installation
LIBRARY = "imgaug"


# Required: Define how to apply transforms to images
def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    """Apply imgaug transform to a single image

    Args:
        transform: ImgAug augmenter instance
        image: numpy array of shape (H, W, C)

    Returns:
        Transformed image as numpy array
    """
    # ImgAug expects a list but we pass single image
    augmented = transform(images=[image])
    return np.ascontiguousarray(augmented[0])


# Helper function to create transforms from specs
def create_transform(spec: TransformSpec) -> Any | None:
    """Create an ImgAug transform from a TransformSpec."""
    params = spec.params

    if spec.name == "Resize":
        return iaa.Resize(
            size=params["target_size"],
            interpolation="linear" if params["interpolation"] == "bilinear" else "nearest",
        )
    if spec.name == "RandomCrop128":
        return iaa.CropToFixedSize(
            width=params["width"],
            height=params["height"],
        )
    if spec.name == "CenterCrop128":
        return iaa.CropToFixedSize(
            width=params["width"],
            height=params["height"],
            position="center",
        )
    if spec.name == "HorizontalFlip":
        return iaa.Fliplr(1.0)  # Always flip
    if spec.name == "VerticalFlip":
        return iaa.Flipud(1.0)  # Always flip
    if spec.name == "Pad":
        return iaa.Pad(
            px=params["padding"],
            pad_mode="constant" if params["border_mode"] == "constant" else "reflect",
            pad_cval=params["fill"],
        )
    if spec.name == "Rotate":
        return iaa.Rotate(
            rotate=params["angle"],
            mode="constant" if params["mode"] == "constant" else "reflect",
            cval=params["fill"],
        )
    if spec.name == "Affine":
        return iaa.Affine(
            rotate=params["angle"],
            translate_px={"x": params["shift"][0], "y": params["shift"][1]},
            scale=params["scale"],
            shear=params["shear"],
            mode="constant" if params["mode"] == "constant" else "reflect",
            cval=params["fill"],
        )
    if spec.name == "Perspective":
        return iaa.PerspectiveTransform(
            scale=params["scale"],
            mode="replicate",  # imgaug only supports 'replicate' or 'constant'
            cval=params["fill"],
        )
    if spec.name == "Elastic":
        return iaa.ElasticTransformation(
            alpha=params["alpha"],
            sigma=params["sigma"],
            mode="linear" if params["interpolation"] == "bilinear" else "nearest",
        )
    if spec.name == "ColorJitter":
        return iaa.Sequential(
            [
                iaa.LinearContrast((1 - params["contrast"], 1 + params["contrast"])),
                iaa.AddToBrightness(
                    (-int(params["brightness"] * 255), int(params["brightness"] * 255)),
                ),
                iaa.AddToSaturation(
                    (int(-params["saturation"] * 100), int(params["saturation"] * 100)),
                ),
                iaa.AddToHue((-int(params["hue"] * 179), int(params["hue"] * 179))),
            ],
            random_order=True,
        )
    if spec.name == "Grayscale":
        return iaa.Grayscale(alpha=1.0)
    if spec.name == "GaussianBlur":
        return iaa.GaussianBlur(sigma=(params["sigma"], params["sigma"]))
    if spec.name == "GaussianNoise":
        return iaa.AdditiveGaussianNoise(
            loc=params["mean"] * 255,
            scale=params["std"] * 255,
            per_channel=params["per_channel"],
        )
    if spec.name == "Invert":
        return iaa.Invert(p=1.0)
    if spec.name == "Posterize":
        return iaa.Posterize(nb_bits=params["bits"])
    if spec.name == "Solarize":
        return iaa.Solarize(p=1.0, threshold=params["threshold"])
    if spec.name == "Sharpen":
        return iaa.Sharpen(alpha=params["alpha"], lightness=params["lightness"][0])
    if spec.name == "AutoContrast":
        return iaa.AllChannelsCLAHE(clip_limit=40, tile_grid_size_px=8)
    if spec.name == "Equalize":
        return iaa.AllChannelsHistogramEqualization()
    if spec.name == "JpegCompression":
        return iaa.JpegCompression(compression=100 - params["quality"])
    if spec.name == "RandomGamma":
        gamma = params["gamma"] / 100
        return iaa.GammaContrast(gamma=gamma)
    if spec.name == "MedianBlur":
        return iaa.MedianBlur(k=params["blur_limit"])
    if spec.name == "MotionBlur":
        return iaa.MotionBlur(
            k=params["kernel_size"],
            angle=params["angle_range"],
            direction=params["direction_range"],
        )
    if spec.name == "CLAHE":
        return iaa.CLAHE(
            clip_limit=params["clip_limit"],
            tile_grid_size_px=params["tile_grid_size"],
        )
    if spec.name == "Brightness":
        brightness_val = params["brightness_limit"]
        if isinstance(brightness_val, tuple):
            brightness_val = brightness_val[0]  # Use first value if tuple
        return iaa.AddToBrightness(
            (int(-brightness_val * 255), int(brightness_val * 255)),
        )
    if spec.name == "Contrast":
        contrast_val = params["contrast_limit"]
        if isinstance(contrast_val, tuple):
            contrast_val = contrast_val[0]  # Use first value if tuple
        return iaa.LinearContrast((1 - contrast_val, 1 + contrast_val))
    if spec.name == "CoarseDropout":
        # imgaug CoarseDropout uses size_percent for hole sizes
        # Convert pixel sizes to percentages (assuming 224x224 images)
        size_percent = (
            params["hole_height_range"][0] / 224.0,
            params["hole_height_range"][1] / 224.0,
        )
        return iaa.CoarseDropout(
            p=1.0,  # Always apply
            size_percent=size_percent,
        )
    # Skip transforms not supported by imgaug
    return None


# Required: Transform definitions from specs
TRANSFORMS = [
    {
        "name": spec.name,
        "transform": create_transform(spec),
    }
    for spec in TRANSFORM_SPECS
    if create_transform(spec) is not None
]
