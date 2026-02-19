"""AugLy implementations of transforms for images in custom format."""

from typing import Any

import augly.image as imaugs

from benchmark.transforms.registry import build_transforms, register_library
from benchmark.transforms.specs import TransformSpec

# Required: Library name for dependency installation
LIBRARY = "augly"


# Required: Define how to apply transforms to images
def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    """Apply augly transform to a single image

    Args:
        transform: AugLy transform instance
        image: PIL Image

    Returns:
        Transformed image as PIL Image
    """
    # AugLy transforms may modify images in-place, so we copy
    return transform(image.copy())


# Helper function to create transforms from specs
def create_transform(spec: TransformSpec) -> Any | None:
    """Create an AugLy transform from a TransformSpec."""
    params = spec.params

    if spec.name == "Resize":
        return lambda img: imaugs.resize(
            img,
            width=params["target_size"],
            height=params["target_size"],
        )
    if spec.name == "HorizontalFlip":
        return imaugs.hflip
    if spec.name == "VerticalFlip":
        return imaugs.vflip
    if spec.name == "Rotate":
        return lambda img: imaugs.rotate(img, degrees=params["angle"])
    if spec.name == "ColorJitter":
        # AugLy doesn't have a direct ColorJitter, use brightness/contrast
        return lambda img: imaugs.brightness(
            imaugs.contrast(img, factor=1 + params["contrast"]),
            factor=1 + params["brightness"],
        )
    if spec.name == "Grayscale":
        return imaugs.grayscale
    if spec.name == "Blur":
        return lambda img: imaugs.blur(img, radius=params["radius"])
    if spec.name == "Brightness":
        brightness_val = params["brightness_limit"]
        if isinstance(brightness_val, tuple):
            brightness_val = brightness_val[0]  # Use first value if tuple
        return lambda img: imaugs.brightness(img, factor=1 + brightness_val)
    if spec.name == "Contrast":
        contrast_val = params["contrast_limit"]
        if isinstance(contrast_val, tuple):
            contrast_val = contrast_val[0]  # Use first value if tuple
        return lambda img: imaugs.contrast(img, factor=1 + contrast_val)
    if spec.name == "Saturation":
        return lambda img: imaugs.saturation(img, factor=params["saturation_factor"])
    if spec.name == "JpegCompression":
        return lambda img: imaugs.encoding_quality(img, quality=params["quality"])
    if spec.name == "RandomGamma":
        # AugLy's apply_lambda doesn't work well, skip this transform
        # Would use gamma = params["gamma"] / 100 if supported
        return None
    if spec.name == "Sharpen":
        alpha = params["alpha"]
        if isinstance(alpha, tuple):
            alpha = alpha[0]  # Use first value if tuple
        return lambda img: imaugs.sharpen(
            img,
            factor=alpha,
        )
    if spec.name == "Pad":
        # AugLy pad function has different API - use color instead of color_tuple
        return lambda img: imaugs.pad(
            img,
            w_factor=params["padding"] / img.width,
            h_factor=params["padding"] / img.height,
            color=(params["fill"], params["fill"], params["fill"]),
        )
    if spec.name == "Perspective":
        # Skip - augly's perspective_transform has a bug with deprecated np.float
        return None
    if spec.name in {"Equalize", "AutoContrast"}:
        # AugLy doesn't support these PIL filters properly, skip
        return None
    if spec.name == "Posterize":
        # AugLy doesn't have posterize, use quantize as approximation
        num_colors = 2 ** params["bits"]
        return lambda img: img.quantize(colors=num_colors).convert("RGB")
    if spec.name == "Invert":
        # AugLy doesn't support these PIL filters properly, skip
        return None
    if spec.name == "Solarize":
        # AugLy's apply_pil_filter doesn't accept 'p' parameter, skip
        return None
    # Skip transforms not supported by augly
    return None


# Register with the central registry
register_library(LIBRARY, create_image_fn=create_transform)

# Required: Transform definitions from specs
TRANSFORMS = build_transforms(LIBRARY, media="image")
