from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image

from benchmark.transforms.image_recipe_specs import (
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    recipe_augmentation_specs,
    recipe_name,
    spec_by_name,
)
from benchmark.transforms.pillow_impl import create_transform

LIBRARY = "pillow"


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


def _to_normalized_chw(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(NORMALIZE_MEAN, dtype=np.float32)) / np.asarray(NORMALIZE_STD, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


class _PillowCropRecipe:
    def __init__(self, augmentation: Any | None = None) -> None:
        self.augmentation = augmentation

    def __call__(self, image: Image.Image) -> np.ndarray:
        image = self._crop(image)
        if self.augmentation is not None:
            image = self.augmentation(image)
        return _to_normalized_chw(image)

    @staticmethod
    def _crop(image: Image.Image) -> Image.Image:
        params = spec_by_name("RandomCrop224").params
        crop_width = params["width"]
        crop_height = params["height"]
        width, height = image.size
        if width < crop_width or height < crop_height:
            image = image.resize((max(width, crop_width), max(height, crop_height)), Image.BILINEAR)
            width, height = image.size
        left = random.randint(0, width - crop_width)  # noqa: S311
        top = random.randint(0, height - crop_height)  # noqa: S311
        return image.crop((left, top, left + crop_width, top + crop_height))


class _PillowRandomResizedCropRecipe:
    def __call__(self, image: Image.Image) -> np.ndarray:
        params = spec_by_name("RandomResizedCrop").params
        size = params["size"]
        image = image.resize(size, Image.BILINEAR)
        return _to_normalized_chw(image)


class _PillowCenterCropRecipe:
    def __call__(self, image: Image.Image) -> np.ndarray:
        params = spec_by_name("CenterCrop224").params
        crop_width = params["width"]
        crop_height = params["height"]
        width, height = image.size
        if width < crop_width or height < crop_height:
            image = image.resize((max(width, crop_width), max(height, crop_height)), Image.BILINEAR)
            width, height = image.size
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))
        return _to_normalized_chw(image)


def _create_transform(name: str) -> Any | None:
    if name == "RandomCrop224":
        return _PillowCropRecipe()
    if name == "RandomResizedCrop":
        return _PillowRandomResizedCropRecipe()
    if name == "CenterCrop224":
        return _PillowCenterCropRecipe()
    spec = spec_by_name(name)
    augmentation = create_transform(spec)
    return None if augmentation is None else _PillowCropRecipe(augmentation)


TRANSFORMS = [
    {"name": recipe_name(spec), "transform": transform}
    for spec in recipe_augmentation_specs(3)
    if (transform := _create_transform(spec.name)) is not None
]
