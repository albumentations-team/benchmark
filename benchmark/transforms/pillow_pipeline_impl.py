from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

from PIL import Image
from torchvision.transforms import PILToTensor

from benchmark.transforms.image_recipe_specs import (
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    recipe_augmentation_specs,
    recipe_name,
    spec_by_name,
)
from benchmark.transforms.pillow_impl import create_transform

if TYPE_CHECKING:
    import torch

LIBRARY = "pillow"
_PIL_TO_TENSOR = PILToTensor()


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


def _to_normalized_chw_tensor(image: Image.Image) -> torch.Tensor:
    tensor = _PIL_TO_TENSOR(image).float().div_(255.0)
    mean = tensor.new_tensor(NORMALIZE_MEAN).view(-1, 1, 1)
    std = tensor.new_tensor(NORMALIZE_STD).view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)


def _pad_to_min_size(image: Image.Image, min_width: int, min_height: int) -> Image.Image:
    width, height = image.size
    padded_width = max(width, min_width)
    padded_height = max(height, min_height)
    if (padded_width, padded_height) == (width, height):
        return image

    padded = Image.new(image.mode, (padded_width, padded_height), color=0)
    left = (padded_width - width) // 2
    top = (padded_height - height) // 2
    padded.paste(image, (left, top))
    return padded


class _PillowCropRecipe:
    def __init__(self, augmentation: Any | None = None) -> None:
        self.augmentation = augmentation

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = self._crop(image)
        if self.augmentation is not None:
            image = self.augmentation(image)
        return _to_normalized_chw_tensor(image)

    @staticmethod
    def _crop(image: Image.Image) -> Image.Image:
        params = spec_by_name("RandomCrop224").params
        crop_width = params["width"]
        crop_height = params["height"]
        image = _pad_to_min_size(image, crop_width, crop_height)
        width, height = image.size
        left = random.randint(0, width - crop_width)  # noqa: S311
        top = random.randint(0, height - crop_height)  # noqa: S311
        return image.crop((left, top, left + crop_width, top + crop_height))


class _PillowRandomResizedCropRecipe:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        params = spec_by_name("RandomResizedCrop").params
        size = params["size"]
        width, height = image.size
        area = width * height
        log_ratio = tuple(math.log(value) for value in params["ratio"])

        for _ in range(10):
            target_area = area * random.uniform(*params["scale"])  # noqa: S311
            aspect_ratio = math.exp(random.uniform(*log_ratio))  # noqa: S311
            crop_width = round(math.sqrt(target_area * aspect_ratio))
            crop_height = round(math.sqrt(target_area / aspect_ratio))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = random.randint(0, width - crop_width)  # noqa: S311
                top = random.randint(0, height - crop_height)  # noqa: S311
                cropped = image.crop((left, top, left + crop_width, top + crop_height))
                return _to_normalized_chw_tensor(cropped.resize(size, Image.Resampling.BILINEAR))

        in_ratio = width / height
        min_ratio, max_ratio = params["ratio"]
        if in_ratio < min_ratio:
            crop_width = width
            crop_height = round(crop_width / min_ratio)
        elif in_ratio > max_ratio:
            crop_height = height
            crop_width = round(crop_height * max_ratio)
        else:
            crop_width = width
            crop_height = height
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return _to_normalized_chw_tensor(cropped.resize(size, Image.Resampling.BILINEAR))


class _PillowCenterCropRecipe:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        params = spec_by_name("CenterCrop224").params
        crop_width = params["width"]
        crop_height = params["height"]
        image = _pad_to_min_size(image, crop_width, crop_height)
        width, height = image.size
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))
        return _to_normalized_chw_tensor(image)


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
