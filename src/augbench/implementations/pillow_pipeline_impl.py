from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

from PIL import Image

from augbench.implementations.gpu_normalize import GpuBatchNormalize
from augbench.implementations.pillow_impl import create_transform
from augbench.implementations.recipe_stages import normalization_stats, transform_specs
from augbench.recipes.runtime import UnsupportedRecipeError

if TYPE_CHECKING:
    import torch

    from augbench.implementations.specs import TransformSpec
    from augbench.recipes.models import RecipeSpec

_PAIR_LENGTH = 2


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


def _pil_to_tensor() -> Any:
    from torchvision.transforms import PILToTensor

    return PILToTensor()


def _pad_to_min_size(image: Image.Image, min_width: int, min_height: int) -> Image.Image:
    width, height = image.size
    padded_width = max(width, min_width)
    padded_height = max(height, min_height)
    if (padded_width, padded_height) == (width, height):
        return image
    padded = Image.new(image.mode, (padded_width, padded_height), color=0)
    padded.paste(image, ((padded_width - width) // 2, (padded_height - height) // 2))
    return padded


class _RandomCrop:
    def __init__(self, *, height: int, width: int) -> None:
        self._height = height
        self._width = width

    def __call__(self, image: Image.Image) -> Image.Image:
        image = _pad_to_min_size(image, self._width, self._height)
        width, height = image.size
        left = random.randint(0, width - self._width)  # noqa: S311
        top = random.randint(0, height - self._height)  # noqa: S311
        return image.crop((left, top, left + self._width, top + self._height))


class _RandomResizedCrop:
    def __init__(self, *, size: tuple[int, int], scale: tuple[float, float], ratio: tuple[float, float]) -> None:
        self._size = size
        self._scale = scale
        self._ratio = ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        area = width * height
        log_ratio = tuple(math.log(value) for value in self._ratio)
        for _ in range(10):
            target_area = area * random.uniform(*self._scale)  # noqa: S311
            aspect_ratio = math.exp(random.uniform(*log_ratio))  # noqa: S311
            crop_width = round(math.sqrt(target_area * aspect_ratio))
            crop_height = round(math.sqrt(target_area / aspect_ratio))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = random.randint(0, width - crop_width)  # noqa: S311
                top = random.randint(0, height - crop_height)  # noqa: S311
                cropped = image.crop((left, top, left + crop_width, top + crop_height))
                return cropped.resize(self._size, Image.Resampling.BILINEAR)
        return image.resize(self._size, Image.Resampling.BILINEAR)


class _PillowRecipe:
    def __init__(self, transforms: tuple[Any, ...]) -> None:
        self._transforms = transforms

    def __call__(self, image: Image.Image) -> torch.Tensor:
        for transform in self._transforms:
            image = transform(image)
        return _pil_to_tensor()(image)


class _DeferredRecipe:
    defer_batch_to_gpu = True

    def __init__(self, *, cpu_transform: Any, recipe: RecipeSpec) -> None:
        mean, std = normalization_stats(recipe)
        self.cpu_transform = cpu_transform
        self.gpu_transform = GpuBatchNormalize(mean=mean, std=std, input_layout="BCHW")


def build_recipe(recipe: RecipeSpec, implementation_id: str) -> _DeferredRecipe:
    if implementation_id != "pillow_cpu":
        raise UnsupportedRecipeError(f"{implementation_id!r} is not a Pillow CPU implementation")
    transforms = tuple(_transform(spec, recipe.recipe_id) for spec in transform_specs(recipe))
    return _DeferredRecipe(cpu_transform=_PillowRecipe(transforms), recipe=recipe)


def _transform(spec: TransformSpec, recipe_id: str) -> Any:
    if spec.name == "RandomCrop224":
        return _RandomCrop(height=int(spec.params["height"]), width=int(spec.params["width"]))
    if spec.name == "RandomResizedCrop":
        size = _pair(spec.params["size"], cast=int, parameter="size")
        scale = _pair(spec.params["scale"], cast=float, parameter="scale")
        ratio = _pair(spec.params["ratio"], cast=float, parameter="ratio")
        return _RandomResizedCrop(size=size, scale=scale, ratio=ratio)
    transform = create_transform(spec)
    if transform is None:
        raise UnsupportedRecipeError(f"Pillow cannot build {spec.name!r} for {recipe_id!r}")
    return transform


def _pair(value: Any, *, cast: Any, parameter: str) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != _PAIR_LENGTH:
        raise ValueError(f"Pillow {parameter} must contain exactly two values")
    return cast(value[0]), cast(value[1])
