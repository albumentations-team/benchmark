from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
import torchvision.transforms.v2 as tv_transforms
from torch import nn
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg

from augbench.implementations.recipe_stages import collatable_prefix, normalization_stats, transform_specs
from augbench.implementations.torchvision_impl import create_transform
from augbench.recipes.runtime import UnsupportedRecipeError

if TYPE_CHECKING:
    from augbench.recipes.models import RecipeSpec


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


class _GpuBatchRecipe(nn.Module):
    def __init__(self, recipe: RecipeSpec) -> None:
        super().__init__()
        mean, std = normalization_stats(recipe)
        self._batch_normalize = tv_transforms.Normalize(mean=mean, std=std, inplace=True)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.device.type != "cuda":
            raise RuntimeError("GPU normalization requires a CUDA batch")
        if batch.dtype == torch.uint8:
            batch = batch.to(dtype=torch.float16).div_(255.0)
        elif batch.is_floating_point():
            batch = batch.to(dtype=torch.float16)
        else:
            raise TypeError(f"TorchVision GPU transforms require uint8 or floating input, got {batch.dtype}")
        return self._batch_normalize(batch)


class _CudaRecipe(nn.Module):
    def __init__(
        self,
        *,
        prefix: nn.Module,
        tail: nn.Module | None,
        jpeg_quality: tuple[int, int] | None,
        recipe: RecipeSpec,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.tail = tail
        self.jpeg_quality = jpeg_quality
        self.normalize = _GpuBatchRecipe(recipe)

    def forward(self, encoded: list[bytes | torch.Tensor]) -> torch.Tensor:
        jpeg_tensors = [
            torch.frombuffer(bytearray(image), dtype=torch.uint8) if isinstance(image, bytes) else image
            for image in encoded
        ]
        images = decode_jpeg(jpeg_tensors, mode=ImageReadMode.RGB, device="cuda")
        batch = torch.stack([self.prefix(image) for image in images])
        if self.jpeg_quality is not None:
            batch = _jpeg_round_trip(batch, self.jpeg_quality)
        elif self.tail is not None:
            batch = batch.to(dtype=torch.float16).div_(255.0)
            batch = torch.stack([self.tail(image) for image in batch])
        return self.normalize(batch)


def _jpeg_round_trip(batch: torch.Tensor, quality_range: tuple[int, int]) -> torch.Tensor:
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(batch.shape[0]):
        quality = random.randint(*quality_range)  # noqa: S311 - augmentation RNG.
        groups[quality].append(index)
    order = []
    encoded = []
    for quality, indices in groups.items():
        encoded.extend(encode_jpeg([batch[index] for index in indices], quality=quality))
        order.extend(indices)
    sizes = [image.numel() for image in encoded]
    decoded = decode_jpeg(list(torch.cat(encoded).cpu().split(sizes)), mode=ImageReadMode.RGB, device=batch.device)
    return torch.stack([image for _, image in sorted(zip(order, decoded, strict=True))])


class _DeferredCpuRecipe(nn.Module):
    defer_batch_to_gpu = True

    def __init__(self, *, transforms: tuple[nn.Module, ...], recipe: RecipeSpec) -> None:
        super().__init__()
        self.cpu_transform = tv_transforms.Compose(list(transforms))
        self.gpu_transform = _GpuBatchRecipe(recipe)


def build_recipe(recipe: RecipeSpec, implementation_id: str) -> nn.Module:
    if implementation_id == "torchvision_cpu":
        transforms = tuple(_transform(spec, recipe.recipe_id) for spec in transform_specs(recipe))
        return _DeferredCpuRecipe(transforms=transforms, recipe=recipe)
    if implementation_id == "torchvision_gpu":
        prefix, tail = collatable_prefix(recipe)
        jpeg_quality = None
        if len(tail) == 1 and tail[0].name == "JpegCompression":
            low, high = tail[0].params["quality_range"]
            jpeg_quality = (low, high)
            tail = ()
        return _CudaRecipe(
            prefix=tv_transforms.Compose([_transform(spec, recipe.recipe_id) for spec in prefix]),
            tail=tv_transforms.Compose([_transform(spec, recipe.recipe_id) for spec in tail]) if tail else None,
            jpeg_quality=jpeg_quality,
            recipe=recipe,
        )
    raise UnsupportedRecipeError(f"{implementation_id!r} is not a TorchVision implementation")


def _transform(spec: Any, recipe_id: str) -> nn.Module:
    transform = create_transform(spec)
    if transform is None:
        raise UnsupportedRecipeError(f"TorchVision cannot build {spec.name!r} for {recipe_id!r}")
    return transform
