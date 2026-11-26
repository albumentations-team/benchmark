from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from augbench.implementations import pillow_impl
from augbench.implementations.randaugment import OPERATIONS, SIGNED, RandAugmentSpec

PARAMS: dict[str, Any] = {
    "num_ops": 2,
    "magnitude": 9,
    "num_magnitude_bins": 31,
    "interpolation": "bilinear",
    "fill": 0,
    "operations": list(OPERATIONS),
}
# Mean absolute difference, in 0-255 units, allowed between an assembled operation and TorchVision's.
TOLERANCE = 1.5


def _image() -> np.ndarray:
    y, x = np.mgrid[0:224, 0:224]
    base = np.stack([x * 255 / 223, y * 255 / 223, (x + y) * 255 / 446], axis=-1)
    texture = 30 * np.sin(x / 9.0)[..., None] * np.cos(y / 7.0)[..., None]
    noise = np.random.default_rng(0).normal(0, 8, base.shape)
    return np.clip(base + texture + noise, 0, 255).astype(np.uint8)


def _signs(name: str) -> tuple[float, ...]:
    return (1.0, -1.0) if name in SIGNED else (1.0,)


def test_the_spec_rejects_a_pool_or_interpolation_it_does_not_assemble() -> None:
    with pytest.raises(ValueError, match="pool"):
        RandAugmentSpec.from_params({**PARAMS, "operations": OPERATIONS[:-1]})
    with pytest.raises(ValueError, match="bilinear"):
        RandAugmentSpec.from_params({**PARAMS, "interpolation": "nearest"})
    with pytest.raises(ValueError, match="magnitude"):
        RandAugmentSpec.from_params({**PARAMS, "magnitude": 31})
    with pytest.raises(ValueError, match="magnitude"):
        RandAugmentSpec.from_params({**PARAMS, "magnitude": 0, "num_magnitude_bins": 1})


def test_constants_match_torchvision_at_the_benchmark_magnitude() -> None:
    transforms = pytest.importorskip("torchvision.transforms.v2")
    space = transforms.RandAugment()._AUGMENTATION_SPACE
    spec = RandAugmentSpec.from_params(PARAMS)

    def magnitude(name: str) -> float:
        return float(space[name][0](31, 224, 224)[9])

    assert spec.shear == pytest.approx(magnitude("ShearX"))
    assert spec.translate_fraction * 224 == pytest.approx(magnitude("TranslateX"), abs=1e-3)
    assert spec.rotate_degrees == pytest.approx(magnitude("Rotate"))
    assert spec.enhance == pytest.approx(magnitude("Brightness"))
    assert spec.posterize_bits == magnitude("Posterize")
    assert spec.solarize_fraction == pytest.approx(magnitude("Solarize"))


def test_pillow_draws_different_operations_for_different_images() -> None:
    transform = pillow_impl._build_rand_augment(PARAMS)
    from PIL import Image

    image = Image.fromarray(_image())
    outputs = set()
    for seed in range(8):
        random.seed(seed)
        outputs.add(np.asarray(transform(image)).tobytes())

    assert len(outputs) > 1


def test_ax_policy_is_one_some_of_of_stock_transforms() -> None:
    albumentations = pytest.importorskip("albumentations")
    from augbench.implementations import albumentationsx_impl

    policy = albumentationsx_impl._build_rand_augment(PARAMS)

    assert isinstance(policy, albumentations.SomeOf)
    assert policy.n == 2
    assert policy.replace is True
    assert len(policy.transforms) == len(OPERATIONS)
    for name, transform in zip(OPERATIONS, policy.transforms, strict=True):
        assert isinstance(transform, albumentations.OneOf) == (name in SIGNED), name
        assert not isinstance(transform, albumentations.Lambda)


@pytest.mark.parametrize("name", OPERATIONS)
def test_assembled_operations_match_torchvision_pixels(name: str) -> None:
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms.v2")
    from PIL import Image

    from augbench.implementations import albumentationsx_impl

    spec = RandAugmentSpec.from_params(PARAMS)
    reference = transforms.RandAugment(
        num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=transforms.InterpolationMode.BILINEAR, fill=0
    )
    space = reference._AUGMENTATION_SPACE
    bins = space[name][0](31, 224, 224)
    strength = float(bins[9]) if bins is not None and bins.numel() else 0.0
    image = _image()
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    pillow = pillow_impl._RandAugment(spec)

    for sign in _signs(name):
        expected = reference._apply_image_or_video_transform(
            tensor, name, strength * sign, transforms.InterpolationMode.BILINEAR, reference._fill
        )
        expected_pixels = expected.permute(1, 2, 0).numpy().astype(int)
        assembled = albumentationsx_impl._rand_augment_operation(name, spec, sign)(image=image)["image"]
        via_pillow = np.asarray(pillow._operations[name](Image.fromarray(image), sign))

        assert np.abs(assembled.astype(int) - expected_pixels).mean() <= TOLERANCE, ("AX", name, sign)
        assert np.abs(via_pillow.astype(int) - expected_pixels).mean() <= TOLERANCE, ("Pillow", name, sign)


def test_ax_policy_recipes_use_a_some_of_only_inside_the_rand_augment_stage() -> None:
    albumentations = pytest.importorskip("albumentations")
    from pathlib import Path

    from augbench.implementations.albumentationsx_pipeline_impl import build_recipe
    from augbench.recipes.load import load_recipe_catalog

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog/recipes/imagenet_train.yaml")
    for recipe in catalog.recipes:
        if "albumentationsx_cpu" not in recipe.supported_implementations:
            continue
        *stages, _to_tensor = build_recipe(recipe, "albumentationsx_cpu").cpu_transform.transforms
        some_of = [stage for stage in stages if isinstance(stage, albumentations.SomeOf)]
        assert len(some_of) == (1 if "RandAugment" in recipe.recipe_id else 0), recipe.recipe_id
        for stage in stages:
            assert not isinstance(stage, albumentations.Lambda | albumentations.Sequential), recipe.recipe_id
            if not isinstance(stage, albumentations.SomeOf):
                assert isinstance(stage, albumentations.BasicTransform), recipe.recipe_id


def test_kornia_policy_is_called_once_per_image() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from augbench.implementations import kornia_impl

    calls: list[tuple[int, ...]] = []

    class Policy(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            calls.append(tuple(image.shape))
            return image

    transform = kornia_impl._PerImageJitter(Policy())
    transform(torch.zeros(3, 3, 8, 8))

    assert calls == [(1, 3, 8, 8)] * 3
