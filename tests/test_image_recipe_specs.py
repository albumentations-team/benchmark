from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from augbench.implementations.pillow_impl import create_transform
from augbench.implementations.recipe_stages import normalization_stats, transform_specs
from augbench.implementations.specs import TransformSpec
from augbench.recipes.load import load_recipe_catalog


def test_recipe_catalog_is_the_single_source_of_runtime_parameters() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    resize = next(recipe for recipe in recipes.recipes if recipe.recipe_id == "Resize224+Normalize+ToTensor")

    assert [spec.name for spec in transform_specs(resize)] == ["Resize"]
    assert transform_specs(resize)[0].params["target_size"] == 224
    assert normalization_stats(resize) == ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def test_rgb_recipes_declare_a_float16_ready_gpu_batch() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")

    assert all(recipe.stages[-1].parameters["dtype"] == "float16" for recipe in recipes.recipes)


def test_r01_pillow_uses_bilinear_without_smoothing() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in recipes.recipes if recipe.augmentation_id == "Resize")
    spec = transform_specs(recipe)[0]
    resize = create_transform(TransformSpec(spec.name, {**spec.params, "target_size": 3}))
    assert resize is not None
    impulse = Image.frombytes("L", (5, 1), bytes([0, 0, 255, 0, 0]))
    assert resize(impulse).tobytes() == bytes([0, 255, 0]) * 3


def test_r46_pillow_centers_x_shear_and_interpolates_zero_border() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in recipes.recipes if recipe.augmentation_id == "Shear")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "Shear")
    shear = create_transform(TransformSpec(spec.name, {**spec.params, "shear": [10, 10]}))
    assert shear is not None

    pixels = shear(Image.new("L", (3, 3), 255)).tobytes()
    assert pixels == bytes([255, 255, 210, 255, 255, 255, 210, 255, 255])


def test_r63_pillow_reorders_channels_using_each_sampled_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    from augbench.implementations import pillow_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "ChannelSwap")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "ChannelSwap")
    swap = create_transform(spec)
    assert swap is not None

    picks = iter(range(3))
    monkeypatch.setattr(pillow_impl.random, "choice", lambda orders: orders[next(picks)])
    image = Image.new("RGB", (1, 1), (15, 99, 200))
    assert [swap(image).getpixel((0, 0)) for _ in range(3)] == [(99, 15, 200), (200, 99, 15), (15, 200, 99)]


@pytest.mark.parametrize("operation", ["Affine", "Shear", "ShiftScaleRotate", "RandomGamma", "Blur", "Emboss"])
def test_pillow_samples_parameters_during_each_transform(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from augbench.implementations import pillow_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == operation)
    spec = next(spec for spec in transform_specs(recipe) if spec.name == operation)
    transform = create_transform(spec)
    assert transform is not None

    image = Image.frombytes("RGB", (224, 224), bytes(range(256)) * 588)
    monkeypatch.setattr(pillow_impl.random, "uniform", lambda low, _high: low)
    monkeypatch.setattr(pillow_impl.random, "randint", lambda low, _high: low)
    monkeypatch.setattr(pillow_impl.random, "choice", lambda values: values[0])
    low = transform(image).tobytes()
    monkeypatch.setattr(pillow_impl.random, "uniform", lambda _low, high: high)
    monkeypatch.setattr(pillow_impl.random, "randint", lambda _low, high: high)
    monkeypatch.setattr(pillow_impl.random, "choice", lambda values: values[-1])
    high = transform(image).tobytes()

    assert low != high


@pytest.mark.parametrize("operation", ["Affine", "Shear", "ShiftScaleRotate", "OpticalDistortion", "Blur"])
def test_kornia_samples_within_a_batch(operation: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from augbench.implementations import kornia_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == operation)
    spec = next(spec for spec in transform_specs(recipe) if spec.name == operation)
    transform = kornia_impl.create_transform(spec)
    assert transform is not None

    image = torch.rand((1, 3, 64, 64), generator=torch.Generator().manual_seed(137))
    batch = image.repeat(8, 1, 1, 1)
    torch.manual_seed(137)
    first = transform(batch)
    torch.manual_seed(137)
    replay = transform(batch)

    torch.testing.assert_close(first, replay, atol=0, rtol=0)
    assert first.shape == batch.shape
    assert any(not torch.equal(first[0], sample) for sample in first[1:])


def test_kornia_elastic_samples_field_amplitude_per_image() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from augbench.implementations import kornia_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "Elastic")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "Elastic")
    variable = kornia_impl.create_transform(spec)
    high = spec.params["kornia_alpha"][1]
    fixed = kornia_impl.create_transform(TransformSpec(spec.name, {**spec.params, "kornia_alpha": [high, high]}))
    assert variable is not None
    assert fixed is not None

    torch.manual_seed(137)
    sampled_noise = variable.generate_parameters((8, 3, 64, 64))["noise"]
    torch.manual_seed(137)
    reference_noise = fixed.generate_parameters((8, 3, 64, 64))["noise"]
    ratios = sampled_noise.flatten(1).norm(dim=1) / reference_noise.flatten(1).norm(dim=1)

    torch.testing.assert_close(sampled_noise, reference_noise * ratios[:, None, None, None])
    assert torch.all((ratios >= 0.5) & (ratios <= 1.0))
    assert ratios.unique().numel() > 1


def test_rain_kornia_matches_ax_coverage_and_intensity() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    pytest.importorskip("albumentations")
    from augbench.implementations import albumentationsx_impl, kornia_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "Rain")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "Rain")
    ax = albumentationsx_impl.create_transform(spec)
    kornia = kornia_impl.create_transform(spec)
    ax.set_random_seed(137)
    torch.manual_seed(137)
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    ax_level = np.stack([ax(image=black)["image"][..., 0] for _ in range(8)]).astype(np.float32) / 255
    kornia_level = kornia(torch.zeros((8, 3, 224, 224)))[:, 0].numpy()

    assert (kornia_level > 0).mean() == pytest.approx((ax_level > 0).mean(), abs=0.05)
    assert kornia_level.mean() == pytest.approx(ax_level.mean(), abs=0.025)


def test_kornia_tps_matches_ax_anchor_sampling() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from augbench.implementations import kornia_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "ThinPlateSpline")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "ThinPlateSpline")
    transform = kornia_impl.create_transform(spec)
    torch.manual_seed(137)
    params = transform.generate_parameters((128, 3, 224, 224))

    expected = torch.tensor([[0, 0], [0, 224], [224, 0], [224, 224], [112, 112]], dtype=torch.float32)
    fixed_pixels = (params["dst"] + 1) * 112 - 0.5
    torch.testing.assert_close(fixed_pixels, expected.expand_as(fixed_pixels), atol=2e-5, rtol=0)
    offsets_px = 112 * (params["src"] - params["dst"])
    assert offsets_px.std().item() == pytest.approx(11.2, abs=1)
