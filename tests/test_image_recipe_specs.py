from importlib import import_module
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
    resize = next(recipe for recipe in recipes.recipes if recipe.recipe_id == "Resize224")

    assert [spec.name for spec in transform_specs(resize)] == ["Resize"]
    assert transform_specs(resize)[0].params["target_size"] == 224
    assert normalization_stats(resize) == ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def test_rgb_recipes_declare_a_float16_ready_gpu_batch() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")

    assert all(recipe.stages[-1].parameters["dtype"] == "float16" for recipe in recipes.recipes)


def test_r09_pillow_matches_native_ax_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    albumentations = pytest.importorskip("albumentations")
    from augbench.implementations import albumentationsx_impl, geometry

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "Perspective")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "Perspective")
    sigma = 0.075
    spec = TransformSpec(spec.name, {**spec.params, "scale": [sigma, sigma]})
    reference = albumentationsx_impl.create_transform(spec)
    pillow = create_transform(spec)
    assert isinstance(reference, albumentations.Perspective)
    assert pillow is not None

    seed = 137
    offsets = iter(np.random.default_rng(seed).normal(0, sigma, size=8))
    reference.set_random_seed(seed)
    monkeypatch.setattr(geometry.random, "gauss", lambda _mean, _sigma: next(offsets))
    y, x = np.indices((224, 224), dtype=np.uint8)
    image = np.stack((x, y, x ^ y), axis=-1)
    expected = reference(image=image)["image"]
    actual = np.asarray(pillow(Image.fromarray(image)))

    np.testing.assert_allclose(actual, expected, atol=2, rtol=0)


def test_r01_pillow_uses_bilinear_without_smoothing() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in recipes.recipes if recipe.augmentation_id == "Resize")
    spec = transform_specs(recipe)[0]
    resize = create_transform(TransformSpec(spec.name, {**spec.params, "target_size": 3}))
    assert resize is not None
    impulse = Image.frombytes("L", (5, 1), bytes([0, 0, 255, 0, 0]))
    assert resize(impulse).tobytes() == bytes([0, 255, 0]) * 3


@pytest.mark.parametrize(
    ("size", "box"),
    [
        ((100, 32), (28, 0, 71, 32)),
        ((32, 100), (0, 28, 32, 71)),
        ((64, 64), (0, 0, 64, 64)),
    ],
)
def test_pillow_random_resized_crop_falls_back_to_centered_aspect_ratio(
    size: tuple[int, int], box: tuple[int, int, int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from augbench.implementations import pillow_pipeline_impl

    monkeypatch.setattr(pillow_pipeline_impl.random, "uniform", lambda _low, high: high)
    image = Image.fromarray(np.arange(size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0]))
    crop = pillow_pipeline_impl._RandomResizedCrop(resize=lambda patch: patch, scale=(1, 1), ratio=(0.75, 4 / 3))

    np.testing.assert_array_equal(np.asarray(crop(image)), np.asarray(image.crop(box)))


def test_pillow_gaussian_blur_respects_the_configured_kernel_size() -> None:
    image = Image.new("RGB", (21, 21))
    image.putpixel((10, 10), (255, 255, 255))
    blur = create_transform(TransformSpec("GaussianBlur", {"sigma": [2, 2], "kernel_size": [5, 5]}))
    assert blur is not None

    result = blur(image)
    assert result.getbbox() == (8, 8, 13, 13)


def test_pillow_unsharp_mask_matches_ax_three_tap_edge_strength() -> None:
    image = np.full((32, 32, 3), 64, dtype=np.uint8)
    image[:, 16:] = 192
    spec = TransformSpec("UnsharpMask", {"blur_limit": [3, 3], "alpha": [0.5, 0.5], "threshold": 10})
    sharpen = create_transform(spec)
    assert sharpen is not None

    edge = np.asarray(sharpen(Image.fromarray(image)))[16, 14:18, 0]
    # The AX 3-tap Gaussian puts 1/4 of the 128-level step across the edge.
    np.testing.assert_allclose(edge, [64, 48, 208, 192], atol=2, rtol=0)


def test_r46_pillow_centers_x_shear_and_interpolates_zero_border() -> None:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in recipes.recipes if recipe.augmentation_id == "Shear")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "Shear")
    shear = create_transform(TransformSpec(spec.name, {**spec.params, "shear": [10, 10]}))
    assert shear is not None

    pixels = shear(Image.new("L", (3, 3), 255)).tobytes()
    assert pixels == bytes([255, 255, 210, 255, 255, 255, 210, 255, 255])


def test_r63_pillow_uses_the_fixed_ax_channel_order() -> None:
    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "RGB <-> BGR")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "ChannelSwap")
    swap = create_transform(spec)
    assert swap is not None

    image = Image.new("RGB", (1, 1), (15, 99, 200))
    assert swap(image).getpixel((0, 0)) == (200, 99, 15)


def test_ax_recipe_stages_are_stock_transforms() -> None:
    albumentations = pytest.importorskip("albumentations")
    from augbench.implementations.albumentationsx_pipeline_impl import build_recipe

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    for recipe in catalog.recipes:
        if "albumentationsx_cpu" not in recipe.supported_implementations:
            continue
        pipeline = build_recipe(recipe, "albumentationsx_cpu").cpu_transform
        assert len(pipeline.transforms) == len(transform_specs(recipe)), recipe.recipe_id
        for transform in pipeline.transforms:
            assert isinstance(transform, albumentations.BasicTransform), (recipe.recipe_id, type(transform))
            assert type(transform).__module__.startswith("albumentations."), (recipe.recipe_id, type(transform))
            assert type(transform).__name__ != "Lambda", recipe.recipe_id


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


@pytest.mark.parametrize(
    "operation", ["Affine", "Shear", "ShiftScaleRotate", "OpticalDistortion", "Perspective", "Blur"]
)
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


def test_kornia_rain_uses_native_streaks_with_calibrated_density() -> None:
    torch = pytest.importorskip("torch")
    kornia = pytest.importorskip("kornia")
    from augbench.implementations import kornia_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "Rain")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "Rain")
    transform = kornia_impl.create_transform(spec)
    assert type(transform) is kornia.augmentation.RandomRain
    torch.manual_seed(137)
    result = transform(torch.zeros((8, 3, 224, 224)))[:, 0]

    assert (result > 0).float().mean().item() == pytest.approx(0.47, abs=0.03)
    assert not torch.equal(result[0], result[1])


def test_kornia_tps_matches_ax_anchor_strength() -> None:
    torch = pytest.importorskip("torch")
    kornia = pytest.importorskip("kornia")
    from augbench.implementations import kornia_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "ThinPlateSpline")
    spec = next(spec for spec in transform_specs(recipe) if spec.name == "ThinPlateSpline")
    transform = kornia_impl.create_transform(spec)
    assert type(transform) is kornia.augmentation.RandomThinPlateSpline
    torch.manual_seed(137)
    params = transform.generate_parameters((128, 3, 224, 224))

    offsets_px = 112 * (params["src"] - params["dst"])
    assert offsets_px.std().item() == pytest.approx(11.2, abs=1)


def test_kornia_snow_maps_ax_threshold_to_native_coefficient() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from augbench.implementations import kornia_impl

    params = {"snow_point_range": [0.2, 0.2], "brightness": 2.0}
    transform = kornia_impl.create_transform(TransformSpec("Snow", params))
    assert transform is not None
    image = torch.full((1, 3, 4, 4), 100 / 255)
    torch.testing.assert_close(transform(image), 2 * image)


@pytest.mark.parametrize("operation", ["RandomCrop224", "CenterCrop", "PadIfNeeded"])
def test_kornia_native_crop_and_padding_match_ax_on_small_images(operation: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("kornia")
    pytest.importorskip("albumentations")
    from augbench.implementations import albumentationsx_impl, kornia_impl

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == operation)
    spec = transform_specs(recipe)[0]
    image = np.full((173, 189, 3), 255, dtype=np.uint8)
    ax = albumentationsx_impl.create_transform(spec)
    kornia = kornia_impl.create_transform(spec)
    assert kornia is not None

    expected = ax(image=image)["image"]
    actual = kornia(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div_(255))
    np.testing.assert_array_equal(actual[0].permute(1, 2, 0).numpy() * 255, expected)


@pytest.mark.parametrize("library", ["torchvision", "kornia"])
def test_perspective_maps_sampled_inner_points_to_the_output_canvas(
    library: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip(library)
    from augbench.implementations import geometry

    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    spec = next(transform_specs(recipe)[-1] for recipe in catalog.recipes if recipe.augmentation_id == "Perspective")
    adapter = import_module(f"augbench.implementations.{library}_impl")
    transform = adapter.create_transform(spec)
    monkeypatch.setattr(geometry.random, "gauss", lambda _mean, _sigma: 0.08)
    torch.manual_seed(137)
    image = torch.ones((3, 224, 224))

    result = transform(image.unsqueeze(0) if library == "kornia" else image)

    assert result.mean() > 0.99
