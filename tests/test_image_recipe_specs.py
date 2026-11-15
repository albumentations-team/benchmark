from pathlib import Path

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
    shear = create_transform(spec)
    assert shear is not None

    pixels = shear(Image.new("L", (3, 3), 255)).tobytes()
    assert pixels == bytes([255, 255, 210, 255, 255, 255, 210, 255, 255])
