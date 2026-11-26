from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from augbench.adapters.dali.native import (
    _apply_erasing,
    _apply_pad,
    _apply_pixel_spread,
    _apply_stage,
    _decode_recipe_input,
)
from augbench.recipes.load import load_recipe_catalog
from augbench.recipes.models import RecipeStage

if TYPE_CHECKING:
    from numpy.typing import NDArray

dali = pytest.importorskip("nvidia.dali")
fn = pytest.importorskip("nvidia.dali.fn")


@pytest.mark.parametrize("operation", ["RandomCrop224", "RandomResizedCrop"])
def test_decode_crop_samples_different_windows_from_identical_jpegs(operation: str) -> None:
    pil = pytest.importorskip("PIL.Image")
    y, x = np.indices((160, 168))
    image = np.stack((x, y, (x + 2 * y) % 256), axis=-1).astype(np.uint8)
    with io.BytesIO() as stream:
        pil.fromarray(image).save(stream, format="JPEG", quality=95)
        encoded = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    stage = RecipeStage(
        operation_id=operation,
        parameters={
            "height": 64,
            "width": 64,
            "size": [64, 64],
            "scale": [0.2, 0.6],
            "ratio": [0.75, 1.33],
            "interpolation": "bilinear",
            "antialias": False,
        },
        stochastic=True,
        randomness_scope="per-image",
    )
    with dali.Pipeline(batch_size=16, num_threads=1, device_id=None, seed=137) as pipeline:
        data = fn.external_source(source=lambda: [encoded] * 16)
        cropped, remaining = _decode_recipe_input(data, (stage,), device="cpu")
        assert not remaining
        pipeline.set_outputs(cropped)
    pipeline.build()
    (output,) = pipeline.run()
    assert all(output.at(index).shape == (64, 64, 3) for index in range(16))
    assert len({output.at(index).tobytes() for index in range(16)}) >= 8


def test_pad_adds_a_border_to_each_image() -> None:
    images = [np.full(shape, 255, dtype=np.uint8) for shape in ((80, 120, 3), (256, 320, 3), (320, 256, 3))]
    with dali.Pipeline(batch_size=len(images), num_threads=1, device_id=None) as pipeline:
        data = fn.external_source(source=lambda: images, layout="HWC")
        pipeline.set_outputs(_apply_pad(data, {"padding": 10, "fill": 17}, 224, 224))
    pipeline.build()
    (output,) = pipeline.run()

    for index, image in enumerate(images):
        expected = np.pad(image, ((10, 10), (10, 10), (0, 0)), constant_values=17)
        np.testing.assert_array_equal(output.at(index), expected)


def test_dali_pixel_spread_shares_offsets_across_rgb_and_blacks_outside_pixels() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("DALI remap requires CUDA")

    height, width, radius, samples = 32, 32, 2, 4
    y, x = np.indices((height, width), dtype=np.int16) + 1
    image = np.stack((x, y, x + y), axis=-1).astype(np.uint8)
    with dali.Pipeline(batch_size=samples, num_threads=1, device_id=0, seed=137) as pipeline:
        data = fn.external_source(source=lambda: [image] * samples, layout="HWC").gpu()
        pipeline.set_outputs(_apply_pixel_spread(data, {"radius": radius}, height, width))
    pipeline.build()
    (device_output,) = pipeline.run()
    output = device_output.as_cpu().as_array()

    valid = output[..., 0] > 0
    assert np.any(~valid)
    assert np.all(valid[:, radius:-radius, radius:-radius])
    assert np.all(output[~valid] == 0)
    np.testing.assert_array_equal(output[..., 2], output[..., 0] + output[..., 1])
    assert np.all(np.abs(output[..., 0] - x)[valid] <= radius)
    assert np.all(np.abs(output[..., 1] - y)[valid] <= radius)
    assert not np.array_equal(output[0], output[1])


@pytest.mark.parametrize(
    "operation",
    [
        "Affine",
        "Shear",
        "ShiftScaleRotate",
        "Blur",
        "GaussianBlur",
        "Emboss",
        "Brightness",
        "Contrast",
        "Saturation",
        "Posterize",
        "Solarize",
        "RandomGamma",
        "JpegCompression",
    ],
)
def test_dali_samples_within_a_batch(operation: str) -> None:
    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog" / "recipes" / "rgb.yaml")
    recipe = next(recipe for recipe in catalog.recipes if recipe.augmentation_id == operation)
    stage = next(stage for stage in recipe.stages if stage.operation_id == operation)
    image = np.random.default_rng(137).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    with dali.Pipeline(batch_size=16, num_threads=1, device_id=None, seed=137) as pipeline:
        data = fn.external_source(source=lambda: [image] * 16, layout="HWC")
        pipeline.set_outputs(_apply_stage(data, stage, output_height=224, output_width=224))
    pipeline.build()
    (output,) = pipeline.run()

    assert all(output.at(index).shape == image.shape for index in range(16))
    assert all(output.at(index).dtype == image.dtype for index in range(16))
    assert len({output.at(index).tobytes() for index in range(16)}) > 1


@pytest.mark.parametrize(("scale", "ratio"), [((0.02, 0.33), (0.3, 3.3)), ((0.12, 0.13), (1.9, 2.0))])
def test_erasing_samples_configured_regions_reproducibly(
    scale: tuple[float, float], ratio: tuple[float, float]
) -> None:
    regions = _erased_regions(seed=137, scale=scale, ratio=ratio)
    np.testing.assert_array_equal(regions, _erased_regions(seed=137, scale=scale, ratio=ratio))
    assert not np.array_equal(regions, _erased_regions(seed=138, scale=scale, ratio=ratio))
    assert len(np.unique(regions, axis=0)) > 1

    heights, widths = regions[:, 2], regions[:, 3]
    areas = heights * widths / (224 * 224)
    aspects = widths / heights
    assert np.all((areas >= scale[0] - 0.005) & (areas <= scale[1] + 0.005))
    assert np.all((aspects >= ratio[0] - 0.03) & (aspects <= ratio[1] + 0.03))


def _erased_regions(*, seed: int, scale: tuple[float, float], ratio: tuple[float, float]) -> NDArray[np.int64]:
    images = [np.full((224, 224, 3), 255, dtype=np.uint8) for _ in range(32)]
    with dali.Pipeline(batch_size=len(images), num_threads=1, device_id=None, seed=seed) as pipeline:
        data = fn.external_source(source=lambda: images, layout="HWC")
        pipeline.set_outputs(_apply_erasing(data, {"scale": list(scale), "ratio": list(ratio), "fill": 17}, 224, 224))
    pipeline.build()
    (output,) = pipeline.run()

    regions = []
    for index, image in enumerate(images):
        actual = output.at(index)
        rows, columns = np.nonzero(actual[:, :, 0] == 17)
        top, bottom = rows.min(), rows.max() + 1
        left, right = columns.min(), columns.max() + 1
        expected = image.copy()
        expected[top:bottom, left:right] = 17
        np.testing.assert_array_equal(actual, expected)
        regions.append((top, left, bottom - top, right - left))
    return np.asarray(regions, dtype=np.int64)
