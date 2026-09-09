from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from augbench.adapters.dali.native import _apply_erasing, _apply_pad

if TYPE_CHECKING:
    from numpy.typing import NDArray

dali = pytest.importorskip("nvidia.dali")
fn = pytest.importorskip("nvidia.dali.fn")


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
