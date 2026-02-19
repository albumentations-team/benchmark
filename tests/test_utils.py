"""Tests for benchmark/utils.py pure utility functions."""

from __future__ import annotations

import pytest

# Need to clear the @cache between tests so parametrize works correctly
import benchmark.utils as utils_module
from benchmark.utils import (
    get_image_loader,
    get_system_info,
    get_video_loader,
    is_variance_stable,
    time_transform,
)


@pytest.fixture(autouse=True)
def clear_loader_cache() -> None:
    utils_module.get_image_loader.cache_clear()
    utils_module.get_video_loader.cache_clear()


class TestGetSystemInfo:
    def test_returns_dict(self) -> None:
        info = get_system_info()
        assert isinstance(info, dict)

    def test_required_keys(self) -> None:
        info = get_system_info()
        for key in ("python_version", "platform", "processor", "cpu_count", "timestamp"):
            assert key in info, f"Missing key: {key}"

    def test_all_values_are_strings(self) -> None:
        info = get_system_info()
        for key, value in info.items():
            assert isinstance(value, str), f"Key {key!r} has non-string value: {value!r}"

    def test_cpu_count_is_positive_integer_string(self) -> None:
        info = get_system_info()
        assert int(info["cpu_count"]) > 0

    def test_timestamp_non_empty(self) -> None:
        info = get_system_info()
        assert info["timestamp"]


class TestGetImageLoader:
    @pytest.mark.parametrize(
        "library",
        ["albumentationsx", "ultralytics", "imgaug", "torchvision", "kornia", "augly"],
    )
    def test_known_library_returns_callable(self, library: str) -> None:
        loader = get_image_loader(library)
        assert callable(loader)

    def test_unknown_library_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported library"):
            get_image_loader("nonexistent_lib")

    def test_same_library_returns_same_callable(self) -> None:
        a = get_image_loader("albumentationsx")
        b = get_image_loader("albumentationsx")
        assert a is b  # cached


class TestGetVideoLoader:
    @pytest.mark.parametrize(
        "library",
        ["albumentationsx", "torchvision", "kornia"],
    )
    def test_known_library_returns_callable(self, library: str) -> None:
        loader = get_video_loader(library)
        assert callable(loader)

    def test_unknown_library_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported library"):
            get_video_loader("augly")

    def test_imgaug_not_supported_for_video(self) -> None:
        with pytest.raises(ValueError, match="Unsupported library"):
            get_video_loader("imgaug")


class TestIsVarianceStable:
    def test_insufficient_data_returns_false(self) -> None:
        # Need window * min_windows = 5 * 3 = 15 samples minimum
        assert not is_variance_stable([1.0] * 14, window=5, threshold=0.05, min_windows=3)

    def test_constant_series_returns_true(self) -> None:
        data = [1000.0] * 20
        assert is_variance_stable(data, window=5, threshold=0.05, min_windows=3)

    def test_changing_variance_between_windows_returns_false(self) -> None:
        # Last window: constant (variance ≈ 0), second-to-last: all over the place (high variance)
        # → variance ratio between trailing windows is large → not stable
        last_window = [500.0] * 5  # variance = 0
        second_window = [1.0, 1000.0, 2.0, 999.0, 500.0]  # high variance
        third_window = [250.0, 250.0, 250.0, 250.0, 250.0]  # variance = 0

        # Build so trailing windows (accessed as negative slices) are as above
        data = third_window + second_window + last_window
        result = is_variance_stable(data, window=5, threshold=0.05, min_windows=3)
        # If any variance ratio > threshold, returns False
        # var(last_window)=0, var(second_window)=large → ratio will be 1.0 → False
        assert not result

    def test_exact_minimum_length(self) -> None:
        # Exactly window * min_windows → should evaluate, not short-circuit
        data = [500.0] * 15
        result = is_variance_stable(data, window=5, threshold=0.05, min_windows=3)
        assert result is True

    @pytest.mark.parametrize("n", [0, 1, 5, 14])
    def test_short_inputs_always_false(self, n: int) -> None:
        data = [1.0] * n
        assert not is_variance_stable(data, window=5, threshold=0.05, min_windows=3)


class TestTimeTransform:
    def test_returns_positive_float(self) -> None:
        images = [object() for _ in range(5)]
        elapsed = time_transform(lambda x: x, images)
        assert isinstance(elapsed, float)
        assert elapsed > 0

    def test_empty_list_returns_float(self) -> None:
        elapsed = time_transform(lambda x: x, [])
        assert isinstance(elapsed, float)
        assert elapsed >= 0

    def test_transform_called_for_each_image(self) -> None:
        call_count = 0

        def counting_transform(x: object) -> object:
            nonlocal call_count
            call_count += 1
            return x

        images = [object() for _ in range(7)]
        time_transform(counting_transform, images)
        assert call_count == 7
