"""Shared fixtures for the benchmark test suite."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest

import benchmark.transforms.registry as registry_module

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_spec_file(tmp_path: Path) -> Path:
    """Minimal valid spec file — no real augmentation library imports."""
    spec = tmp_path / "valid_spec.py"
    spec.write_text(
        """\
LIBRARY = "testlib"

def __call__(transform, image):
    return transform(image)

TRANSFORMS = [
    {"name": "Noop", "transform": lambda x: x},
    {"name": "Identity", "transform": lambda x: x},
]
""",
    )
    return spec


@pytest.fixture
def make_spec_file(tmp_path: Path):
    """Factory fixture that writes a spec file with optional overrides.

    Pass keyword arguments to override or omit top-level definitions:
      - library: str | None  (None → omit LIBRARY)
      - call_fn: str | None  (None → omit __call__)
      - transforms: str | None  (None → omit TRANSFORMS; pass raw Python source string)
    """

    def _make(
        library: str | None = "testlib",
        call_fn: str | None = "def __call__(transform, image):\n    return transform(image)\n",
        transforms: str | None = 'TRANSFORMS = [{"name": "Noop", "transform": lambda x: x}]',
    ) -> Path:
        lines: list[str] = []
        if library is not None:
            lines.append(f'LIBRARY = "{library}"')
        if call_fn is not None:
            lines.append(call_fn)
        if transforms is not None:
            lines.append(transforms)
        path = tmp_path / "spec.py"
        path.write_text("\n".join(lines) + "\n")
        return path

    return _make


def _make_result_entry(median_throughput: float = 1000.0, *, early_stopped: bool = False) -> dict[str, Any]:
    return {
        "supported": True,
        "warmup_iterations": 10,
        "throughputs": [median_throughput] * 5,
        "median_throughput": median_throughput,
        "std_throughput": 10.0,
        "times": [0.001] * 5,
        "mean_time": 0.001,
        "std_time": 0.0001,
        "variance_stable": not early_stopped,
        "early_stopped": early_stopped,
        "early_stop_reason": "too slow" if early_stopped else None,
    }


def _make_result_json(
    library: str = "albumentationsx",
    media: str = "image",
    transforms: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if transforms is None:
        transforms = {
            "HorizontalFlip": _make_result_entry(1000.0),
            "GaussianBlur": _make_result_entry(500.0),
        }
    num_key = "num_videos" if media == "video" else "num_images"
    return {
        "metadata": {
            "system_info": {"python_version": "3.13", "platform": "test", "processor": "x86", "cpu_count": "8"},
            "library_versions": {library: "1.0.0", "numpy": "2.0.0"},
            "thread_settings": {},
            "benchmark_params": {num_key: 100, "num_runs": 5},
        },
        "results": transforms,
    }


@pytest.fixture
def minimal_result_json(tmp_path: Path) -> Path:
    """Single image-mode result JSON file for albumentationsx."""
    path = tmp_path / "albumentationsx_results.json"
    path.write_text(json.dumps(_make_result_json()))
    return path


@pytest.fixture
def result_dir_fixture(tmp_path: Path) -> Path:
    """Directory with two result JSON files (image + video)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    (results_dir / "albumentationsx_results.json").write_text(
        json.dumps(_make_result_json("albumentationsx", "image")),
    )
    (results_dir / "kornia_video_results.json").write_text(
        json.dumps(_make_result_json("kornia", "video")),
    )
    return results_dir


@pytest.fixture
def baseline_and_current_dirs(tmp_path: Path):
    """Two result directories for regression testing.

    Returns (baseline_dir, current_dir). baseline has throughput 1000,
    current has throughput 500 for HorizontalFlip (a clear regression).
    """
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()

    baseline_transforms = {
        "HorizontalFlip": _make_result_entry(1000.0),
        "GaussianBlur": _make_result_entry(500.0),
    }
    current_transforms = {
        "HorizontalFlip": _make_result_entry(500.0),  # regression: -50%
        "GaussianBlur": _make_result_entry(600.0),  # improvement: +20%
    }

    (baseline_dir / "albumentationsx_results.json").write_text(
        json.dumps(_make_result_json("albumentationsx", "image", baseline_transforms)),
    )
    (current_dir / "albumentationsx_results.json").write_text(
        json.dumps(_make_result_json("albumentationsx", "image", current_transforms)),
    )
    return baseline_dir, current_dir


@pytest.fixture(autouse=True)
def registry_snapshot():
    """Restore global registry state after each test that mutates it.

    autouse=True so it runs for every test — cheap (just a deepcopy) and
    protects against register_library() calls leaking between tests.

    Both TRANSFORMS and _BY_NAME are rebuilt from a single deepcopy so
    they reference the same TransformDef objects (register_library mutates
    objects found via _BY_NAME, and tests assert via TRANSFORMS).
    """
    original_transforms = copy.deepcopy(registry_module.TRANSFORMS)
    # Rebuild _BY_NAME from the same copied objects so they share identity
    original_by_name = {td.name: td for td in original_transforms}
    yield
    registry_module.TRANSFORMS.clear()
    registry_module.TRANSFORMS.extend(original_transforms)
    registry_module._BY_NAME.clear()
    registry_module._BY_NAME.update(original_by_name)
