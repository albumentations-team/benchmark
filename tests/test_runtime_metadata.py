import importlib.metadata
from types import SimpleNamespace

import pytest

from augbench.rgb_executor import _runtime_metadata


@pytest.mark.parametrize(
    ("implementation_id", "distribution"),
    [
        ("albumentationsx_cpu", "albumentationsx"),
        ("dali_gpu", "nvidia-dali-cuda120"),
        ("kornia_cpu", "kornia"),
        ("pillow_cpu", "pillow"),
        ("torchvision_gpu", "torchvision"),
    ],
)
def test_runtime_metadata_uses_the_locked_distribution_name(
    monkeypatch: pytest.MonkeyPatch,
    implementation_id: str,
    distribution: str,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: f"version:{name}")

    runtime = _runtime_metadata(SimpleNamespace(implementation_id=implementation_id))

    assert runtime["library_version"] == f"version:{distribution}"
