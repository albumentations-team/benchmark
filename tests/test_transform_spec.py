"""Tests for benchmark/transforms/specs.py."""

from __future__ import annotations

import pytest

from benchmark.transforms.specs import TRANSFORM_SPECS, TransformSpec


class TestTransformSpec:
    def test_stores_name_and_params(self) -> None:
        spec = TransformSpec(name="Foo", params={"a": 1, "b": "x"})
        assert spec.name == "Foo"
        assert spec.params == {"a": 1, "b": "x"}

    def test_empty_params(self) -> None:
        spec = TransformSpec(name="HorizontalFlip", params={})
        assert spec.params == {}

    def test_str_with_params(self) -> None:
        spec = TransformSpec(name="Resize", params={"target_size": 512, "interpolation": "bilinear"})
        result = str(spec)
        assert result.startswith("Resize(")
        assert "target_size=512" in result
        assert "interpolation=bilinear" in result

    def test_str_no_params(self) -> None:
        spec = TransformSpec(name="HorizontalFlip", params={})
        assert str(spec) == "HorizontalFlip()"

    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("Resize", {"target_size": 256}),
            ("GaussianBlur", {"sigma": 2.0, "kernel_size": (5, 5)}),
            ("Normalize", {"mean": (0.5,), "std": (0.5,)}),
        ],
    )
    def test_roundtrip_various_params(self, name: str, params: dict) -> None:
        spec = TransformSpec(name=name, params=params)
        assert spec.name == name
        assert spec.params == params


class TestTransformSpecsList:
    def test_non_empty(self) -> None:
        assert len(TRANSFORM_SPECS) > 0

    def test_all_are_transform_spec_instances(self) -> None:
        for spec in TRANSFORM_SPECS:
            assert isinstance(spec, TransformSpec), f"{spec!r} is not a TransformSpec"

    def test_all_have_non_empty_names(self) -> None:
        for spec in TRANSFORM_SPECS:
            assert spec.name, f"Empty name found in TRANSFORM_SPECS: {spec!r}"

    def test_no_duplicate_names(self) -> None:
        names = [spec.name for spec in TRANSFORM_SPECS]
        duplicates = {n for n in names if names.count(n) > 1}
        assert not duplicates, f"Duplicate transform names: {duplicates}"

    def test_all_params_are_dicts(self) -> None:
        for spec in TRANSFORM_SPECS:
            assert isinstance(spec.params, dict), f"{spec.name}.params is not a dict"

    def test_known_transforms_present(self) -> None:
        names = {spec.name for spec in TRANSFORM_SPECS}
        expected = {"HorizontalFlip", "VerticalFlip", "Resize", "GaussianBlur", "Normalize", "Rotate"}
        missing = expected - names
        assert not missing, f"Expected transforms not found: {missing}"
