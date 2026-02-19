"""Tests for load_from_python_file() and BenchmarkRunner.filter_transforms()."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from benchmark.runner import BenchmarkRunner, load_from_python_file


class TestLoadFromPythonFile:
    def test_valid_spec_returns_tuple(self, tmp_spec_file: Path) -> None:
        library, call_fn, transforms = load_from_python_file(tmp_spec_file)
        assert isinstance(library, str)
        assert callable(call_fn)
        assert isinstance(transforms, list)

    def test_valid_spec_library_name(self, tmp_spec_file: Path) -> None:
        library, _, _ = load_from_python_file(tmp_spec_file)
        assert library == "testlib"

    def test_valid_spec_transforms_list(self, tmp_spec_file: Path) -> None:
        _, _, transforms = load_from_python_file(tmp_spec_file)
        assert len(transforms) == 2
        assert transforms[0]["name"] == "Noop"
        assert transforms[1]["name"] == "Identity"

    def test_valid_spec_call_fn_is_callable(self, tmp_spec_file: Path) -> None:
        _, call_fn, _ = load_from_python_file(tmp_spec_file)
        assert callable(call_fn)

    def test_missing_library_raises_value_error(self, make_spec_file) -> None:
        path = make_spec_file(library=None)
        with pytest.raises(ValueError, match="LIBRARY"):
            load_from_python_file(path)

    def test_missing_transforms_raises_value_error(self, make_spec_file) -> None:
        path = make_spec_file(transforms=None)
        with pytest.raises(ValueError, match="TRANSFORMS"):
            load_from_python_file(path)

    def test_missing_call_fn_raises_type_error(self, make_spec_file) -> None:
        path = make_spec_file(call_fn=None)
        with pytest.raises(TypeError, match="__call__"):
            load_from_python_file(path)

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.py"
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            load_from_python_file(missing)

    @pytest.mark.parametrize(
        "bad_transforms",
        [
            "TRANSFORMS = ['not_a_dict']",
            "TRANSFORMS = [42]",
            "TRANSFORMS = [None]",
        ],
    )
    def test_transforms_not_dict_raises_type_error(self, make_spec_file, bad_transforms: str) -> None:
        path = make_spec_file(transforms=bad_transforms)
        with pytest.raises(TypeError, match="dictionary"):
            load_from_python_file(path)

    @pytest.mark.parametrize(
        "bad_transforms",
        [
            'TRANSFORMS = [{"transform": lambda x: x}]',  # missing "name"
            'TRANSFORMS = [{"name": "Foo"}]',  # missing "transform"
        ],
    )
    def test_transforms_missing_required_keys_raises_value_error(
        self,
        make_spec_file,
        bad_transforms: str,
    ) -> None:
        path = make_spec_file(transforms=bad_transforms)
        with pytest.raises(ValueError, match="missing keys"):
            load_from_python_file(path)


class TestFilterTransforms:
    SAMPLE: ClassVar[list[dict]] = [
        {"name": "Foo", "transform": object()},
        {"name": "Bar", "transform": object()},
        {"name": "Baz", "transform": object()},
    ]

    def test_none_returns_all(self) -> None:
        result = BenchmarkRunner.filter_transforms(self.SAMPLE, None)
        assert result == self.SAMPLE

    def test_empty_list_returns_empty(self) -> None:
        result = BenchmarkRunner.filter_transforms(self.SAMPLE, [])
        assert result == []

    def test_single_match(self) -> None:
        result = BenchmarkRunner.filter_transforms(self.SAMPLE, ["Foo"])
        assert len(result) == 1
        assert result[0]["name"] == "Foo"

    def test_multiple_matches(self) -> None:
        result = BenchmarkRunner.filter_transforms(self.SAMPLE, ["Foo", "Baz"])
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"Foo", "Baz"}

    def test_no_match_returns_empty(self) -> None:
        result = BenchmarkRunner.filter_transforms(self.SAMPLE, ["Missing"])
        assert result == []

    def test_case_sensitive_no_match(self) -> None:
        result = BenchmarkRunner.filter_transforms(self.SAMPLE, ["foo"])
        assert result == []

    def test_case_sensitive_match(self) -> None:
        result = BenchmarkRunner.filter_transforms(self.SAMPLE, ["Foo"])
        assert len(result) == 1

    def test_empty_input_list(self) -> None:
        result = BenchmarkRunner.filter_transforms([], ["Foo"])
        assert result == []

    def test_preserves_transform_objects(self) -> None:
        original_obj = self.SAMPLE[0]["transform"]
        result = BenchmarkRunner.filter_transforms(self.SAMPLE, ["Foo"])
        assert result[0]["transform"] is original_obj
