"""Tests for benchmark/transforms/registry.py."""

from __future__ import annotations

from benchmark.transforms.registry import _BY_NAME, TRANSFORMS, build_transforms, register_library
from benchmark.transforms.specs import TRANSFORM_SPECS


class TestRegistryInitialState:
    def test_transforms_length_matches_specs(self) -> None:
        assert len(TRANSFORMS) == len(TRANSFORM_SPECS)

    def test_transforms_names_match_specs(self) -> None:
        spec_names = [s.name for s in TRANSFORM_SPECS]
        reg_names = [t.name for t in TRANSFORMS]
        assert reg_names == spec_names

    def test_by_name_covers_all_transforms(self) -> None:
        for td in TRANSFORMS:
            assert td.name in _BY_NAME
            assert _BY_NAME[td.name].name == td.name

    def test_by_name_length_matches_transforms(self) -> None:
        assert len(_BY_NAME) == len(TRANSFORMS)

    def test_initial_image_impls_empty(self) -> None:
        for td in TRANSFORMS:
            assert td.image_impls == {}

    def test_initial_video_impls_empty(self) -> None:
        for td in TRANSFORMS:
            assert td.video_impls == {}


class TestBuildTransforms:
    def test_returns_empty_for_unknown_library(self) -> None:
        result = build_transforms("nonexistent_lib", media="image")
        assert result == []

    def test_returns_empty_for_unknown_library_video(self) -> None:
        result = build_transforms("nonexistent_lib", media="video")
        assert result == []

    def test_result_dicts_have_required_keys(self) -> None:
        # Register a minimal image factory first
        register_library("_test_lib", create_image_fn=lambda _spec: object())
        result = build_transforms("_test_lib", media="image")
        for entry in result:
            assert "name" in entry
            assert "transform" in entry

    def test_excludes_none_transforms(self) -> None:
        # Factory returns None for every spec
        register_library("_null_lib", create_image_fn=lambda _spec: None)
        result = build_transforms("_null_lib", media="image")
        assert result == []

    def test_includes_only_non_none(self) -> None:
        first_name = TRANSFORM_SPECS[0].name

        def selective_factory(spec) -> object | None:
            return object() if spec.name == first_name else None

        register_library("_selective_lib", create_image_fn=selective_factory)
        result = build_transforms("_selective_lib", media="image")
        assert len(result) == 1
        assert result[0]["name"] == first_name

    def test_video_uses_video_impls(self) -> None:
        first_name = TRANSFORM_SPECS[0].name

        def video_factory(spec) -> object | None:
            return object() if spec.name == first_name else None

        register_library("_video_lib", create_video_fn=video_factory)
        image_result = build_transforms("_video_lib", media="image")
        video_result = build_transforms("_video_lib", media="video")
        assert image_result == []
        assert len(video_result) == 1
        assert video_result[0]["name"] == first_name


class TestRegisterLibrary:
    def test_registers_image_impls(self) -> None:
        sentinel = object()
        register_library("_reg_test", create_image_fn=lambda _spec: sentinel)
        for td in TRANSFORMS:
            assert "_reg_test" in td.image_impls
            assert td.image_impls["_reg_test"] is sentinel

    def test_registers_video_impls(self) -> None:
        sentinel = object()
        register_library("_reg_video", create_video_fn=lambda _spec: sentinel)
        for td in TRANSFORMS:
            assert "_reg_video" in td.video_impls

    def test_graceful_on_factory_exception(self) -> None:
        def bad_factory(_spec) -> None:
            raise RuntimeError("intentional failure")

        # Should not raise — exceptions are caught and logged
        register_library("_bad_lib", create_image_fn=bad_factory)
        for td in TRANSFORMS:
            assert td.image_impls.get("_bad_lib") is None

    def test_none_factory_skips_image(self) -> None:
        register_library("_skip_lib", create_image_fn=None, create_video_fn=None)
        for td in TRANSFORMS:
            assert "_skip_lib" not in td.image_impls
            assert "_skip_lib" not in td.video_impls

    def test_build_after_register_returns_transforms(self) -> None:
        transform_obj = object()
        register_library("_build_test", create_image_fn=lambda _spec: transform_obj)
        result = build_transforms("_build_test", media="image")
        assert len(result) == len(TRANSFORM_SPECS)
        assert all(entry["transform"] is transform_obj for entry in result)
