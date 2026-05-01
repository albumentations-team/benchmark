from __future__ import annotations

from pathlib import Path

from benchmark.matrix import (
    benchmark_matrix,
    paper_transform_set_file,
    requirements_for_env_group,
    spec_map_for_scenario,
)


def test_matrix_entries_have_existing_specs_and_requirements() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for entry in benchmark_matrix():
        if entry.spec_path is not None:
            assert (repo_root / entry.spec_path).exists(), entry
        for req in requirements_for_env_group(entry.env_group, entry.requirements_media, repo_root):
            assert req.exists(), (entry, req)


def test_matrix_covers_rgb_9ch_video_and_dali_pipeline() -> None:
    entries = benchmark_matrix()
    keys = {(entry.scenario, entry.mode, entry.library, entry.backend) for entry in entries}

    assert ("image-rgb", "micro", "albumentationsx", "pyperf") in keys
    assert ("image-9ch", "pipeline", "kornia", "pipeline") in keys
    assert ("video-16f", "micro", "torchvision", "pyperf") in keys
    assert ("video-16f", "pipeline", "dali", "dali_pipeline") in keys


def test_spec_maps_are_scenario_and_mode_specific() -> None:
    assert spec_map_for_scenario("image-rgb", "micro")["kornia"].endswith("kornia_impl.py")
    assert spec_map_for_scenario("image-9ch", "micro")["kornia"].endswith("kornia_multichannel_impl.py")
    assert spec_map_for_scenario("image-rgb", "pipeline")["kornia"].endswith("kornia_pipeline_impl.py")
    assert spec_map_for_scenario("video-16f", "pipeline")["kornia"].endswith("kornia_video_impl.py")


def test_paper_transform_sets_are_declared_in_matrix() -> None:
    assert paper_transform_set_file("image-rgb") == "docs/paper_transform_sets/rgb.md"
    assert paper_transform_set_file("image-9ch") == "docs/paper_transform_sets/9ch.md"
    assert paper_transform_set_file("video-16f") == "docs/paper_transform_sets/video.md"


def test_pipeline_device_policy_includes_gpu_and_mps_where_applicable() -> None:
    entries = {(entry.scenario, entry.mode, entry.library): entry for entry in benchmark_matrix()}

    assert entries[("image-rgb", "pipeline", "kornia")].devices == ("none", "cuda", "mps", "auto")
    assert entries[("image-rgb", "pipeline", "pillow")].devices == ("none",)
    assert entries[("video-16f", "pipeline", "dali")].devices == ("cuda", "auto")
