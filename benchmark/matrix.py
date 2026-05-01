from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.scenarios import BenchmarkMode, ScenarioName

MediaName = Literal["image", "video"]
DeviceOption = Literal["none", "cuda", "mps", "auto"]
PipelineScope = Literal[
    "memory_dataloader_augment",
    "decode_dataloader_augment",
    "decode_dataloader_augment_batch_copy",
]


@dataclass(frozen=True)
class LibraryScenarioConfig:
    scenario: ScenarioName
    mode: BenchmarkMode
    library: str
    spec_path: str | None
    requirements_media: MediaName
    env_group: str
    devices: tuple[DeviceOption, ...] = ("none",)
    pipeline_scopes: tuple[PipelineScope, ...] = ()
    backend: Literal["pyperf", "pipeline", "decode", "dali_pipeline"] = "pyperf"


IMAGE_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_impl.py",
    "albumentations_mit": "benchmark/transforms/albumentations_mit_impl.py",
    "torchvision": "benchmark/transforms/torchvision_impl.py",
    "kornia": "benchmark/transforms/kornia_impl.py",
    "pillow": "benchmark/transforms/pillow_impl.py",
}

IMAGE_PIPELINE_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_pipeline_impl.py",
    "torchvision": "benchmark/transforms/torchvision_pipeline_impl.py",
    "kornia": "benchmark/transforms/kornia_pipeline_impl.py",
    "pillow": "benchmark/transforms/pillow_pipeline_impl.py",
}

MULTICHANNEL_IMAGE_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_multichannel_impl.py",
    "albumentations_mit": "benchmark/transforms/albumentations_mit_multichannel_impl.py",
    "torchvision": "benchmark/transforms/torchvision_multichannel_impl.py",
    "kornia": "benchmark/transforms/kornia_multichannel_impl.py",
}

MULTICHANNEL_IMAGE_PIPELINE_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_multichannel_pipeline_impl.py",
    "torchvision": "benchmark/transforms/torchvision_multichannel_pipeline_impl.py",
    "kornia": "benchmark/transforms/kornia_multichannel_pipeline_impl.py",
}

VIDEO_SPECS: dict[str, str] = {
    "albumentationsx": "benchmark/transforms/albumentationsx_video_impl.py",
    "albumentations_mit": "benchmark/transforms/albumentations_mit_video_impl.py",
    "torchvision": "benchmark/transforms/torchvision_video_impl.py",
    "kornia": "benchmark/transforms/kornia_video_impl.py",
}

IMAGE_REQUIREMENTS: dict[str, str] = {
    "albumentationsx": "requirements/albumentationsx.txt",
    "albumentations_mit": "requirements/albumentations_mit.txt",
    "torchvision": "requirements/torchvision.txt",
    "kornia": "requirements/kornia.txt",
    "pillow": "requirements/pillow.txt",
}

VIDEO_REQUIREMENTS: dict[str, str] = {
    "albumentationsx": "requirements/albumentationsx.txt",
    "albumentations_mit": "requirements/albumentations_mit.txt",
    "torchvision": "requirements/torchvision-video.txt",
    "kornia": "requirements/kornia-video.txt",
    "dali": "requirements/dali-video.txt",
}

ENV_GROUPS: dict[MediaName, dict[str, tuple[str, ...]]] = {
    "image": {
        "albumentationsx": ("albumentationsx",),
        "albumentations_mit": ("albumentations_mit",),
        "torch_stack": ("torchvision", "kornia", "pillow"),
    },
    "video": {
        "albumentationsx_video": ("albumentationsx",),
        "albumentations_mit_video": ("albumentations_mit",),
        "torch_video": ("torchvision", "kornia"),
        "dali_video": ("dali",),
    },
}

PAPER_TRANSFORM_SET_FILES: dict[ScenarioName, str] = {
    "image-rgb": "docs/paper_transform_sets/rgb.md",
    "image-9ch": "docs/paper_transform_sets/9ch.md",
    "video-16f": "docs/paper_transform_sets/video.md",
}

PIPELINE_SCOPES: tuple[PipelineScope, ...] = (
    "memory_dataloader_augment",
    "decode_dataloader_augment",
    "decode_dataloader_augment_batch_copy",
)


def _scenario_spec_maps() -> dict[tuple[ScenarioName, BenchmarkMode], dict[str, str]]:
    return {
        ("image-rgb", "micro"): IMAGE_SPECS,
        ("image-rgb", "pipeline"): IMAGE_PIPELINE_SPECS,
        ("image-9ch", "micro"): MULTICHANNEL_IMAGE_SPECS,
        ("image-9ch", "pipeline"): MULTICHANNEL_IMAGE_PIPELINE_SPECS,
        ("video-16f", "micro"): VIDEO_SPECS,
        ("video-16f", "pipeline"): VIDEO_SPECS,
    }


def spec_map_for_scenario(scenario_name: str, mode: str) -> dict[str, str]:
    try:
        return _scenario_spec_maps()[(scenario_name, mode)]  # type: ignore[index]
    except KeyError as e:
        msg = f"No transform spec map for {scenario_name!r}/{mode!r}"
        raise ValueError(msg) from e


def paper_transform_set_file(scenario_name: str) -> str:
    try:
        return PAPER_TRANSFORM_SET_FILES[scenario_name]  # type: ignore[index]
    except KeyError as e:
        msg = f"--transform-set paper is not defined for scenario {scenario_name!r}"
        raise ValueError(msg) from e


def library_env_group(library: str, media: str) -> str:
    media_key = cast("MediaName", media)
    for group, libraries in ENV_GROUPS.get(media_key, {}).items():
        if library in libraries:
            return group
    return library


def requirements_for_env_group(env_group: str, media: str, repo_root: Path) -> list[Path]:
    media_key = cast("MediaName", media)
    req_map = VIDEO_REQUIREMENTS if media == "video" else IMAGE_REQUIREMENTS
    group_libraries = ENV_GROUPS.get(media_key, {}).get(env_group, (env_group.removesuffix("_video"),))
    requirements_paths = [repo_root / "requirements" / "requirements.txt"]
    seen = {requirements_paths[0]}
    for group_library in group_libraries:
        req_rel = req_map.get(group_library)
        if req_rel is None:
            continue
        req_path = repo_root / req_rel
        if req_path not in seen:
            requirements_paths.append(req_path)
            seen.add(req_path)
    return requirements_paths


def _devices_for(scenario_name: ScenarioName, mode: BenchmarkMode, library: str) -> tuple[DeviceOption, ...]:
    if mode != "pipeline":
        return ("none",)
    if scenario_name == "video-16f" or library in {"torchvision", "kornia"}:
        return ("none", "cuda", "mps", "auto")
    return ("none",)


def benchmark_matrix() -> tuple[LibraryScenarioConfig, ...]:
    entries: list[LibraryScenarioConfig] = []
    for (scenario_name, mode), spec_map in _scenario_spec_maps().items():
        media: MediaName = "video" if scenario_name.startswith("video") else "image"
        backend: Literal["pyperf", "pipeline", "decode", "dali_pipeline"] = (
            "pipeline" if mode == "pipeline" else "pyperf"
        )
        for library, spec_path in spec_map.items():
            entries.append(
                LibraryScenarioConfig(
                    scenario=scenario_name,
                    mode=mode,
                    library=library,
                    spec_path=spec_path,
                    requirements_media=media,
                    env_group=library_env_group(library, media),
                    devices=_devices_for(scenario_name, mode, library),
                    pipeline_scopes=PIPELINE_SCOPES if mode == "pipeline" else (),
                    backend=backend,
                ),
            )
    entries.append(
        LibraryScenarioConfig(
            scenario="video-16f",
            mode="pipeline",
            library="dali",
            spec_path=None,
            requirements_media="video",
            env_group=library_env_group("dali", "video"),
            devices=("cuda", "auto"),
            pipeline_scopes=PIPELINE_SCOPES,
            backend="dali_pipeline",
        ),
    )
    return tuple(entries)
