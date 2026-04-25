from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

BenchmarkMode = Literal["micro", "pipeline", "decode"]
ScenarioName = Literal["image-rgb", "image-9ch", "video-decode-16f", "video-16f"]


@dataclass(frozen=True)
class Scenario:
    name: ScenarioName
    media: Literal["image", "video"]
    default_mode: BenchmarkMode
    num_channels: int = 3
    clip_length: int | None = None
    micro_libraries: tuple[str, ...] = ()
    pipeline_libraries: tuple[str, ...] = ()
    decoders: tuple[str, ...] = ()

    def supports_mode(self, mode: BenchmarkMode) -> bool:
        if mode == "micro":
            return bool(self.micro_libraries)
        if mode == "pipeline":
            return bool(self.pipeline_libraries)
        return bool(self.decoders)

    def libraries_for_mode(self, mode: BenchmarkMode) -> tuple[str, ...]:
        if mode == "micro":
            return self.micro_libraries
        if mode == "pipeline":
            return self.pipeline_libraries
        return ()


SCENARIOS: dict[str, Scenario] = {
    "image-rgb": Scenario(
        name="image-rgb",
        media="image",
        default_mode="micro",
        num_channels=3,
        micro_libraries=("albumentationsx", "torchvision", "kornia", "pillow"),
        pipeline_libraries=("albumentationsx", "torchvision", "kornia", "pillow"),
    ),
    "image-9ch": Scenario(
        name="image-9ch",
        media="image",
        default_mode="micro",
        num_channels=9,
        micro_libraries=("albumentationsx", "torchvision", "kornia"),
        pipeline_libraries=("albumentationsx", "torchvision", "kornia"),
    ),
    "video-decode-16f": Scenario(
        name="video-decode-16f",
        media="video",
        default_mode="decode",
        clip_length=16,
        decoders=("opencv", "pyav", "decord", "torchvision", "torchcodec", "dali"),
    ),
    "video-16f": Scenario(
        name="video-16f",
        media="video",
        default_mode="micro",
        clip_length=16,
        micro_libraries=("albumentationsx", "torchvision", "kornia"),
        pipeline_libraries=("albumentationsx", "torchvision", "kornia", "dali"),
    ),
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as e:
        available = ", ".join(sorted(SCENARIOS))
        msg = f"Unknown scenario {name!r}. Available scenarios: {available}"
        raise ValueError(msg) from e


def resolve_mode(scenario: Scenario, mode: str | None) -> BenchmarkMode:
    selected = mode or scenario.default_mode
    if selected not in {"micro", "pipeline", "decode"}:
        msg = f"Unknown benchmark mode {selected!r}"
        raise ValueError(msg)
    typed_mode = cast("BenchmarkMode", selected)
    if not scenario.supports_mode(typed_mode):
        msg = f"Scenario {scenario.name!r} does not support mode {selected!r}"
        raise ValueError(msg)
    return typed_mode


def resolve_libraries(scenario: Scenario, mode: BenchmarkMode, requested: list[str] | None) -> list[str]:
    available = scenario.libraries_for_mode(mode)
    if not available:
        return []
    selected = requested or list(available)
    unknown = set(selected) - set(available)
    if unknown:
        msg = f"Unknown libraries for {scenario.name}/{mode}: {sorted(unknown)}. Available: {list(available)}"
        raise ValueError(msg)
    return selected


def resolve_decoders(scenario: Scenario, requested: list[str] | None) -> list[str]:
    available = scenario.decoders
    selected = requested or list(available)
    unknown = set(selected) - set(available)
    if unknown:
        msg = f"Unknown decoders for {scenario.name}: {sorted(unknown)}. Available: {list(available)}"
        raise ValueError(msg)
    return selected
