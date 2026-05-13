from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark.config.models import BenchmarkRunConfig
from benchmark.matrix import paper_transform_set_file

if TYPE_CHECKING:
    from pathlib import Path


def read_markdown_text_block(path: Path) -> list[str]:
    lines: list[str] = []
    in_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "```text":
            in_block = True
            continue
        if in_block and stripped == "```":
            break
        if in_block and stripped:
            lines.append(stripped)
    if not lines:
        raise ValueError(f"No text transform block found in {path}")
    return lines


def paper_transform_names(repo_root: Path, scenario_name: str, mode: str) -> list[str]:
    transform_set_path = repo_root / paper_transform_set_file(scenario_name)
    names = read_markdown_text_block(transform_set_path)
    if mode != "pipeline":
        return names
    if scenario_name in {"image-rgb", "image-9ch"}:
        from benchmark.transforms.image_recipe_specs import recipe_name, spec_by_name

        return [recipe_name(spec_by_name(name)) for name in names if name != "Normalize"]
    if scenario_name == "video-16f":
        from benchmark.transforms.video_recipe_specs import recipe_name, spec_by_name

        return [recipe_name(spec_by_name(name)) for name in names if name != "Normalize"]
    return names


def resolve_config_transform_set(config: BenchmarkRunConfig, repo_root: Path) -> BenchmarkRunConfig:
    if config.selection.transform_set is None or config.selection.transforms:
        return config
    if config.selection.scenario is None:
        raise ValueError("selection.transform_set requires selection.scenario")
    if config.selection.transform_set != "paper":
        raise ValueError(f"Unknown transform set {config.selection.transform_set!r}")

    mode = config.resolved_mode()
    data = config.model_dump()
    data["selection"]["transforms"] = paper_transform_names(repo_root, config.selection.scenario, mode)
    return BenchmarkRunConfig.model_validate(data)
