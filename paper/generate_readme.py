# ruff: noqa: INP001
"""Update README results and figures from one explicitly selected, complete RGB run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import fmean, median
from typing import TYPE_CHECKING

from PIL import Image

from augbench.recipes.load import load_recipe_catalog
from augbench.run_config import FamilyRunConfig
from augbench.run_records import RunRecord
from paper.generate_results import (
    AX,
    IMPLEMENTATIONS,
    PILLOW_ALLSIMD,
    RUN_ID,
    _common_memories,
    _mean_ratios,
    _prepare,
    _shared_recipes,
    _single,
    _write_heatmap,
    _write_mean_bars,
    _write_memory_bars,
    load_run_results,
)

if TYPE_CHECKING:
    from paper.generate_results import _PaperData

_BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
_END = "<!-- END GENERATED RESULTS -->"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="immutable run.json for the selected run")
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument(
        "--recipes", type=Path, help="selected run's recipe catalog; defaults to the repository catalog"
    )
    args = parser.parse_args()
    run = RunRecord.model_validate_json(args.run.read_bytes())
    config = FamilyRunConfig.model_validate(run.family_config)
    recipes_path = args.recipes or root / config.recipes
    if hashlib.sha256(recipes_path.read_bytes()).hexdigest() != run.inputs.recipe_catalog_sha256:
        raise ValueError("recipe catalog checksum differs from the selected run; supply its catalog with --recipes")
    base = {implementation for implementation, _ in IMPLEMENTATIONS}
    if set(config.implementations) not in (base, base | {PILLOW_ALLSIMD[0]}):
        raise ValueError("README figures require the RGB implementations in the selected run")
    recipes = load_recipe_catalog(recipes_path)
    data = _prepare(load_run_results(args.cells, run, recipes), recipes.recipes)
    if not data.common:
        raise ValueError("the selected run has no recipes shared by all paths")
    readme = root / "README.md"
    updated = replace_results(readme.read_text(), readme_results(data, run))
    with tempfile.TemporaryDirectory(prefix="augbench-readme-") as directory:
        staging = Path(directory)
        render_figures(staging, root / "paper/figures.tex", data, config)
        destination = root / "docs/generated"
        destination.mkdir(parents=True, exist_ok=True)
        for figure in staging.glob("*.png"):
            shutil.copyfile(figure, destination / figure.name)
        (destination / "run.json").write_text(run.model_dump_json(indent=2) + "\n")
        (destination / "results.json").write_text(
            json.dumps({**export_results(data, run), "figures": export_figures(staging)}, indent=2, allow_nan=False)
            + "\n"
        )
        readme.write_text(updated)


def export_figures(directory: Path) -> dict[str, dict[str, str | int]]:
    figures = {}
    for path in sorted(directory.glob("*.png")):
        with Image.open(path) as image:
            width, height = image.size
        figures[path.stem] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": width,
            "height": height,
        }
    return figures


def export_results(data: _PaperData, run: RunRecord) -> dict[str, object]:
    """Export the selected run using the README's aggregation and supported recipe sets."""
    libraries = [
        {
            "id": implementation,
            "label": "AlbumentationsX CPU" if implementation == AX else label,
            "version": _single(
                {
                    record.runtime["library_version"]
                    for (impl, _), records in data.groups.items()
                    if impl == implementation
                    for record in records
                }
            ),
        }
        for implementation, label in data.implementations
    ]
    recipes = []
    for index, recipe in enumerate(data.recipes, 1):
        results = {}
        for implementation, _ in data.implementations:
            key = (implementation, recipe.recipe_id)
            if key not in data.groups:
                continue
            records = data.groups[key]
            speeds = [record.throughput.value for record in records]
            memories = [record.gpu_memory.peak_mib for record in records]
            results[implementation] = {
                "throughput": {"median": data.speed[key], "min": min(speeds), "max": max(speeds)},
                "gpu_memory": {"median": data.memory[key], "min": min(memories), "max": max(memories)},
                "observations": [
                    {
                        "seed": record.cell.seed,
                        "cell_id": record.cell_id,
                        "throughput_images_s": record.throughput.value,
                        "gpu_memory_peak_mib": record.gpu_memory.peak_mib,
                    }
                    for record in records
                ],
            }
        recipes.append({"key": f"R{index:02d}", "id": recipe.recipe_id, "results": results})
    groups = [("all", tuple(implementation for implementation, _ in data.implementations))]
    groups.extend((family.lower(), (AX, *paths)) for family, paths in data.comparisons)
    summaries = []
    for name, implementations in groups:
        shared = _shared_recipes(data, implementations)
        means = _mean_ratios(data, implementations)
        summaries.append(
            {
                "id": name,
                "recipe_ids": shared,
                "paths": [
                    {
                        "id": implementation,
                        "mean_relative_throughput": means[implementation],
                        "median_gpu_memory_mib": median(data.memory[(implementation, recipe)] for recipe in shared),
                    }
                    for implementation in implementations
                ],
            }
        )
    pairwise_comparisons = []
    measurement_notes = []
    if any(implementation == PILLOW_ALLSIMD[0] for implementation, _ in data.implementations):
        pillow_shared, pillow_ratios = _pillow_allsimd_ratios(data)
        pairwise_comparisons.append(
            {
                "id": "pillow_allsimd_vs_pillow",
                "recipe_ids": pillow_shared,
                "mean_throughput_ratio": fmean(pillow_ratios),
                "median_throughput_ratio": median(pillow_ratios),
            }
        )
        measurement_notes.append(
            "Pillow-AllSIMD ran in its own worker process. Other paths share the base worker, so their "
            "NVML peaks include a different process lifetime."
        )
    return {
        "schema_version": 1,
        "run": run.model_dump(mode="json"),
        "is_published_run": run.run_id == RUN_ID,
        "measurement_count": sum(map(len, data.groups.values())),
        "gpu_memory_poll_ms": _single(
            {record.gpu_memory.poll_interval_ms for records in data.groups.values() for record in records}
        ),
        "libraries": libraries,
        "recipes": recipes,
        "summaries": summaries,
        "pairwise_comparisons": pairwise_comparisons,
        "measurement_notes": measurement_notes,
    }


def replace_results(readme: str, generated: str) -> str:
    if readme.count(_BEGIN) != 1 or readme.count(_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    before, remaining = readme.split(_BEGIN)
    _, after = remaining.split(_END)
    return before + _BEGIN + "\n\n" + generated.rstrip() + "\n\n" + _END + after


def _pillow_allsimd_ratios(data: _PaperData) -> tuple[tuple[str, ...], list[float]]:
    recipes = _shared_recipes(data, ["pillow_cpu", PILLOW_ALLSIMD[0]])
    ratios = [data.speed[(PILLOW_ALLSIMD[0], recipe)] / data.speed[("pillow_cpu", recipe)] for recipe in recipes]
    return recipes, ratios


def readme_results(data: _PaperData, run: RunRecord) -> str:
    config = FamilyRunConfig.model_validate(run.family_config)
    execution = config.execution
    means = _mean_ratios(data, [implementation for implementation, _ in data.implementations])
    memories = _common_memories(data)
    labels = dict(data.implementations)
    labels[AX] = "AlbumentationsX CPU"
    count = len(data.common)
    pillow_variant_note = None
    if any(implementation == PILLOW_ALLSIMD[0] for implementation, _ in data.implementations):
        shared, speedups = _pillow_allsimd_ratios(data)
        pillow_variant_note = (
            f"Across {len(shared)} shared recipes, Pillow-AllSIMD was {fmean(speedups):.2f}\N{MULTIPLICATION SIGN} "
            f"as fast as Pillow on average, with a {median(speedups):.2f}\N{MULTIPLICATION SIGN} median ratio."
        )
        median_blur = next(
            (
                (index, recipe)
                for index, recipe in enumerate(data.recipes, 1)
                if recipe.augmentation_id == "MedianBlur" and recipe.recipe_id in shared
            ),
            None,
        )
        if median_blur is not None:
            index, recipe = median_blur
            ratio = data.speed[(PILLOW_ALLSIMD[0], recipe.recipe_id)] / data.speed[("pillow_cpu", recipe.recipe_id)]
            pillow_variant_note += f" Its R{index:02} MedianBlur ratio was {ratio:.2f}\N{MULTIPLICATION SIGN}."
    lines = [
        "## Results for the selected run",
        "",
        (
            f"Run [`{run.run_id[:12]}`](docs/generated/run.json); "
            f"[measured source `{run.inputs.git_commit[:7]}`]"
            f"(https://github.com/albumentations-team/benchmark/tree/{run.inputs.git_commit}). "
            f"{sum(map(len, data.groups.values()))} measurements, {len(data.groups)} implementation/recipe pairs, "
            f"{len({recipe for _, recipe in data.groups})} recipes, {len(execution.seeds)} seeds per pair."
        ),
        "",
        (
            "This is the run reported in the [published preprint](https://arxiv.org/abs/2609.06635)."
            if run.run_id == RUN_ID
            else "This selection differs from the published run. The article's results remain in "
            "[paper/generated](paper/generated)."
        ),
        "",
        (
            f"All rows below use the same {count} recipes. Throughput is the arithmetic mean of per-recipe ratios: "
            f"each path's median over {len(execution.seeds)} seeds divided by AX's median for that recipe. "
            "GPU memory is the median of those recipes' peak-memory medians."
        ),
        "",
        (
            "| Measured path | Version | Mean throughput / AX (higher is faster) | "
            "Peak process GPU memory (MiB, lower is better) |"
        ),
        "| --- | --- | ---: | ---: |",
    ]
    for implementation in sorted(means, key=means.__getitem__, reverse=True):
        version = _single(
            {
                record.runtime["library_version"]
                for (impl, _), records in data.groups.items()
                if impl == implementation
                for record in records
            }
        )
        lines.append(
            f"| {labels[implementation]} | {version} | "
            f"{means[implementation]:.2f}\N{MULTIPLICATION SIGN} | {memories[implementation]:,.0f}"
            f"{'†' if implementation == PILLOW_ALLSIMD[0] else ''} |"
        )
    if pillow_variant_note is not None:
        lines.extend(
            [
                "",
                pillow_variant_note,
                "",
                (
                    "†Pillow-AllSIMD ran in its own worker process to load its PIL package. "
                    "Other paths share the base worker, so these NVML peaks reflect different process lifetimes."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "### Per-recipe throughput",
            "",
            (
                "Each cell compares the same recipe's seed medians. AX images/s gives the absolute scale. "
                "Color shows the ratio on a logarithmic scale, not statistical significance. "
                "The published DALI Crop and Affine mappings differ from other paths; see the interpretation below."
            ),
            "",
            _image("common-heatmap", "Per-recipe throughput ratios and absolute AX throughput"),
            "",
            "### Throughput and GPU memory on the common recipe set",
            "",
            _image("common-mean-bars", f"Mean relative throughput on {count} common recipes"),
            "",
            _image("common-memory-bars", f"Median peak process GPU memory on {count} common recipes"),
            "",
            "### Broader pairwise comparisons",
            "",
            (
                "Each chart uses the exact recipe intersection of the paths it shows. "
                "Recipe sets differ between charts, so their averages cannot rank all libraries together."
            ),
        ],
    )
    for family, paths in data.comparisons:
        shared = len(_shared_recipes(data, [AX, *paths]))
        lines.extend(["", _image(f"pairwise-{family.lower()}", f"AX vs {family} on {shared} shared recipes")])
    lines.extend(
        [
            "",
            (
                "The counts below include recipes supported by only one competing path. "
                "AX wins when it exceeds every available competing path. Ratios and memory differences "
                "use the fastest competitor per recipe; exact ties select the first path in the comparison."
            ),
            "",
            (
                "| Competitor | Shared recipes | AX faster | Median competitor / AX throughput | "
                "Median competitor - AX GPU memory (MiB) |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ],
    )
    for family, rows in data.pairwise.items():
        lines.append(
            f"| {family} | {len(rows)} | {sum(row.ratio < 1 for row in rows)} | "
            f"{median(row.ratio for row in rows):.2f}\N{MULTIPLICATION SIGN} | "
            f"{median(row.memory_delta for row in rows):+,.0f} |"
        )
    lines.extend(
        [
            "",
            *_workload_lines(data, run, config),
            "",
            _image("boundary", "Throughput and GPU-memory measurement windows"),
        ]
    )
    return "\n".join(lines)


def _workload_lines(data: _PaperData, run: RunRecord, config: FamilyRunConfig) -> list[str]:
    execution = config.execution
    poll_ms = _single({record.gpu_memory.poll_interval_ms for records in data.groups.values() for record in records})
    shape = "x".join(map(str, (execution.batch_size, *config.output.shape)))
    return [
        "### Measurement settings",
        "",
        (
            f"Machine: `{run.hardware['machine_type']}`, `{run.hardware['accelerator']}`. "
            f"Output: CUDA `float16` BCHW `{shape}`. Dataset: {config.dataset.item_count:,} selected files. "
            f"Seeds: {', '.join(map(str, execution.seeds))}. "
            f"Workers: {execution.num_workers}; prefetch setting: {execution.prefetch_factor}; "
            f"persistent workers: {str(execution.persistent_workers).lower()}; "
            f"dataset prewarm: {str(execution.prewarm_dataset).lower()}."
        ),
        "",
        (
            f"Warm-up batches: {execution.warmup_batches}. Throughput measures "
            f"{execution.measured_batches} batches ({execution.batch_size * execution.measured_batches:,} images), "
            "ending after CUDA synchronization. Construction and worker startup are excluded; prefetch effects remain."
        ),
        "",
        (
            f"GPU memory is sampled by NVML every {poll_ms} ms from before pipeline construction through "
            "final synchronization and cleanup in the same pass. The peak includes process and library allocations. "
            "Shorter peaks may be missed; CPU and model memory are not measured."
        ),
    ]


def _image(name: str, description: str) -> str:
    return f"![{description}](docs/generated/{name}.png)"


def render_figures(output: Path, style: Path, data: _PaperData, config: FamilyRunConfig) -> None:
    generated = output / "generated"
    generated.mkdir()
    _write_heatmap(generated / "common-heatmap.tex", data)
    _write_memory_bars(generated / "common-memory-bars.tex", data)
    memory_limit = math.ceil(max(_common_memories(data).values()) / 500) * 500
    commands = {
        "boundary": rf"\InputBoundary[{'prewarmed bytes' if config.execution.prewarm_dataset else 'compressed bytes'}]"
        rf"{{{config.execution.measured_batches}}}",
        "common-heatmap": rf"\CommonHeatmap[{int(PILLOW_ALLSIMD in data.implementations)}]{{{len(data.common)}}}",
        "common-memory-bars": rf"\MemoryPlot[{memory_limit}]{{{len(data.implementations)}}}",
    }
    comparisons = {"common-mean-bars": [implementation for implementation, _ in data.implementations]}
    comparisons.update({f"pairwise-{family.lower()}": [AX, *paths] for family, paths in data.comparisons})
    for name, paths in comparisons.items():
        means = _mean_ratios(data, paths)
        _write_mean_bars(generated / f"{name}.tex", means)
        limit = max(1.5, math.ceil(max(means.values()) * 4) / 4)
        commands[name] = rf"\MeanRatioPlot[{limit}]{{{name}.tex}}{{{len(paths)}}}"
    shutil.copyfile(style, output / "figures.tex")
    source = output / "readme-figures.tex"
    source.write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{tikz,hyperref}",
                r"\hypersetup{hidelinks}",
                r"\usetikzlibrary{arrows.meta,positioning}",
                r"\usepackage[active,tightpage]{preview}",
                r"\PreviewEnvironment{tikzpicture}",
                r"\setlength{\PreviewBorder}{8pt}",
                r"\input{figures.tex}",
                r"\begin{document}",
                *commands.values(),
                r"\end{document}",
            ]
        )
        + "\n"
    )
    _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name], output)
    for page, name in enumerate(commands, 1):
        _run(
            [
                "pdftocairo",
                "-png",
                "-r",
                "180",
                "-singlefile",
                "-f",
                str(page),
                "-l",
                str(page),
                "readme-figures.pdf",
                name,
            ],
            output,
        )


def _run(command: list[str], directory: Path) -> None:
    completed = subprocess.run(  # noqa: S603
        command, cwd=directory, check=False, capture_output=True, text=True, errors="replace"
    )
    if completed.returncode:
        raise RuntimeError(f"{command[0]} failed:\n{completed.stdout}\n{completed.stderr}")


if __name__ == "__main__":
    main()
