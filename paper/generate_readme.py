# ruff: noqa: INP001
"""Update README results and figures from validated production runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import fmean, median
from typing import TYPE_CHECKING

from PIL import Image

from augbench.run_config import FamilyRunConfig
from paper.generate_results import (
    _COMMON_RECIPE_RATIO_PATHS,
    AX,
    PILLOW_ALLSIMD,
    RUN_ID,
    _common_memories,
    _mean_ratios,
    _shared_recipes,
    _single,
    add_run_arguments,
    batch_rate_interval,
    load_report,
    pillow_allsimd_ratios,
    write_plot_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from augbench.run_records import ResultRecord, RunRecord
    from paper.generate_results import _PaperData

_BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
_END = "<!-- END GENERATED RESULTS -->"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    args = parser.parse_args()
    data, run, ax_run = load_report(args, root)
    captions = figure_captions(data)
    readme = root / "README.md"
    updated = replace_results(readme.read_text(), readme_results(data, run))
    with tempfile.TemporaryDirectory(prefix="augbench-readme-") as directory:
        staging = Path(directory)
        render_figures(staging, root / "paper", data)
        destination = root / "docs/generated"
        destination.mkdir(parents=True, exist_ok=True)
        for figure in staging.glob("*.png"):
            shutil.copyfile(figure, destination / figure.name)
        (destination / "run.json").write_text(run.model_dump_json(indent=2) + "\n")
        (destination / "results.json").write_text(
            json.dumps(
                {
                    **export_results(data, run, ax_run, data.supplemental_runs),
                    "figures": export_figures(staging, captions),
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        readme.write_text(updated)


def export_figures(directory: Path, captions: Mapping[str, str]) -> dict[str, dict[str, str | int]]:
    figures: dict[str, dict[str, str | int]] = {}
    for name, caption in captions.items():
        path = directory / f"{name}.png"
        with Image.open(path) as image:
            width, height = image.size
        figures[name] = {
            "caption": caption,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": width,
            "height": height,
        }
    return figures


def _worker_note(data: _PaperData) -> str:
    scopes = {record.runtime.get("worker_scope") for records in data.groups.values() for record in records}
    if scopes == {"implementation_seed"}:
        return "Each implementation and seed ran in a fresh process. NVML sampled its baseline before pipeline setup."
    if scopes == {"implementation"}:
        return "Each implementation ran in its own process. NVML sampled the baseline before pipeline construction."
    return ""


def _observation(record: ResultRecord) -> dict[str, str | int | float]:
    return {
        "seed": record.cell.seed,
        "cell_id": record.cell_id,
        "run_id": record.run_id,
        "throughput_images_s": record.throughput.value,
        "gpu_memory_peak_mib": record.gpu_memory.peak_mib,
        **(
            {"gpu_memory_baseline_mib": record.gpu_memory.baseline_mib}
            if record.gpu_memory.baseline_mib is not None
            else {}
        ),
        **{field: record.runtime[field] for field in ("worker_hostname", "worker_scope") if field in record.runtime},
    }


def export_results(
    data: _PaperData,
    run: RunRecord,
    ax_run: RunRecord | None = None,
    supplemental_runs: tuple[RunRecord, ...] = (),
) -> dict[str, object]:
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
                "observations": [_observation(record) for record in records],
            }
        recipes.append({"key": f"R{index:02d}", "id": recipe.recipe_id, "results": results})
    groups = [("all", tuple(implementation for implementation, _ in data.implementations))]
    groups.extend((family.lower(), (AX, *paths)) for family, paths in data.comparisons)
    summaries = []
    batch_size = FamilyRunConfig.model_validate(run.family_config).execution.batch_size
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
                        "median_batches_per_second": (
                            median(data.speed[implementation, recipe] for recipe in shared) / batch_size
                        ),
                        "batch_rate_bootstrap_interval": batch_rate_interval(data, implementation, shared),
                        "median_gpu_memory_mib": median(data.memory[(implementation, recipe)] for recipe in shared),
                    }
                    for implementation in implementations
                ],
            }
        )
    pairwise_comparisons = []
    worker_note = _worker_note(data)
    measurement_notes = [worker_note] if worker_note else []
    if any(implementation == PILLOW_ALLSIMD[0] for implementation, _ in data.implementations):
        pillow_shared, pillow_ratios = pillow_allsimd_ratios(data)
        pairwise_comparisons.append(
            {
                "id": "pillow_allsimd_vs_pillow",
                "recipe_ids": pillow_shared,
                "mean_throughput_ratio": fmean(pillow_ratios),
                "median_throughput_ratio": median(pillow_ratios),
            }
        )
        if not worker_note:
            measurement_notes.append(
                "Pillow-AllSIMD ran in its own worker process. Other paths share the base worker, so their "
                "NVML peaks include a different process lifetime."
            )
    return {
        "schema_version": 2,
        "run": run.model_dump(mode="json"),
        **({"ax_run": ax_run.model_dump(mode="json")} if ax_run else {}),
        "supplemental_runs": [source.model_dump(mode="json") for source in supplemental_runs],
        "is_published_run": ax_run is None and not supplemental_runs and run.run_id == RUN_ID,
        "measurement_count": sum(map(len, data.groups.values())),
        "gpu_memory_poll_ms": _single(
            {record.gpu_memory.poll_interval_ms for records in data.groups.values() for record in records}
        ),
        "libraries": libraries,
        "recipes": recipes,
        "summaries": summaries,
        "pairwise_comparisons": pairwise_comparisons,
        "measurement_notes": measurement_notes,
        **(
            {"torchvision_seed_run": data.torchvision_seed_run.model_dump(mode="json")}
            if data.torchvision_seed_run
            else {}
        ),
    }


def replace_results(readme: str, generated: str) -> str:
    if readme.count(_BEGIN) != 1 or readme.count(_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    before, remaining = readme.split(_BEGIN)
    _, after = remaining.split(_END)
    return before + _BEGIN + "\n\n" + generated.rstrip() + "\n\n" + _END + after


def readme_results(data: _PaperData, run: RunRecord) -> str:
    config = FamilyRunConfig.model_validate(run.family_config)
    means = _mean_ratios(data, [implementation for implementation, _ in data.implementations])
    memories = _common_memories(data)
    labels = dict(data.implementations)
    labels[AX] = "AlbumentationsX CPU"
    count = len(data.common)
    captions = figure_captions(data)
    worker_note = _worker_note(data)
    pillow_variant_note = None
    if any(implementation == PILLOW_ALLSIMD[0] for implementation, _ in data.implementations):
        shared, speedups = pillow_allsimd_ratios(data)
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
        "## RGB results",
        "",
        (
            f"{sum(map(len, data.groups.values())):,} measurements across {len(data.groups)} "
            f"implementation\N{EN DASH}recipe pairs. "
            "See the [per-seed results and manifests](docs/generated/results.json)."
        ),
        "",
        _image("common-summary", captions["common-summary"]),
        "",
        "### Mean per-recipe throughput ratios",
        "",
        (
            f"All rows below use the same {count} recipes. Throughput is the arithmetic mean of per-recipe ratios: "
            "each path's median across its seeds divided by AX's median for that recipe. "
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
            f"{'†' if implementation == PILLOW_ALLSIMD[0] and not worker_note else ''} |"
        )
    if pillow_variant_note is not None:
        lines.extend(["", pillow_variant_note])
    if worker_note:
        lines.extend(["", worker_note])
    elif pillow_variant_note is not None:
        lines.extend(
            [
                "",
                (
                    "†Pillow-AllSIMD ran in its own worker process to load its PIL package. "
                    "Other paths share the base worker, so these NVML peaks reflect different process lifetimes."
                ),
            ]
        )
    ratio_paths = _COMMON_RECIPE_RATIO_PATHS
    lines.extend(
        [
            "",
            "### Throughput ratio on each shared recipe",
            "",
            (
                "Each value is the path's median throughput across its seeds divided by AX's median "
                "for that recipe. A value above 1.00 means that path delivered batches faster than AX."
            ),
            "",
            "| Shared recipe | " + " | ".join(label for _, label in ratio_paths) + " |",
            "| --- | " + " | ".join("---:" for _ in ratio_paths) + " |",
        ]
    )
    for index, recipe in enumerate(data.recipes, start=1):
        if recipe.recipe_id not in data.common:
            continue
        key = f"R{index:02d}"
        ratios = [
            data.speed[(implementation, recipe.recipe_id)] / data.speed[(AX, recipe.recipe_id)]
            for implementation, _ in ratio_paths
        ]
        lines.append(f"| {key} {recipe.augmentation_id} | " + " | ".join(f"{value:.2f}" for value in ratios) + " |")
    lines.extend(
        [
            "",
            "### AlbumentationsX against each library",
            "",
            "Each chart uses the recipes shared by every path it shows. Compare rows within each chart.",
        ],
    )
    for family, _ in data.comparisons:
        name = f"pairwise-{family.lower()}"
        lines.extend(["", _image(name, captions[name])])
    lines.extend(
        [
            "",
            *_workload_lines(data, run, config),
            "",
            _image("batch-wait", captions["batch-wait"]),
        ]
    )
    return "\n".join(lines)


def _workload_lines(data: _PaperData, run: RunRecord, config: FamilyRunConfig) -> list[str]:
    execution = config.execution
    poll_ms = _single({record.gpu_memory.poll_interval_ms for records in data.groups.values() for record in records})
    shape = "x".join(map(str, (execution.batch_size, *config.output.shape)))
    dataset = f"first {config.dataset.item_count:,} sorted ImageNet validation JPEGs"
    if config.dataset.selection == "first-sorted-non-cmyk":
        dataset = f"{config.dataset.item_count:,} non-CMYK ImageNet validation JPEGs with EXIF stripped before timing"
    seeds = f"Seeds {', '.join(map(str, execution.seeds))} for all paths."
    if data.torchvision_seed_run:
        extra = ", ".join(map(str, data.torchvision_seed_run.family_config["execution"]["seeds"]))
        seeds += f" TorchVision GPU also includes seeds {extra}."
    return [
        "### Measurement settings",
        "",
        (
            f"Machine: `{run.hardware['machine_type']}`, `{run.hardware['accelerator']}`. "
            f"Output: CUDA `float16` BCHW `{shape}`. "
            f"Dataset: {dataset}. "
            f"{seeds} "
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


def figure_captions(data: _PaperData) -> dict[str, str]:
    captions = {
        "batch-wait": "When a batch arrives late, the GPU sits idle and completes fewer steps in the same time.",
        "common-summary": (
            f"Ready batches per second and peak process GPU memory on {len(data.common)} common recipes. "
            "Each metric takes the median across seeds, then across recipes. "
            "Whiskers show the central 95% of batch-rate estimates from resampling whole seed jobs."
        ),
    }
    for family, paths in data.comparisons:
        count = len(_shared_recipes(data, [AX, *paths]))
        captions[f"pairwise-{family.lower()}"] = (
            f"AX vs {family} on {count} shared recipes with 95% seed bootstrap intervals for batch rate."
        )
    return captions


def render_figures(output: Path, paper_root: Path, data: _PaperData) -> None:
    generated = output / "generated"
    generated.mkdir()
    write_plot_inputs(generated, data)
    commands = {
        "batch-wait": r"\input{batch-wait.tex}",
        "common-summary": r"\input{delivery-plot.tex}",
        **{f"pairwise-{family.lower()}": rf"\LibraryComparison{{{family.lower()}}}" for family, _ in data.comparisons},
    }
    sources = ["batch-wait.tex", "delivery-plot.tex", "library-comparison.tex", "plot-colors.tex"]
    for filename in sources:
        shutil.copyfile(paper_root / filename, output / filename)
    source = output / "readme-figures.tex"
    source.write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{tikz}",
                r"\usepackage[active,tightpage]{preview}",
                r"\PreviewEnvironment{tikzpicture}",
                r"\setlength{\PreviewBorder}{8pt}",
                r"\input{plot-colors.tex}",
                r"\input{library-comparison.tex}",
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
