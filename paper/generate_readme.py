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
from typing import TYPE_CHECKING, Any

from PIL import Image

from augbench.run_config import FamilyRunConfig
from paper.figures import LABELS
from paper.generate_results import (
    AX,
    PILLOW_ALLSIMD,
    RUN_ID,
    _dali_decode_metrics,
    _mean_ratios,
    _shared_recipes,
    _single,
    add_run_arguments,
    load_report,
    pillow_allsimd_ratios,
    ratio_to_ax,
    write_figures,
)
from paper.generate_training import recipe_label
from paper.intervals import median_interval

if TYPE_CHECKING:
    from collections.abc import Mapping

    from augbench.run_records import RunRecord
    from paper.generate_results import _PaperData

_BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
_END = "<!-- END GENERATED RESULTS -->"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument(
        "--training-results",
        type=Path,
        default=root / "docs/generated/training-results.json",
        help="export written by paper.generate_training, which also writes the training figure source",
    )
    args = parser.parse_args()
    data, run = load_report(args, root)
    training = json.loads(args.training_results.read_text())
    captions = figure_captions(data)
    readme = root / "README.md"
    updated = replace_results(readme.read_text(), readme_results(data, run, training))
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
                    **export_results(data, run, data.supplemental_runs),
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


# Each implementation, or each implementation and seed, ran in its own worker process.
_ISOLATED_SCOPES = {"implementation", "implementation_seed"}


def _isolated_workers(data: _PaperData) -> bool:
    return all(
        record.runtime.get("worker_scope") in _ISOLATED_SCOPES for records in data.groups.values() for record in records
    )


def export_results(
    data: _PaperData,
    run: RunRecord,
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
            interval = median_interval(speeds)
            versus = None if implementation == AX else ratio_to_ax(data, implementation, recipe.recipe_id)
            results[implementation] = {
                "throughput": {
                    "median": data.speed[key],
                    "min": min(speeds),
                    "max": max(speeds),
                    "interval": {"low": interval.low, "high": interval.high, "confidence": interval.confidence},
                },
                **(
                    {
                        "ratio_to_ax": {
                            "ratio": versus.ratio,
                            "low": versus.low,
                            "high": versus.high,
                            "confidence": versus.confidence,
                            "verdict": versus.verdict,
                        }
                    }
                    if versus
                    else {}
                ),
                "gpu_memory": {"median": data.memory[key], "min": min(memories), "max": max(memories)},
                "observations": [
                    {
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
    isolated_workers = _isolated_workers(data)
    measurement_notes = (
        ["Each implementation ran in its own process. The baseline is sampled before pipeline construction."]
        if isolated_workers
        else []
    )
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
        if not isolated_workers:
            measurement_notes.append(
                "Pillow-AllSIMD ran in its own worker process. Other paths share the base worker, so their "
                "NVML peaks include a different process lifetime."
            )
    return {
        "schema_version": 2,
        "run": run.model_dump(mode="json"),
        "supplemental_runs": [source.model_dump(mode="json") for source in supplemental_runs],
        "is_published_run": not supplemental_runs and run.run_id == RUN_ID,
        "measurement_count": sum(map(len, data.groups.values())),
        "gpu_memory_poll_ms": _single(
            {record.gpu_memory.poll_interval_ms for records in data.groups.values() for record in records}
        ),
        "libraries": libraries,
        "recipes": recipes,
        "summaries": summaries,
        "ax_verdicts": _ax_verdicts(data),
        "pairwise_comparisons": pairwise_comparisons,
        "measurement_notes": measurement_notes,
    }


def _ax_verdicts(data: _PaperData) -> list[dict[str, object]]:
    """Count recipes where each path is faster, slower, or indistinguishable from AX over seeds."""
    verdicts = []
    for implementation, _ in data.implementations:
        if implementation == AX:
            continue
        outcomes = [
            ratio_to_ax(data, implementation, recipe.recipe_id).verdict
            for recipe in data.recipes
            if (implementation, recipe.recipe_id) in data.groups and (AX, recipe.recipe_id) in data.groups
        ]
        verdicts.append(
            {
                "id": implementation,
                "recipes": len(outcomes),
                "faster": outcomes.count("faster"),
                "indistinguishable": outcomes.count("indistinguishable"),
                "slower": outcomes.count("slower"),
            }
        )
    return verdicts


def replace_results(readme: str, generated: str) -> str:
    if readme.count(_BEGIN) != 1 or readme.count(_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    before, remaining = readme.split(_BEGIN)
    _, after = remaining.split(_END)
    return before + _BEGIN + "\n\n" + generated.rstrip() + "\n\n" + _END + after


def _versions(data: _PaperData) -> dict[str, str]:
    return {
        implementation: _single(
            {
                record.runtime["library_version"]
                for (path, _), group in data.groups.items()
                if path == implementation
                for record in group
            }
        )
        for implementation, _ in data.implementations
    }


def readme_results(data: _PaperData, run: RunRecord, training: Mapping[str, object]) -> str:
    config = FamilyRunConfig.model_validate(run.family_config)
    captions = figure_captions(data)
    versions = _versions(data)
    paths = [implementation for implementation, _ in data.implementations]
    common = {path: [data.speed[path, recipe] for recipe in data.common] for path in paths}
    memory = {path: median(data.memory[path, recipe] for recipe in data.common) for path in paths}
    lines = [
        "## Results",
        "",
        *_key_results(data, common, memory, training),
        "",
        (
            f"{sum(map(len, data.groups.values())):,} input-pipeline measurements cover {len(data.groups)} "
            f"path\N{EN DASH}recipe pairs with {len(config.execution.seeds)} seeds each. Every measurement "
            "records throughput and peak process GPU memory in the same pass. "
            "[Per-seed results](docs/generated/results.json) \N{MIDDLE DOT} "
            "[training-step results](docs/generated/training-summary.md)"
        ),
        "",
        "### Input pipeline alone",
        "",
        (
            "The timer runs from JPEG files on local disk to a normalized CUDA `float16` batch of "
            f"{_TIMES.join(map(str, (config.execution.batch_size, *config.output.shape)))}. "
            "Each path uses its own decoder and places its work on the CPU or the GPU."
        ),
        "",
        _image("common-summary", captions["common-summary"]),
        "",
        f"Medians over the {len(data.common)} recipes that all eight paths support:",
        "",
        "| Path | Version | Images/s | Slowest to fastest recipe | Peak GPU memory (MiB) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {LABELS[path]} | {versions[path]} | {median(common[path]):,.0f} | "
        f"{min(common[path]):,.0f}\N{EN DASH}{max(common[path]):,.0f} | {memory[path]:,.0f} |"
        for path in sorted(paths, key=lambda path: -median(common[path]))
    )
    lines += [
        "",
        "### Each path against AlbumentationsX",
        "",
        (
            "The author maintains AlbumentationsX (AX), and AX transforms define the reference effect of each recipe. "
            "This section therefore compares every path with AX on the recipes both support. A recipe counts as "
            "faster or slower only when the exact 95% interval over seeds excludes equal throughput; the rest are "
            "tied."
        ),
        "",
        _image("ax-pairs", captions["ax-pairs"]),
        "",
        "| Path | Shared recipes | Path faster | Tied | AX faster | Median ratio, path / AX |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def typical_ratio(path: str) -> float:
        return median(data.speed[path, recipe] / data.speed[AX, recipe] for recipe in _shared_recipes(data, [AX, path]))

    for path in sorted((path for path in paths if path != AX), key=typical_ratio, reverse=True):
        shared = _shared_recipes(data, [AX, path])
        verdicts = [ratio_to_ax(data, path, recipe).verdict for recipe in shared]
        ratio = median(data.speed[path, recipe] / data.speed[AX, recipe] for recipe in shared)
        lines.append(
            f"| {LABELS[path]} | {len(shared)} | {verdicts.count('faster')} | {verdicts.count('indistinguishable')} | "
            f"{verdicts.count('slower')} | {ratio:.2f} |"
        )
    lines += ["", *_training_lines(training), "", *_workload_lines(data, run, config, training)]
    return "\n".join(lines)


def _dali_decoder_note(data: _PaperData) -> str:
    decode = _dali_decode_metrics(data)
    return (
        " DALI decodes only the crop window when a recipe starts with a crop: "
        f"{decode['DALIWindowMedian']} images/s on {decode['DALIWindowRecipes']} recipes, against "
        f"{decode['DALIFullMedian']} on the {decode['DALIFullRecipes']} that need the whole image."
    )


def _key_results(
    data: _PaperData, common: Mapping[str, list[float]], memory: Mapping[str, float], training: Mapping[str, Any]
) -> list[str]:
    fastest = max(common, key=lambda path: median(common[path]))
    cpu = [path for path in common if path.endswith("_cpu")]
    cpu_speeds = [median(common[path]) for path in cpu]
    cpu_memory = [memory[path] for path in cpu]
    lines = [
        (
            f"- **Input pipeline alone.** {LABELS[fastest]} delivers the most images per second: "
            f"{median(common[fastest]):,.0f}, the median over the {len(data.common)} recipes all paths share, "
            f"while holding {memory[fastest]:,.0f} MiB of GPU memory. The CPU paths deliver "
            f"{min(cpu_speeds):,.0f} to {max(cpu_speeds):,.0f} images/s and hold "
            f"{min(cpu_memory):,.0f} to {max(cpu_memory):,.0f} MiB."
            f"{_dali_decoder_note(data) if fastest == 'dali_gpu' else ''}"
        )
    ]
    for item in training["runs"]:
        rows = [row for row in training["results"] if row["model"] == item["model"]]
        ceiling = median(row["ceiling_median"] for row in rows)
        floors = {
            path: min(row["of_ceiling_median"] for row in rows if row["implementation"] == path)
            for path in dict.fromkeys(row["implementation"] for row in rows)
        }
        fed = [LABELS[path] for path, share in floors.items() if share >= _FED]
        behind = sorted((share, path) for path, share in floors.items() if share < _FED)
        kept = (
            f"{_series(fed)} {'keeps' if len(fed) == 1 else 'keep'} the step at {100 * _FED:.0f}% or more of that "
            "rate on every recipe."
            if fed
            else f"No path keeps the step at {100 * _FED:.0f}% or more of that rate on every recipe."
        )
        lines.append(
            f"- **{_MODEL_NAMES[item['model']]} training step ({ceiling:,.0f} images/s without input).** "
            + kept
            + (
                f" On their heaviest recipe the {'others' if fed else 'paths'} reach "
                + _series([f"{100 * share:.0f}% ({LABELS[path]})" for share, path in behind])
                + "."
                if behind
                else ""
            )
        )
    return lines


def _series(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _training_lines(training: Mapping[str, Any]) -> list[str]:
    runs = {item["model"]: item for item in training["runs"]}
    rows = training["results"]
    recipes = list(dict.fromkeys(row["recipe_id"] for row in rows))
    lines = [
        "### Feeding a training step",
        "",
        (
            "Here every delivered batch feeds one training step of an untrained torchvision ResNet: "
            "`torch.compile`, bfloat16 autocast, SGD, on the same L4. The *ceiling* is the same step on a batch "
            "already in GPU memory, measured in the same process before the loader starts. A path at 100% keeps the "
            "model fully fed; below that, the model waits for input or shares the GPU with the path's own work."
        ),
        "",
        _image("training", "Training-step throughput as a share of the model-only ceiling, per path and recipe"),
    ]
    for model, run in runs.items():
        model_rows = [row for row in rows if row["model"] == model]
        ceiling = median(row["ceiling_median"] for row in model_rows)
        lines += [
            "",
            (
                f"**{_MODEL_NAMES[model]}, batch {run['batch_size']}.** Ceiling {ceiling:,.0f} images/s. "
                "Each cell is the share of the ceiling, with images/s in parentheses."
            ),
            "",
            "| Path | " + " | ".join(recipe_label(recipe) for recipe in recipes) + " |",
            "| --- | " + " | ".join("---:" for _ in recipes) + " |",
        ]
        paths = list(dict.fromkeys(row["implementation"] for row in model_rows))
        paths.sort(key=lambda path: -median(r["of_ceiling_median"] for r in model_rows if r["implementation"] == path))
        for path in paths:
            cells = []
            for recipe in recipes:
                row = next((r for r in model_rows if r["implementation"] == path and r["recipe_id"] == recipe), None)
                cells.append(
                    "\N{EN DASH}"
                    if row is None
                    else f"{100 * row['of_ceiling_median']:.0f}% ({row['images_per_second_median']:,.0f})"
                )
            lines.append(f"| {LABELS[path]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "A dash means the path has no implementation of that recipe. Kornia GPU and TorchVision GPU augment on "
            "the GPU inside the training loop, so their work takes GPU time from the step."
        ),
    ]
    return lines


_MODEL_NAMES = {"resnet18": "ResNet-18", "resnet50": "ResNet-50"}
_TIMES = "\N{MULTIPLICATION SIGN}"
_FED = 0.95


def _training_window(training: Mapping[str, Any]) -> str:
    windows = {(run["warmup_batches"], run["measured_batches"]) for run in training["runs"]}
    ((warmup, measured),) = windows
    return f"{warmup} warm-up and {measured} measured batches"


def _workload_lines(
    data: _PaperData, run: RunRecord, config: FamilyRunConfig, training: Mapping[str, Any]
) -> list[str]:
    execution = config.execution
    poll_ms = _single({record.gpu_memory.poll_interval_ms for records in data.groups.values() for record in records})
    return [
        "### Measurement settings",
        "",
        (
            f"One Standard `{run.hardware['machine_type']}` VM with one `{run.hardware['accelerator']}` per job. "
            f"Input pipeline: {config.dataset.item_count:,} sorted ImageNet validation JPEGs (all but one CMYK file), "
            f"{execution.warmup_batches} warm-up batches and {execution.measured_batches} measured batches "
            f"({execution.batch_size * execution.measured_batches:,} images). Training step: the same files, "
            f"{_training_window(training)}. Seeds: "
            f"{', '.join(map(str, execution.seeds))}; every path reads the same file order for a given seed. "
            f"DataLoader paths use {execution.num_workers} workers with prefetch factor {execution.prefetch_factor}; "
            "DALI uses the same number of threads."
        ),
        "",
        (
            f"NVML samples the process's GPU memory every {poll_ms} ms from before pipeline construction to the end "
            "of the cell. Short peaks between samples can be missed. CPU memory is not measured. Training runs: "
            + ", ".join(f"`{item['run_id'][:12]}` ({_MODEL_NAMES[item['model']]})" for item in training["runs"])
            + f". Input-pipeline run: `{run.run_id[:12]}`."
        ),
    ]


def _image(name: str, description: str) -> str:
    return f"![{description}](docs/generated/{name}.png)"


def figure_captions(data: _PaperData) -> dict[str, str]:
    captions = {
        "batch-wait": "When a batch arrives late, the GPU sits idle and completes fewer steps in the same time.",
        "common-summary": (
            f"Images per second on the {len(data.common)} recipes all paths share: one dot per recipe, the bar marks "
            "the median. Right: peak process GPU memory."
        ),
        "ax-pairs": (
            "Each path's throughput divided by AX's on every recipe both support. Marks show whether "
            "the exact 95% interval over seeds excludes equal throughput."
        ),
        "training": "Training-step throughput as a share of the model-only ceiling, with the seed interval.",
    }
    for family, paths in data.comparisons:
        count = len(_shared_recipes(data, [AX, *paths]))
        captions[f"pairwise-{family.lower()}"] = f"{family} against AX on {count} shared recipes, per recipe"
    return captions


def render_figures(output: Path, paper_root: Path, data: _PaperData) -> None:
    """Render the paper's figure sources to PNG; the training figure comes from paper.generate_training."""
    generated = output / "generated"
    generated.mkdir()
    write_figures(generated, data)
    shutil.copyfile(paper_root / "generated/fig-training.tex", generated / "fig-training.tex")
    commands = {
        "batch-wait": r"\input{batch-wait.tex}",
        "common-summary": r"\input{generated/fig-rgb-overview.tex}",
        "ax-pairs": r"\input{generated/fig-ax-pairs.tex}",
        "training": r"\input{generated/fig-training.tex}",
        **{
            f"pairwise-{family.lower()}": rf"\input{{generated/fig-pairwise-{family.lower()}.tex}}"
            for family, _ in data.comparisons
        },
    }
    for filename in ("batch-wait.tex", "plot-colors.tex"):
        shutil.copyfile(paper_root / filename, output / filename)
    source = output / "readme-figures.tex"
    source.write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{pgfplots}",
                r"\pgfplotsset{compat=1.18}",
                r"\usepackage[active,tightpage]{preview}",
                r"\PreviewEnvironment{tikzpicture}",
                r"\setlength{\PreviewBorder}{8pt}",
                r"\input{plot-colors.tex}",
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
