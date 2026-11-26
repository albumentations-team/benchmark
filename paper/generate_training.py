# ruff: noqa: INP001
"""Summarize the ImageNet training-step runs with seed-level intervals.

Each input is one complete run (one model). The generator validates every cell against its manifest and
catalog, then writes a JSON export, a CSV, and a Markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import median

from augbench.run_records import ResultRecord, RunRecord, TrainingObservation
from paper import figures
from paper.generate_results import AX, _command, _run_catalog, _tex, load_run_results
from paper.intervals import median_interval, ratio_interval

_LABELS = figures.LABELS
_RECIPE_PREFIX = "RandomResizedCrop+HorizontalFlip"
_FAMILY = "imagenet_train"
_MODEL_NAMES = {"resnet18": "ResNet-18", "resnet50": "ResNet-50"}
_MODEL_COMMANDS = {"resnet18": "ResNetEighteen", "resnet50": "ResNetFifty"}
# These paths augment on the GPU inside the training loop, after it receives the batch; every other path hands
# the loop a finished batch.
_AUGMENT_AFTER_TRANSFER = frozenset({"kornia_gpu", "torchvision_gpu"})


@dataclass(frozen=True)
class TrainingRun:
    run: RunRecord
    model: str
    batch_size: int
    recipes: tuple[str, ...]
    paths: tuple[str, ...]
    groups: dict[tuple[str, str], list[ResultRecord]]


@dataclass(frozen=True)
class Row:
    model: str
    batch_size: int
    recipe_id: str
    implementation: str
    seeds: int
    images_per_second_median: float
    images_per_second_low: float
    images_per_second_high: float
    interval_confidence: float
    ceiling_median: float
    of_ceiling_median: float
    of_ceiling_low: float
    of_ceiling_high: float
    loader_wait_fraction_median: float
    gpu_peak_mib_median: float
    gpu_steady_peak_mib_median: float
    ratio_to_ax: float | None = None
    ratio_to_ax_low: float | None = None
    ratio_to_ax_high: float | None = None
    ratio_to_ax_confidence: float | None = None
    ratio_to_ax_verdict: str | None = None


def _training(record: ResultRecord) -> TrainingObservation:
    if record.training is None:
        raise ValueError(f"cell {record.cell_id} lacks its training observation")
    return record.training


def load_training_run(manifest: Path, cells: Path, catalog: Path) -> TrainingRun:
    run = RunRecord.model_validate_json(manifest.read_bytes())
    if run.family_config["family"] != _FAMILY:
        raise ValueError(f"{manifest} is not a {_FAMILY} run")
    recipes = _run_catalog(run, catalog)
    records = load_run_results(cells, run, recipes)
    if missing := [record.cell_id for record in records if record.training is None]:
        raise ValueError(f"cells lack their training observation: {missing[:3]}")
    groups: dict[tuple[str, str], list[ResultRecord]] = {}
    for record in records:
        groups.setdefault((record.cell.implementation, record.cell.recipe_id), []).append(record)
    for group in groups.values():
        group.sort(key=lambda record: record.cell.seed)
    paths = tuple(path for path in run.family_config["implementations"] if any(key[0] == path for key in groups))
    return TrainingRun(
        run=run,
        model=_training(next(iter(groups.values()))[0]).model,
        batch_size=run.family_config["execution"]["batch_size"],
        recipes=tuple(recipe.recipe_id for recipe in recipes.recipes),
        paths=paths,
        groups=groups,
    )


def _speeds(records: list[ResultRecord]) -> list[float]:
    return [record.throughput.value for record in records]


def _of_ceiling(records: list[ResultRecord]) -> list[float]:
    """Each cell's rate over its own model-only ceiling; both come from one process, so machines cancel."""
    return [record.throughput.value / _training(record).ceiling_images_per_second for record in records]


def _row(training: TrainingRun, recipe: str, path: str, records: list[ResultRecord]) -> Row:
    speeds = _speeds(records)
    speed_interval = median_interval(speeds)
    share = _of_ceiling(records)
    share_interval = median_interval(share)
    reference = training.groups.get((AX, recipe))
    versus = None if path == AX or reference is None else ratio_interval(speeds, _speeds(reference))
    return Row(
        model=training.model,
        batch_size=training.batch_size,
        recipe_id=recipe,
        implementation=path,
        seeds=len(records),
        images_per_second_median=median(speeds),
        images_per_second_low=speed_interval.low,
        images_per_second_high=speed_interval.high,
        interval_confidence=speed_interval.confidence,
        ceiling_median=median(_training(record).ceiling_images_per_second for record in records),
        of_ceiling_median=median(share),
        of_ceiling_low=share_interval.low,
        of_ceiling_high=share_interval.high,
        loader_wait_fraction_median=median(
            _training(record).loader_wait_seconds / record.throughput.duration_seconds for record in records
        ),
        gpu_peak_mib_median=median(record.gpu_memory.peak_mib for record in records),
        gpu_steady_peak_mib_median=median(_training(record).steady_peak_mib for record in records),
        ratio_to_ax=None if versus is None else versus.ratio,
        ratio_to_ax_low=None if versus is None else versus.low,
        ratio_to_ax_high=None if versus is None else versus.high,
        ratio_to_ax_confidence=None if versus is None else versus.confidence,
        ratio_to_ax_verdict=None if versus is None else versus.verdict,
    )


def summarize(training: TrainingRun) -> list[Row]:
    return [
        _row(training, recipe, path, training.groups[path, recipe])
        for recipe in training.recipes
        for path in training.paths
        if (path, recipe) in training.groups
    ]


def _short(recipe: str) -> str:
    return recipe.removeprefix(_RECIPE_PREFIX).lstrip("+") or "Crop and flip only"


def recipe_label(recipe: str) -> str:
    """Name a ladder step by what it adds to the crop and flip that every recipe starts with."""
    added = recipe.removeprefix(_RECIPE_PREFIX).lstrip("+")
    return "+ " + added.replace("+", " + ") if added else "Crop + flip"


def _versus(row: Row) -> str:
    if row.ratio_to_ax is None:
        return "reference" if row.implementation == AX else "-"
    verdict = "tied" if row.ratio_to_ax_verdict == "indistinguishable" else row.ratio_to_ax_verdict
    return f"{row.ratio_to_ax:.2f}\u00d7 {verdict}"


def _table_line(row: Row) -> str:
    speed = (
        f"{row.images_per_second_median:,.0f} ({row.images_per_second_low:,.0f}\u2013{row.images_per_second_high:,.0f})"
    )
    share = f"{100 * row.of_ceiling_median:.0f}% ({100 * row.of_ceiling_low:.0f}\u2013{100 * row.of_ceiling_high:.0f})"
    return (
        f"| {_LABELS[row.implementation]} | {speed} | {share} | {_versus(row)} | "
        f"{100 * row.loader_wait_fraction_median:.0f}% | {row.gpu_peak_mib_median:,.0f} |"
    )


def markdown(summaries: list[tuple[TrainingRun, list[Row]]]) -> str:
    first, first_rows = summaries[0]
    execution = first.run.family_config["execution"]
    seeds = len(execution["seeds"])
    confidence = first_rows[0].interval_confidence
    intro = (
        f"Each cell trains on the batches an input path delivers and reports images per second over "
        f"{execution['measured_batches']} batches. Every recipe starts with RandomResizedCrop and HorizontalFlip. "
        f"Ranges are exact order-statistic intervals over {seeds} seeds; they cover the median with "
        f"{100 * confidence:.2f}% probability. *Of ceiling* divides each cell's rate by the model-only rate "
        "measured in the same process. *vs AX* is the ratio of medians with an exact 95% interval; *tied* means "
        "the interval includes 1.00."
    )
    lines = ["# ImageNet training-step results", "", *textwrap.wrap(intro, width=115)]
    for training, rows in summaries:
        ceilings = [
            _training(record).ceiling_images_per_second for records in training.groups.values() for record in records
        ]
        lines += [
            "",
            f"## {_MODEL_NAMES[training.model]}, batch {training.batch_size}",
            "",
            (
                f"Model-only ceiling: {median(ceilings):,.0f} images/s "
                f"(range {min(ceilings):,.0f}\u2013{max(ceilings):,.0f} over {len(ceilings)} cells)."
            ),
        ]
        for recipe in training.recipes:
            ranked = sorted((row for row in rows if row.recipe_id == recipe), key=lambda r: -r.images_per_second_median)
            lines += [
                "",
                f"### {_short(recipe)}",
                "",
                "| Path | Images/s | Of ceiling | vs AX | Loader wait | Peak VRAM (MiB) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *(_table_line(row) for row in ranked),
            ]
    lines += [
        "",
        "Loader wait is the share of the window the training loop spent pulling the next batch. GPU work that a path",
        "runs after the batch arrives (Kornia GPU, TorchVision GPU) belongs to the step, and DALI's wait includes",
        "its own GPU work for that batch.",
    ]
    return "\n".join(lines) + "\n"


def _range(median_value: float, low: float, high: float, scale: float = 1.0) -> str:
    return f"{scale * median_value:,.0f} ({scale * low:,.0f}--{scale * high:,.0f})"


def _latex_row(recipe: str, row: Row) -> str:
    cells = (
        _tex(recipe_label(recipe)),
        _tex(_LABELS[row.implementation]),
        _range(row.images_per_second_median, row.images_per_second_low, row.images_per_second_high),
        _range(row.of_ceiling_median, row.of_ceiling_low, row.of_ceiling_high, scale=100),
        f"{100 * row.loader_wait_fraction_median:.0f}",
        f"{row.gpu_peak_mib_median:,.0f}",
    )
    return " & ".join(cells) + r" \\"


def _training_metrics(summaries: list[tuple[TrainingRun, list[Row]]]) -> dict[str, str]:
    cells = sum(len(group) for training, _ in summaries for group in training.groups.values())
    metrics = {
        "TrainCellCount": str(cells),
        "TrainWarmupBatches": str(summaries[0][0].run.family_config["execution"]["warmup_batches"]),
        "TrainMeasuredBatches": str(summaries[0][0].run.family_config["execution"]["measured_batches"]),
    }
    for training, rows in summaries:
        model = _MODEL_COMMANDS[training.model]
        ceilings = [
            _training(record).ceiling_images_per_second for group in training.groups.values() for record in group
        ]
        metrics[f"TrainCeiling{model}"] = f"{median(ceilings):,.0f}"
        metrics[f"TrainBatch{model}"] = str(training.batch_size)
        metrics[f"TrainRunID{model}"] = training.run.run_id
        metrics[f"TrainSource{model}"] = training.run.inputs.git_commit
        before_transfer = [row.of_ceiling_median for row in rows if row.implementation not in _AUGMENT_AFTER_TRANSFER]
        metrics[f"TrainReadyFloor{model}"] = f"{100 * min(before_transfer):.0f}"
        for path in training.paths:
            shares = [row.of_ceiling_median for row in rows if row.implementation == path]
            metrics[f"TrainMin{model}{_command(path)}"] = f"{100 * min(shares):.0f}"
            metrics[f"TrainMax{model}{_command(path)}"] = f"{100 * max(shares):.0f}"
    return metrics


def write_paper_outputs(summaries: list[tuple[TrainingRun, list[Row]]], output: Path) -> None:
    """Write the training figure, its macros, and the appendix table for the paper."""
    output.mkdir(parents=True, exist_ok=True)
    recipes = summaries[0][0].recipes
    panels = [(f"{_MODEL_NAMES[training.model]}, batch {training.batch_size}", rows) for training, rows in summaries]
    (output / "fig-training.tex").write_text(figures.training_shares(panels, [recipe_label(r) for r in recipes]))
    header = "% Generated by paper/generate_training.py. Do not edit by hand.\n"
    macros = (rf"\newcommand{{\{name}}}{{{value}}}" for name, value in _training_metrics(summaries).items())
    (output / "training-metrics.tex").write_text(header + "\n".join(macros) + "\n")
    lines = []
    for training, rows in summaries:
        title = f"{_MODEL_NAMES[training.model]}, batch {training.batch_size}"
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{title}}}}} \\ \midrule")
        for recipe in training.recipes:
            ranked = sorted((r for r in rows if r.recipe_id == recipe), key=lambda r: -r.images_per_second_median)
            lines.extend(_latex_row(recipe, row) for row in ranked)
            lines.append(r"\midrule")
    (output / "training-results.tex").write_text(header + "\n".join(lines[:-1]) + "\n")


def write_outputs(runs: list[TrainingRun], output: Path, paper_output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summaries = [(training, summarize(training)) for training in runs]
    rows = [row for _training_run, run_rows in summaries for row in run_rows]
    with (output / "training-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(Row)], lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    export = {
        "schema_version": 1,
        "runs": [
            {
                "model": training.model,
                "batch_size": training.batch_size,
                "run_id": training.run.run_id,
                "warmup_batches": training.run.family_config["execution"]["warmup_batches"],
                "measured_batches": training.run.family_config["execution"]["measured_batches"],
            }
            for training in runs
        ],
        "results": [asdict(row) for row in rows],
    }
    (output / "training-results.json").write_text(json.dumps(export, indent=2, allow_nan=False) + "\n")
    (output / "training-summary.md").write_text(markdown(summaries))
    write_paper_outputs(summaries, paper_output)


def main() -> None:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        required=True,
        type=Path,
        metavar=("MANIFEST", "CELLS", "RECIPES"),
        help="one complete run: its run.json, its cells directory, and its recipe catalog",
    )
    parser.add_argument("--output", type=Path, default=root / "docs/generated")
    parser.add_argument("--paper-output", type=Path, default=root / "paper/generated")
    args = parser.parse_args()
    write_outputs([load_training_run(*inputs) for inputs in args.run], args.output, args.paper_output)


if __name__ == "__main__":
    main()
