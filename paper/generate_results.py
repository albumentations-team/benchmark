# ruff: noqa: INP001
"""Generate the RGB paper and its audit table from validated production runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import TYPE_CHECKING

import yaml

from augbench.matrix import build_matrix
from augbench.recipes.load import load_recipe_catalog
from augbench.result_store import decode_result
from augbench.run_config import FamilyRunConfig
from augbench.run_records import RunRecord, build_run_record

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from augbench.recipes.models import RecipeCatalog, RecipeSpec
    from augbench.run_records import ResultRecord

RUN_ID = "3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c"
INVALID_DALI_FLIP_RUN = "75bc062fe770cfec3d857f401e16366deb01f6582c8645e952930e3cd3d788fb"
AX = "albumentationsx_cpu"
PILLOW_ALLSIMD = ("pillow_allsimd_cpu", "Pillow-AllSIMD CPU (fast)")
IMPLEMENTATIONS = (
    (AX, "AlbumentationsX"),
    ("pillow_cpu", "Pillow CPU"),
    ("torchvision_cpu", "TorchVision CPU"),
    ("torchvision_gpu", "TorchVision GPU"),
    ("kornia_cpu", "Kornia CPU"),
    ("kornia_gpu", "Kornia GPU"),
    ("dali_gpu", "DALI GPU"),
)
PAIRWISE = (
    ("Kornia", ("kornia_cpu", "kornia_gpu")),
    ("TorchVision", ("torchvision_cpu", "torchvision_gpu")),
    ("Pillow", ("pillow_cpu",)),
    ("DALI", ("dali_gpu",)),
)
RGB_2D_COVERAGE_DOMAINS = frozenset({"dropout_or_multi_image", "geometry", "pixel"})


@dataclass(frozen=True)
class _PairwiseRow:
    recipe: str
    implementation: str
    ax_speed: float
    competitor_speed: float
    ax_memory: float
    competitor_memory: float

    @property
    def ratio(self) -> float:
        return self.competitor_speed / self.ax_speed

    @property
    def memory_delta(self) -> float:
        return self.competitor_memory - self.ax_memory


@dataclass(frozen=True)
class _PaperData:
    groups: dict[tuple[str, str], list[ResultRecord]]
    recipes: tuple[RecipeSpec, ...]
    speed: dict[tuple[str, str], float]
    memory: dict[tuple[str, str], float]
    implementations: tuple[tuple[str, str], ...]
    comparisons: tuple[tuple[str, tuple[str, ...]], ...]
    common: tuple[str, ...]
    pairwise: dict[str, list[_PairwiseRow]]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    data, run, ax_run = load_report(args, root)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_metrics(args.output / "metrics.tex", data)
    write_plot_inputs(args.output, data)
    _write_csv(args.output / "recipe-results.csv", data)
    _write_recipe_catalog(args.output / "recipes.tex", data.recipes)
    _write_recipe_results(args.output / "recipe-results.tex", data)
    _write_versions(args.output / "versions.tex", data)
    _write_coverage(args.output, root / "catalog/operations.yaml")
    _write_sources(args.output / "sources.tex", run, ax_run)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="immutable run.json for the full RGB run")
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, help="full run's catalog; defaults to the repository catalog")
    parser.add_argument(
        "--ax-run", nargs=3, type=Path, metavar=("MANIFEST", "CELLS", "RECIPES"), help="optional AX-only rerun"
    )


def _run_catalog(run: RunRecord, path: Path) -> RecipeCatalog:
    if hashlib.sha256(path.read_bytes()).hexdigest() != run.inputs.recipe_catalog_sha256:
        raise ValueError(f"{path} does not match the recipe catalog checksum for {run.run_id}")
    return load_recipe_catalog(path)


def load_report(args: argparse.Namespace, root: Path) -> tuple[_PaperData, RunRecord, RunRecord | None]:
    run = RunRecord.model_validate_json(args.run.read_bytes())
    config = FamilyRunConfig.model_validate(run.family_config)
    base = {implementation for implementation, _ in IMPLEMENTATIONS}
    if config.measurements is not None or set(config.implementations) not in (base, base | {PILLOW_ALLSIMD[0]}):
        raise ValueError("report requires a complete RGB run")
    if run.run_id == INVALID_DALI_FLIP_RUN:
        raise ValueError("this run contains incorrect DALI VerticalFlip measurements")
    recipes = _run_catalog(run, args.recipes or root / config.recipes)
    records = load_run_results(args.cells, run, recipes)
    ax_run = None
    if args.ax_run:
        ax_run, ax_records = _load_ax_run(args.ax_run, run)
        records = [record for record in records if record.cell.implementation != AX] + ax_records
    data = _prepare(records, recipes.recipes)
    if not data.common:
        raise ValueError("the selected results have no recipes shared by all paths")
    return data, run, ax_run


def _load_ax_run(inputs: Sequence[Path], run: RunRecord) -> tuple[RunRecord, list[ResultRecord]]:
    manifest, cells, catalog = inputs
    ax_run = RunRecord.model_validate_json(manifest.read_bytes())
    if (
        ax_run.family_config != {**run.family_config, "implementations": [AX]}
        or ax_run.hardware != run.hardware
        or ax_run.inputs.dataset_archive_sha256 != run.inputs.dataset_archive_sha256
        or ax_run.inputs.recipe_catalog_sha256 != run.inputs.recipe_catalog_sha256
        or ax_run.inputs.environment_lock_sha256 != run.inputs.environment_lock_sha256
    ):
        raise ValueError("AX rerun must use the same workload, hardware, and package locks")
    recipes = _run_catalog(ax_run, catalog)
    return ax_run, load_run_results(cells, ax_run, recipes)


def load_run_results(cells_directory: Path, run: RunRecord, recipes: RecipeCatalog) -> list[ResultRecord]:
    identity = build_run_record(family_config=run.family_config, inputs=run.inputs, hardware=run.hardware)
    if identity != run:
        raise ValueError("run manifest does not match its immutable identity")
    cells = build_matrix(run_id=run.run_id, config=FamilyRunConfig.model_validate(run.family_config), recipes=recipes)
    records = [
        decode_result(path.read_bytes(), expected_cell_id=path.stem) for path in sorted(cells_directory.glob("*.json"))
    ]
    found = {record.cell_id for record in records}
    expected = {cell.cell_id for cell in cells}
    if found != expected:
        raise ValueError("results do not match the complete selected run matrix")
    config = run.family_config
    output = config["output"]
    execution = config["execution"]
    expected_shape = (execution["batch_size"], output["channels"], output["height"], output["width"])
    expected_items = execution["batch_size"] * execution["measured_batches"]
    for record in records:
        if record.output.shape != expected_shape or record.throughput.completed_items != expected_items:
            raise ValueError(f"cell {record.cell_id} violates the frozen output or timing window")
        if not math.isfinite(record.throughput.value) or not math.isfinite(record.gpu_memory.peak_mib):
            raise ValueError(f"cell {record.cell_id} has a non-finite metric")
    return records


def _prepare(records: Sequence[ResultRecord], recipes: tuple[RecipeSpec, ...]) -> _PaperData:
    groups: dict[tuple[str, str], list[ResultRecord]] = defaultdict(list)
    for record in records:
        groups[(record.cell.implementation, record.cell.recipe_id)].append(record)
    for group in groups.values():
        group.sort(key=lambda record: record.cell.seed)
    speed = {key: median(record.throughput.value for record in group) for key, group in groups.items()}
    memory = {key: median(record.gpu_memory.peak_mib for record in group) for key, group in groups.items()}
    support: dict[str, set[str]] = defaultdict(set)
    for implementation, recipe in groups:
        support[implementation].add(recipe)
    implementations: tuple[tuple[str, str], ...] = IMPLEMENTATIONS
    comparisons: tuple[tuple[str, tuple[str, ...]], ...] = PAIRWISE
    if PILLOW_ALLSIMD[0] in support:
        implementations = (*implementations, PILLOW_ALLSIMD)
        comparisons = tuple(
            (family, (*paths, PILLOW_ALLSIMD[0]) if family == "Pillow" else paths) for family, paths in comparisons
        )
    common = set.intersection(*(support[implementation] for implementation, _ in implementations))
    ordered = tuple(recipe.recipe_id for recipe in recipes if recipe.recipe_id in common)
    return _PaperData(
        dict(groups),
        recipes,
        speed,
        memory,
        implementations,
        comparisons,
        ordered,
        _pairwise(speed, memory, support, comparisons),
    )


def _pairwise(
    speed: dict[tuple[str, str], float],
    memory: dict[tuple[str, str], float],
    support: dict[str, set[str]],
    comparisons: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, list[_PairwiseRow]]:
    rows: dict[str, list[_PairwiseRow]] = {}
    for family, implementations in comparisons:
        available = set.union(*(support[implementation] for implementation in implementations))
        family_rows = []
        for recipe in sorted(support[AX] & available):
            candidates = [implementation for implementation in implementations if (implementation, recipe) in speed]
            # CPU precedes GPU in PAIRWISE, so max also gives CPU an exact tie.
            fastest = max(candidates, key=lambda implementation: speed[(implementation, recipe)])
            family_rows.append(
                _PairwiseRow(
                    recipe,
                    fastest,
                    speed[(AX, recipe)],
                    speed[(fastest, recipe)],
                    memory[(AX, recipe)],
                    memory[(fastest, recipe)],
                )
            )
        rows[family] = family_rows
    return rows


def _write_tex(path: Path, lines: Iterable[str]) -> None:
    path.write_text("% Generated by paper/generate_results.py. Do not edit by hand.\n" + "\n".join(lines) + "\n")


def _tex(value: str) -> str:
    replacements = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(character, character) for character in value)


def _name(value: str) -> str:
    parts = re.split(r"(?<=[a-z])(?=[A-Z])|(?<=[._/=(),])", value)
    return r"\allowbreak{}".join(_tex(part) for part in parts)


def _recipe_key(recipe: str, recipes: Sequence[RecipeSpec]) -> str:
    return f"R{next(index for index, item in enumerate(recipes, 1) if item.recipe_id == recipe):02d}"


def _write_metrics(path: Path, data: _PaperData) -> None:
    common_means = _mean_ratios(data, [implementation for implementation, _ in data.implementations])
    common_memories = _common_memories(data)
    metrics = {
        "ProductionCellCount": str(sum(map(len, data.groups.values()))),
        "ProductionPairCount": str(len(data.groups)),
        "RecipeCount": str(len(data.recipes)),
        "CommonRecipeCount": str(len(data.common)),
        "CommonMeanDaliRatio": f"{common_means['dali_gpu']:.2f}",
    }
    batch_size = _single({group[0].output.shape[0] for group in data.groups.values()})
    metrics.update(_common_headline_metrics(data, common_memories, batch_size))
    crop = "RandomCrop224+Normalize+ToTensor"
    for implementation, label in ((AX, "AX"), ("dali_gpu", "DALI")):
        if (implementation, crop) in data.speed:
            metrics[f"Crop{label}BatchRate"] = f"{data.speed[implementation, crop] / batch_size:.1f}"
    for family, paths in data.comparisons:
        compared = [AX, *paths]
        metrics[f"Plot{family}Shared"] = str(len(_shared_recipes(data, compared)))
    for family, rows in data.pairwise.items():
        metrics[f"Pair{family}Shared"] = str(len(rows))
        metrics[f"Pair{family}Wins"] = str(sum(row.ratio < 1 for row in rows))
        if family == "DALI":
            metrics["PairDALIFaster"] = str(sum(row.ratio > 1 for row in rows))
            observations = [
                (ax, dali)
                for row in rows
                for ax, dali in zip(data.groups[AX, row.recipe], data.groups["dali_gpu", row.recipe], strict=True)
            ]
            metrics["PairDALISeedCount"] = str(len(observations))
            metrics["PairDALISeedFaster"] = str(
                sum(dali.throughput.value > ax.throughput.value for ax, dali in observations)
            )
        metrics[f"Pair{family}Ratio"] = f"{median(row.ratio for row in rows):.2f}"
        metrics[f"Pair{family}MemoryDelta"] = f"{median(row.memory_delta for row in rows):+,.0f}"
    metrics["MemoryPollMs"] = str(
        _single({record.gpu_memory.poll_interval_ms for group in data.groups.values() for record in group})
    )
    if PILLOW_ALLSIMD in data.implementations:
        recipes, ratios = pillow_allsimd_ratios(data)
        metrics["AllSIMDShared"] = str(len(recipes))
        metrics["AllSIMDMeanRatio"] = f"{fmean(ratios):.2f}"
        metrics["AllSIMDMedianRatio"] = f"{median(ratios):.2f}"
        blur = next((recipe.recipe_id for recipe in data.recipes if recipe.augmentation_id == "MedianBlur"), None)
        if blur is not None and blur in recipes:
            metrics["AllSIMDMedianBlurRatio"] = f"{ratios[recipes.index(blur)]:.2f}"
    for recipe in data.recipes:
        if recipe.augmentation_id == "Shear":
            metrics["AXShearSpeed"] = f"{data.speed[(AX, recipe.recipe_id)]:,.0f}"
    _write_tex(path, (rf"\newcommand{{\{command}}}{{{value}}}" for command, value in metrics.items()))


def _common_headline_metrics(data: _PaperData, common_memories: Mapping[str, float], batch_size: int) -> dict[str, str]:
    metrics = {
        "CommonDALIFastest": str(
            sum(
                all(
                    data.speed["dali_gpu", recipe] > data.speed[other, recipe]
                    for other, _ in data.implementations
                    if other != "dali_gpu"
                )
                for recipe in data.common
            )
        ),
    }
    for implementation, command in ((AX, "AlbumentationsX"), ("dali_gpu", "DALIGPU")):
        speed = median(data.speed[implementation, recipe] for recipe in data.common)
        memory = common_memories[implementation]
        metrics[f"CommonMemory{command}"] = f"{memory:,.0f}"
        metrics[f"CommonBatchRate{command}"] = f"{speed / batch_size:.1f}"
        baselines = [
            [result.gpu_memory.baseline_mib for result in data.groups[implementation, recipe]] for recipe in data.common
        ]
        if all(value is not None for group in baselines for value in group):
            baseline = median(median(value for value in group if value is not None) for group in baselines)
            metrics[f"CommonBaseline{command}"] = f"{baseline:,.0f}"
    return metrics


def _shared_recipes(data: _PaperData, implementations: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        recipe.recipe_id
        for recipe in data.recipes
        if all((implementation, recipe.recipe_id) in data.speed for implementation in implementations)
    )


def write_plot_inputs(output: Path, data: _PaperData) -> None:
    batch_size = _single({group[0].output.shape[0] for group in data.groups.values()})
    labels = dict(data.implementations)
    labels.update(pillow_cpu="Pillow", pillow_allsimd_cpu="Pillow-AllSIMD (fast)", dali_gpu="DALI")
    rates = {
        path: median(data.speed[path, recipe] for recipe in data.common) / batch_size
        for path, _ in data.implementations
    }
    memories = _common_memories(data)
    ordered = sorted(rates, key=rates.__getitem__)
    lines = [
        rf"\def\DeliveryTopRow{{{len(ordered) - 1}}}",
        rf"\def\DeliveryAXRate{{{rates[AX]:.8f}}}",
        r"\def\DeliveryRows{%",
    ]
    for row, path in enumerate(ordered):
        color = _rate_color(rates[path], rates[AX])
        lines.append(
            rf"\DeliveryRow{{{row}}}{{{_tex(labels[path])}}}{{{rates[path]:.8f}}}{{{memories[path]}}}{{{color}}}"
        )
    _write_tex(output / "delivery-rows.tex", [*lines, "}"])
    _write_library_comparisons(output, data, batch_size)


def _rate_color(rate: float, ax_rate: float) -> str:
    if rate == ax_rate:
        return "ratereference"
    return "rateabove" if rate > ax_rate else "ratebelow"


def _write_library_comparisons(output: Path, data: _PaperData, batch_size: int) -> None:
    labels = dict(data.implementations)
    labels.update(pillow_allsimd_cpu="Pillow-AllSIMD (fast)", dali_gpu="DALI")
    for family, paths in data.comparisons:
        compared = (AX, *paths)
        recipes = _shared_recipes(data, compared)
        keys = ", ".join(_recipe_key(recipe, data.recipes) for recipe in recipes)
        rates = {path: median(data.speed[path, recipe] for recipe in recipes) / batch_size for path in compared}
        ordered = sorted(compared, key=rates.__getitem__, reverse=True)
        lines = [
            rf"\def\ComparisonRecipes{{{keys}}}",
            rf"\def\ComparisonTopRow{{{len(ordered) - 1}}}",
            r"\def\ComparisonRows{%",
        ]
        for row, path in enumerate(reversed(ordered)):
            rate = rates[path]
            memory = median(data.memory[path, recipe] for recipe in recipes)
            color = _rate_color(rate, rates[AX])
            result = "" if path == AX else _recipe_winner(data, path, labels[path], recipes)
            lines.append(
                rf"\ComparisonBar{{{row}}}{{{_tex(labels[path])}}}{{{rate:.8f}}}"
                rf"{{{memory:,.0f}}}{{{color}}}{{{_tex(result)}}}"
            )
        _write_tex(output / f"comparison-{family.lower()}.tex", [*lines, "}"])


def _recipe_winner(data: _PaperData, path: str, label: str, recipes: Sequence[str]) -> str:
    ax_wins = sum(data.speed[AX, recipe] > data.speed[path, recipe] for recipe in recipes)
    path_wins = sum(data.speed[path, recipe] > data.speed[AX, recipe] for recipe in recipes)
    if ax_wins == path_wins:
        return f"{ax_wins} wins each" if ax_wins else "All recipes tied"
    winner = "AX" if ax_wins > path_wins else label
    return f"{winner} faster on {max(ax_wins, path_wins)}/{len(recipes)}"


def pillow_allsimd_ratios(data: _PaperData) -> tuple[tuple[str, ...], list[float]]:
    recipes = _shared_recipes(data, ["pillow_cpu", PILLOW_ALLSIMD[0]])
    return recipes, [data.speed[(PILLOW_ALLSIMD[0], recipe)] / data.speed[("pillow_cpu", recipe)] for recipe in recipes]


def _mean_ratios(data: _PaperData, implementations: Sequence[str]) -> dict[str, float]:
    recipes = _shared_recipes(data, implementations)
    return {
        implementation: fmean(data.speed[(implementation, recipe)] / data.speed[(AX, recipe)] for recipe in recipes)
        for implementation in implementations
    }


def _common_memories(data: _PaperData) -> dict[str, float]:
    return {
        implementation: median(data.memory[(implementation, recipe)] for recipe in data.common)
        for implementation, _ in data.implementations
    }


def _single[T](values: set[T]) -> T:
    if len(values) != 1:
        raise ValueError(f"expected one frozen value, got {values}")
    return next(iter(values))


def _write_versions(path: Path, data: _PaperData) -> None:
    lines = []
    for implementation, label in data.implementations:
        versions = {
            r.runtime["library_version"]
            for (impl, _), group in data.groups.items()
            if impl == implementation
            for r in group
        }
        lines.append(rf"{label} & \texttt{{{_tex(_single(versions))}}} \\")
    _write_tex(path, lines)


def _write_sources(path: Path, run: RunRecord, ax_run: RunRecord | None) -> None:
    lines: list[str] = []
    for label, source in (("Full", run), ("AX", ax_run or run)):
        lines.extend(
            (
                rf"\newcommand{{\{label}RunID}}{{{source.run_id}}}",
                rf"\newcommand{{\{label}SourceRevision}}{{{source.inputs.git_commit}}}",
            )
        )
    _write_tex(path, lines)


def _csv_row(group: list[ResultRecord], recipe: RecipeSpec, data: _PaperData) -> dict[str, object]:
    implementation = group[0].cell.implementation
    key = (implementation, recipe.recipe_id)
    selected = any(
        row.implementation == implementation and row.recipe == recipe.recipe_id
        for rows in data.pairwise.values()
        for row in rows
    )
    run_ids = {record.run_id for record in group}
    row: dict[str, object] = {
        "run_id": _single(run_ids) if len(run_ids) == 1 else "",
        "implementation": implementation,
        "library_version": group[0].runtime["library_version"],
        "recipe_key": _recipe_key(recipe.recipe_id, data.recipes),
        "recipe_id": recipe.recipe_id,
        "common_intersection": recipe.recipe_id in data.common,
        "selected_pairwise_path": selected,
        "throughput_median_images_s": data.speed[key],
        "gpu_memory_median_mib": data.memory[key],
        "recipe_stages_json": json.dumps(
            [stage.model_dump(mode="json") for stage in recipe.stages], separators=(",", ":")
        ),
    }
    for record in group:
        seed = record.cell.seed
        row.update(
            {
                f"run_id_{seed}": record.run_id,
                f"cell_id_{seed}": record.cell_id,
                f"completed_images_{seed}": record.throughput.completed_items,
                f"duration_seconds_{seed}": record.throughput.duration_seconds,
                f"throughput_images_s_{seed}": record.throughput.value,
                f"gpu_memory_peak_mib_{seed}": record.gpu_memory.peak_mib,
                f"gpu_memory_baseline_mib_{seed}": record.gpu_memory.baseline_mib,
                f"gpu_memory_poll_ms_{seed}": record.gpu_memory.poll_interval_ms,
            }
        )
    return row


def _write_csv(path: Path, data: _PaperData) -> None:
    rows = [
        _csv_row(data.groups[(implementation, recipe.recipe_id)], recipe, data)
        for implementation, _ in data.implementations
        for recipe in data.recipes
        if (implementation, recipe.recipe_id) in data.groups
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parameters(parameters: Mapping[str, object]) -> str:
    return (
        "; ".join(rf"\texttt{{{_tex(key)}}}={_tex(json.dumps(value))}" for key, value in parameters.items())
        or "no parameters"
    )


def _write_recipe_catalog(path: Path, recipes: Sequence[RecipeSpec]) -> None:
    lines = []
    for recipe in recipes:
        key = _recipe_key(recipe.recipe_id, recipes)
        stages = recipe.stages[:-2]
        sequence = r" $\to$ ".join(_name(stage.operation_id) for stage in stages)
        details = [
            rf"\textbf{{{_name(stage.operation_id)}}}: {_parameters(stage.parameters)}. "
            rf"Sampling: {_tex(stage.randomness_scope)}."
            for stage in stages
        ]
        lines.append(rf"\hypertarget{{{key}}}{{\textbf{{{key}}}}} {sequence} & " + r"\newline ".join(details) + r" \\")
    _write_tex(path, lines)


def _write_recipe_results(path: Path, data: _PaperData) -> None:
    lines: list[str] = []
    for implementation, label in data.implementations:
        lines.extend(
            (
                rf"\subsection{{{label}}}",
                r"\begin{longtable}{lrrrrrr}",
                (
                    r"\textbf{Recipe} & \multicolumn{3}{c}{\textbf{Throughput (images/s)}} & "
                    r"\multicolumn{3}{c}{\textbf{GPU memory (MiB)}} \\"
                ),
                r" & Median & Min & Max & Median & Min & Max \\ \midrule \endhead",
            )
        )
        for recipe in data.recipes:
            group = data.groups.get((implementation, recipe.recipe_id))
            if group is None:
                continue
            speed = [record.throughput.value for record in group]
            memory = [record.gpu_memory.peak_mib for record in group]
            key = _recipe_key(recipe.recipe_id, data.recipes)
            values = [median(speed), min(speed), max(speed), median(memory), min(memory), max(memory)]
            lines.append(rf"\hyperlink{{{key}}}{{{key}}} & " + " & ".join(f"{v:,.0f}" for v in values) + r" \\")
        lines.append(r"\bottomrule\end{longtable}")
    _write_tex(path, lines)


def _write_coverage(output: Path, path: Path) -> None:
    catalog = yaml.safe_load(path.read_text())
    operations = [item for item in catalog["operations"] if item["domain"] in RGB_2D_COVERAGE_DOMAINS]
    counts = {"albumentationsx": len(operations)}
    mappings = []
    for library in ("kornia", "torchvision", "pillow"):
        statuses = {operation["support"][library]["status"] for operation in operations}
        if not statuses <= {"direct", "none"}:
            raise ValueError(f"unexpected {library} coverage status: {statuses}")
        counts[library] = sum(operation["support"][library]["status"] == "direct" for operation in operations)
    for operation in operations:
        cells = [_name(operation["operation_id"])]
        cells.extend(_name(operation["support"][lib]["api"] or "---") for lib in ("kornia", "torchvision", "pillow"))
        mappings.append(" & ".join(cells) + r" \\")
    names = {"albumentationsx": "AlbumentationsX", "kornia": "Kornia", "torchvision": "TorchVision", "pillow": "Pillow"}
    _write_tex(
        output / "coverage.tex",
        (rf"{names[lib]} & {count} & {catalog['versions'][lib]} \\" for lib, count in counts.items()),
    )
    _write_tex(output / "coverage-mapping.tex", mappings)
    _write_tex(
        output / "coverage-metrics.tex",
        [
            *(rf"\newcommand{{\Coverage{names[lib]}}}{{{count}}}" for lib, count in counts.items()),
            rf"\newcommand{{\CoverageCatalogSHA}}{{{hashlib.sha256(path.read_bytes()).hexdigest()}}}",
        ],
    )


if __name__ == "__main__":
    main()
