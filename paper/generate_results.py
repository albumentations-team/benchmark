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
    from augbench.run_records import CellKey, ResultRecord

RUN_ID = "3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c"
DALI_DOUBLE_FLIP_RUN = "75bc062fe770cfec3d857f401e16366deb01f6582c8645e952930e3cd3d788fb"
AX = "albumentationsx_cpu"
DALI_VERTICAL_FLIP = ("dali_gpu", "RandomCrop224+VerticalFlip+Normalize+ToTensor")
AX_CHANNEL_DROPOUT = (AX, "RandomCrop224+ChannelDropout+Normalize+ToTensor")
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
    data, run, ax_run, correction_runs = load_report(args, root)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_metrics(args.output / "metrics.tex", data)
    _write_heatmap(args.output / "common-heatmap.tex", data)
    _write_mean_bars(
        args.output / "common-mean-bars.tex",
        _mean_ratios(data, [implementation for implementation, _ in data.implementations]),
    )
    _write_memory_bars(args.output / "common-memory-bars.tex", data)
    for family, paths in data.comparisons:
        _write_mean_bars(args.output / f"pairwise-{family.lower()}.tex", _mean_ratios(data, [AX, *paths]))
    _write_csv(args.output / "recipe-results.csv", data)
    _write_recipe_catalog(args.output / "recipes.tex", data.recipes)
    _write_recipe_results(args.output / "recipe-results.tex", data)
    _write_versions(args.output / "versions.tex", data)
    _write_coverage(args.output, root / "catalog/operations.yaml")
    _write_sources(args.output / "sources.tex", run, ax_run, correction_runs)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="immutable run.json for the full RGB run")
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, help="full run's catalog; defaults to the repository catalog")
    parser.add_argument(
        "--ax-run", nargs=3, type=Path, metavar=("MANIFEST", "CELLS", "RECIPES"), help="optional AX-only rerun"
    )
    parser.add_argument(
        "--correction-run",
        action="append",
        nargs=3,
        type=Path,
        metavar=("MANIFEST", "CELLS", "RECIPES"),
        help="rerun of the corrected DALI VerticalFlip and non-AX Shear cells; repeat for multiple runs",
    )
    parser.add_argument(
        "--extension-run",
        action="append",
        nargs=3,
        type=Path,
        metavar=("MANIFEST", "CELLS", "RECIPES"),
        help="additional measurements for the expanded RGB catalog; repeat for multiple runs",
    )


def _run_catalog(run: RunRecord, path: Path) -> RecipeCatalog:
    if hashlib.sha256(path.read_bytes()).hexdigest() != run.inputs.recipe_catalog_sha256:
        raise ValueError(f"{path} does not match the recipe catalog checksum for {run.run_id}")
    return load_recipe_catalog(path)


def load_report(
    args: argparse.Namespace, root: Path
) -> tuple[_PaperData, RunRecord, RunRecord | None, tuple[RunRecord, ...]]:
    run = RunRecord.model_validate_json(args.run.read_bytes())
    config = FamilyRunConfig.model_validate(run.family_config)
    base = {implementation for implementation, _ in IMPLEMENTATIONS}
    if config.measurements is not None or set(config.implementations) not in (base, base | {PILLOW_ALLSIMD[0]}):
        raise ValueError("report requires a complete RGB run")
    recipes = _run_catalog(run, args.recipes or root / config.recipes)
    records = load_run_results(args.cells, run, recipes)
    ax_run = None
    if args.ax_run:
        ax_run, ax_recipes, ax_records = _load_ax_run(args.ax_run, run)
        records = _select_ax_rerun(recipes, ax_recipes, records, ax_records)
        recipes = ax_recipes
    correction_runs: tuple[RunRecord, ...] = ()
    if args.correction_run:
        if run.run_id != DALI_DOUBLE_FLIP_RUN or ax_run is None:
            raise ValueError("the flip and shear correction requires the November full run and the AX rerun")
        sources = [_load_additional_run(inputs, run, ax_run, recipes) for inputs in args.correction_run]
        correction_runs = tuple(source for source, _ in sources)
        records = _restore_corrected_cells(
            records,
            [record for _, batch in sources for record in batch],
            build_matrix(run_id=run.run_id, config=config, recipes=recipes),
        )
    elif run.run_id == DALI_DOUBLE_FLIP_RUN:
        raise ValueError("this full run requires the corrected flip and shear measurements")
    if args.extension_run:
        expanded = load_recipe_catalog(root / config.recipes)
        sources = [_load_additional_run(inputs, run, ax_run or run, expanded) for inputs in args.extension_run]
        records = _extend_matrix(
            records, [record for _, batch in sources for record in batch], recipes, expanded, config
        )
        recipes = expanded
        correction_runs = (*correction_runs, *(source for source, _ in sources))
    data = _prepare(records, recipes.recipes)
    if not data.common:
        raise ValueError("the selected results have no recipes shared by all paths")
    return data, run, ax_run, correction_runs


def _load_ax_run(inputs: Sequence[Path], run: RunRecord) -> tuple[RunRecord, RecipeCatalog, list[ResultRecord]]:
    manifest, cells, catalog = inputs
    ax_run = RunRecord.model_validate_json(manifest.read_bytes())
    if ax_run.family_config != {**run.family_config, "implementations": [AX]} or ax_run.hardware != run.hardware:
        raise ValueError("AX rerun must use the same workload and hardware settings")
    recipes = _run_catalog(ax_run, catalog)
    return ax_run, recipes, load_run_results(cells, ax_run, recipes)


def _load_additional_run(
    inputs: Sequence[Path], run: RunRecord, ax_run: RunRecord, recipes: RecipeCatalog
) -> tuple[RunRecord, list[ResultRecord]]:
    manifest, cells, catalog = inputs
    correction = RunRecord.model_validate_json(manifest.read_bytes())
    implementations = correction.family_config["implementations"]
    workload = {key: value for key, value in correction.family_config.items() if key != "measurements"}
    if (
        workload != {**run.family_config, "implementations": implementations}
        or not set(implementations) <= set(run.family_config["implementations"])
        or correction.hardware != run.hardware
        or correction.inputs.environment_lock_sha256 != ax_run.inputs.environment_lock_sha256
    ):
        raise ValueError("additional run differs in workload, hardware settings, or package locks")
    correction_recipes = _run_catalog(correction, catalog)
    by_recipe = {recipe.recipe_id: recipe for recipe in recipes.recipes}
    for recipe in correction_recipes.recipes:
        original = by_recipe.get(recipe.recipe_id)
        if original is None or recipe.stages != original.stages or recipe.channels != original.channels:
            raise ValueError(f"additional run changed the definition of {recipe.recipe_id}")
    return correction, load_run_results(cells, correction, correction_recipes, require_complete=False)


def _restore_corrected_cells(
    records: list[ResultRecord], corrected: list[ResultRecord], expected: Sequence[CellKey]
) -> list[ResultRecord]:
    retained = [
        record for record in records if (record.cell.implementation, record.cell.recipe_id) != DALI_VERTICAL_FLIP
    ]
    return _fill_matrix(retained, corrected, expected)


def _extend_matrix(
    records: list[ResultRecord],
    added: list[ResultRecord],
    previous: RecipeCatalog,
    expanded: RecipeCatalog,
    config: FamilyRunConfig,
) -> list[ResultRecord]:
    current = {recipe.recipe_id: recipe for recipe in expanded.recipes}
    unchanged = {
        recipe.recipe_id
        for recipe in previous.recipes
        if recipe.recipe_id in current and recipe.stages == current[recipe.recipe_id].stages
    }
    raw_crop = {
        recipe.recipe_id
        for recipe in expanded.recipes
        if any(stage.operation_id == "RandomCrop224" for stage in recipe.stages)
    }
    retained = [
        record
        for record in records
        if record.cell.recipe_id in unchanged
        and (record.cell.implementation, record.cell.recipe_id) != AX_CHANNEL_DROPOUT
        and (record.cell.implementation != "dali_gpu" or record.cell.recipe_id not in raw_crop)
    ]
    expected = build_matrix(run_id=records[0].run_id, config=config, recipes=expanded)
    return _fill_matrix(retained, added, expected)


def _fill_matrix(
    retained: list[ResultRecord], added: list[ResultRecord], expected: Sequence[CellKey]
) -> list[ResultRecord]:
    def identity(cell: CellKey) -> tuple[str, str, int]:
        return cell.implementation, cell.recipe_id, cell.seed

    keys = {identity(cell) for cell in expected}
    kept = {identity(record.cell) for record in retained}
    missing = keys - kept
    if (
        len(retained) != len(kept)
        or not kept <= keys
        or len(added) != len(missing)
        or {identity(record.cell) for record in added} != missing
    ):
        raise ValueError("additional cells must fill exactly the missing measurements")
    return retained + added


def _select_ax_rerun(
    recipes: RecipeCatalog,
    ax_recipes: RecipeCatalog,
    records: list[ResultRecord],
    ax_records: list[ResultRecord],
) -> list[ResultRecord]:
    if {(r.cell.recipe_id, r.cell.seed) for r in ax_records} != {
        (r.cell.recipe_id, r.cell.seed) for r in records if r.cell.implementation == AX
    }:
        raise ValueError("AX rerun must measure the same recipe and seed pairs")
    changed = {new.recipe_id for old, new in zip(recipes.recipes, ax_recipes.recipes, strict=True) if old != new}
    if changed - {recipe.recipe_id for recipe in recipes.recipes if recipe.augmentation_id == "Shear"}:
        raise ValueError("only Shear may differ in the AX rerun catalog")
    return [r for r in records if r.cell.implementation != AX and r.cell.recipe_id not in changed] + ax_records


def load_run_results(
    cells_directory: Path, run: RunRecord, recipes: RecipeCatalog, *, require_complete: bool = True
) -> list[ResultRecord]:
    identity = build_run_record(family_config=run.family_config, inputs=run.inputs, hardware=run.hardware)
    if identity != run:
        raise ValueError("run manifest does not match its immutable identity")
    cells = build_matrix(run_id=run.run_id, config=FamilyRunConfig.model_validate(run.family_config), recipes=recipes)
    records = [
        decode_result(path.read_bytes(), expected_cell_id=path.stem) for path in sorted(cells_directory.glob("*.json"))
    ]
    found = {record.cell_id for record in records}
    expected = {cell.cell_id for cell in cells}
    if not found <= expected or (require_complete and found != expected):
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
        "CommonRatioAxisMax": str(_ratio_axis_max(common_means)),
        "MemoryAxisMax": str(_memory_axis_max(common_memories)),
    }
    for implementation, label in data.implementations:
        command = "AllSIMD" if implementation == PILLOW_ALLSIMD[0] else label.replace(" ", "")
        metrics[f"CommonSpeed{command}"] = (
            f"{median(data.speed[(implementation, recipe)] for recipe in data.common):,.0f}"
        )
        metrics[f"CommonMemory{command}"] = f"{common_memories[implementation]:,.0f}"
    for family, paths in data.comparisons:
        compared = [AX, *paths]
        metrics[f"Plot{family}Shared"] = str(len(_shared_recipes(data, compared)))
        metrics[f"Plot{family}AxisMax"] = str(_ratio_axis_max(_mean_ratios(data, compared)))
    for family, rows in data.pairwise.items():
        metrics[f"Pair{family}Shared"] = str(len(rows))
        metrics[f"Pair{family}Wins"] = str(sum(row.ratio < 1 for row in rows))
        if family == "DALI":
            metrics["PairDALIFaster"] = str(sum(row.ratio > 1 for row in rows))
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


def _write_heatmap(path: Path, data: _PaperData) -> None:
    ratios = [
        data.speed[(implementation, recipe)] / data.speed[(AX, recipe)]
        for recipe in data.common
        for implementation, _ in data.implementations[1:]
    ]
    limit = max(1, math.ceil(max(abs(math.log2(value)) for value in ratios)))
    ax_column = len(data.implementations) - 0.35
    lines = [rf"\def\HeatLimit{{{limit}}}"]
    for y, recipe in enumerate(data.common):
        label = (
            recipe.removesuffix("+Normalize+ToTensor")
            .replace("RandomCrop224", "Crop")
            .replace("Resize224", "Resize")
            .replace("+", " + ")
        )
        key = _recipe_key(recipe, data.recipes)
        lines.append(rf"\node[anchor=east,font=\small] at (-0.65,{y}) {{\hyperlink{{{key}}}{{{key}}} {label}}};")
        lines.append(rf"\node[font=\small] at ({ax_column:.2f},{y}) {{{data.speed[(AX, recipe)]:,.0f}}};")
        for x, (implementation, _) in enumerate(data.implementations[1:]):
            ratio = data.speed[(implementation, recipe)] / data.speed[(AX, recipe)]
            color = "faster" if ratio >= 1 else "slower"
            shade = abs(math.log2(ratio)) / limit * 65
            lines.append(rf"\HeatCell{{{x}}}{{{y}}}{{{ratio:.2f}}}{{{color}!{shade:.2f}!white}}")
    _write_tex(path, lines)


def _shared_recipes(data: _PaperData, implementations: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        recipe.recipe_id
        for recipe in data.recipes
        if all((implementation, recipe.recipe_id) in data.speed for implementation in implementations)
    )


def pillow_allsimd_ratios(data: _PaperData) -> tuple[tuple[str, ...], list[float]]:
    recipes = _shared_recipes(data, ["pillow_cpu", PILLOW_ALLSIMD[0]])
    return recipes, [data.speed[(PILLOW_ALLSIMD[0], recipe)] / data.speed[("pillow_cpu", recipe)] for recipe in recipes]


def _mean_ratios(data: _PaperData, implementations: Sequence[str]) -> dict[str, float]:
    recipes = _shared_recipes(data, implementations)
    return {
        implementation: fmean(data.speed[(implementation, recipe)] / data.speed[(AX, recipe)] for recipe in recipes)
        for implementation in implementations
    }


def _write_mean_bars(path: Path, means: dict[str, float]) -> None:
    labels = dict((*IMPLEMENTATIONS, PILLOW_ALLSIMD))
    ordered = sorted(means, key=lambda implementation: means[implementation], reverse=True)
    lines: list[str] = []
    for index, implementation in enumerate(ordered):
        label = labels[implementation]
        value = means[implementation]
        y = len(ordered) - index - 1
        color = "faster!65!white" if value > 1 else "slower!55!white"
        display_label = f"{label} CPU" if implementation == AX else label
        if implementation == AX:
            color = "gray!45"
        lines.extend(
            (
                rf"\fill[{color}] (0,{y - 0.27:.2f}) rectangle ({value:.10f},{y + 0.27:.2f});",
                rf"\node[anchor=east,font=\small] at (-0.03,{y}) {{{display_label}}};",
                rf"\node[anchor=west,font=\small] at ({value + 0.02:.10f},{y}) {{{value:.2f}$\times$}};",
            )
        )
    _write_tex(path, lines)


def _common_memories(data: _PaperData) -> dict[str, float]:
    return {
        implementation: median(data.memory[(implementation, recipe)] for recipe in data.common)
        for implementation, _ in data.implementations
    }


def _ratio_axis_max(means: Mapping[str, float]) -> float:
    return max(1.5, math.ceil(max(means.values()) * 4) / 4)


def _memory_axis_max(memories: Mapping[str, float]) -> int:
    return math.ceil(max(memories.values()) / 500) * 500


def _write_memory_bars(path: Path, data: _PaperData) -> None:
    values = _common_memories(data)
    ordered = sorted(data.implementations, key=lambda item: values[item[0]])
    lines: list[str] = []
    for index, (implementation, label) in enumerate(ordered):
        value = values[implementation]
        y = len(ordered) - index - 1
        color = "gray!45" if implementation == AX else "memorycolor!55!white"
        display_label = f"{label} CPU" if implementation == AX else label
        lines.extend(
            (
                rf"\fill[{color}] (0,{y - 0.27:.2f}) rectangle ({value:.10f},{y + 0.27:.2f});",
                rf"\node[anchor=east,font=\small] at (-60,{y}) {{{display_label}}};",
                rf"\node[anchor=west,font=\small] at ({value + 40:.10f},{y}) {{{value:,.0f}}};",
            ),
        )
    _write_tex(path, lines)


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


def _write_sources(
    path: Path, run: RunRecord, ax_run: RunRecord | None, correction_runs: tuple[RunRecord, ...]
) -> None:
    lines: list[str] = []
    for label, source in (("Full", run), ("AX", ax_run or run)):
        lines.extend(
            (
                rf"\newcommand{{\{label}RunID}}{{{source.run_id}}}",
                rf"\newcommand{{\{label}SourceRevision}}{{{source.inputs.git_commit}}}",
            )
        )
    lines.append(r"\newcommand{\CorrectionRunEntries}{%")
    for index, source in enumerate(correction_runs, 1):
        lines.append(
            rf"\item[Correction {index}] \texttt{{\seqsplit{{{source.run_id}}}}}\\"
            rf" Source \texttt{{{source.inputs.git_commit}}}.%"
        )
    _write_tex(path, [*lines, "}"])


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
