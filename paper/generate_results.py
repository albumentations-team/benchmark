# ruff: noqa: INP001
"""Generate the RGB paper and its audit table from one validated production run."""

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

from augbench.frozen_rgb_run import build_frozen_rgb_run
from augbench.recipes.load import load_recipe_catalog
from augbench.result_store import decode_result

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from augbench.recipes.models import RecipeSpec
    from augbench.run_records import ResultRecord

RUN_ID = "3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c"
GIT_COMMIT = "5fc35f6fdd177c286cbc4f5e39d1520576d6464a"
CODE_ARCHIVE_SHA256 = "61238619ace0dc471bc5df7a4f165e15052ffb67046ceae07c26fd7f370eac6d"
AX = "albumentationsx_cpu"
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
    common: tuple[str, ...]
    pairwise: dict[str, list[_PairwiseRow]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    records = _load_complete_run(args.cells, root)
    data = _prepare(records, load_recipe_catalog(root / "catalog/recipes/rgb.yaml").recipes)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_metrics(args.output / "metrics.tex", data)
    _write_heatmap(args.output / "common-heatmap.tex", data)
    _write_mean_bars(
        args.output / "common-mean-bars.tex",
        _mean_ratios(data, [implementation for implementation, _ in IMPLEMENTATIONS]),
    )
    _write_memory_bars(args.output / "common-memory-bars.tex", data)
    for family, paths in PAIRWISE:
        _write_mean_bars(args.output / f"pairwise-{family.lower()}.tex", _mean_ratios(data, [AX, *paths]))
    _write_csv(args.output / "recipe-results.csv", data)
    _write_recipe_catalog(args.output / "recipes.tex", data.recipes)
    _write_recipe_results(args.output / "recipe-results.tex", data)
    _write_versions(args.output / "versions.tex", data)
    _write_coverage(args.output, root / "catalog/operations.yaml")


def _load_complete_run(cells_directory: Path, repository_root: Path) -> list[ResultRecord]:
    records = [
        decode_result(path.read_bytes(), expected_cell_id=path.stem) for path in sorted(cells_directory.glob("*.json"))
    ]
    frozen = build_frozen_rgb_run(
        repository_root=repository_root,
        git_commit=GIT_COMMIT,
        code_archive_sha256=CODE_ARCHIVE_SHA256,
    )
    expected = {cell.cell_id for cell in frozen.cells}
    actual = {record.cell_id for record in records}
    if frozen.run.run_id != RUN_ID or actual != expected or any(record.run_id != RUN_ID for record in records):
        raise RuntimeError("paper results do not match the frozen production run")
    config = frozen.run.family_config
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
    common = set.intersection(*(support[implementation] for implementation, _ in IMPLEMENTATIONS))
    ordered = tuple(recipe.recipe_id for recipe in recipes if recipe.recipe_id in common)
    return _PaperData(dict(groups), recipes, speed, memory, ordered, _pairwise(speed, memory, support))


def _pairwise(
    speed: dict[tuple[str, str], float],
    memory: dict[tuple[str, str], float],
    support: dict[str, set[str]],
) -> dict[str, list[_PairwiseRow]]:
    rows: dict[str, list[_PairwiseRow]] = {}
    for family, implementations in PAIRWISE:
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
    common_means = _mean_ratios(data, [implementation for implementation, _ in IMPLEMENTATIONS])
    metrics = {
        "ProductionCellCount": str(sum(map(len, data.groups.values()))),
        "CommonRecipeCount": str(len(data.common)),
        "CommonMeanDaliRatio": f"{common_means['dali_gpu']:.2f}",
    }
    for implementation, label in IMPLEMENTATIONS:
        command = label.replace(" ", "")
        metrics[f"CommonSpeed{command}"] = (
            f"{median(data.speed[(implementation, recipe)] for recipe in data.common):,.0f}"
        )
        metrics[f"CommonMemory{command}"] = (
            f"{median(data.memory[(implementation, recipe)] for recipe in data.common):,.0f}"
        )
    for family, paths in PAIRWISE:
        metrics[f"Plot{family}Shared"] = str(len(_shared_recipes(data, [AX, *paths])))
    for family, rows in data.pairwise.items():
        metrics[f"Pair{family}Shared"] = str(len(rows))
        metrics[f"Pair{family}Wins"] = str(sum(row.ratio < 1 for row in rows))
        metrics[f"Pair{family}Ratio"] = f"{median(row.ratio for row in rows):.2f}"
        metrics[f"Pair{family}MemoryDelta"] = f"{median(row.memory_delta for row in rows):+,.0f}"
    metrics["MemoryPollMs"] = str(
        _single({record.gpu_memory.poll_interval_ms for group in data.groups.values() for record in group})
    )
    _write_tex(path, (rf"\newcommand{{\{command}}}{{{value}}}" for command, value in metrics.items()))


def _write_heatmap(path: Path, data: _PaperData) -> None:
    ratios = [
        data.speed[(implementation, recipe)] / data.speed[(AX, recipe)]
        for recipe in data.common
        for implementation, _ in IMPLEMENTATIONS[1:]
    ]
    limit = max(1, math.ceil(max(abs(math.log2(value)) for value in ratios)))
    lines = [rf"\def\HeatLimit{{{limit}}}"]
    for y, recipe in enumerate(data.common):
        label = recipe.removesuffix("+Normalize+ToTensor").replace("RandomCrop224", "Crop").replace("+", " + ")
        key = _recipe_key(recipe, data.recipes)
        lines.append(rf"\node[anchor=east,font=\small] at (-0.65,{y}) {{\hyperlink{{{key}}}{{{key}}} {label}}};")
        lines.append(rf"\node[font=\small] at (6.65,{y}) {{{data.speed[(AX, recipe)]:,.0f}}};")
        for x, (implementation, _) in enumerate(IMPLEMENTATIONS[1:]):
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


def _mean_ratios(data: _PaperData, implementations: Sequence[str]) -> dict[str, float]:
    recipes = _shared_recipes(data, implementations)
    return {
        implementation: fmean(data.speed[(implementation, recipe)] / data.speed[(AX, recipe)] for recipe in recipes)
        for implementation in implementations
    }


def _write_mean_bars(path: Path, means: dict[str, float]) -> None:
    labels = dict(IMPLEMENTATIONS)
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


def _write_memory_bars(path: Path, data: _PaperData) -> None:
    values = {
        implementation: median(data.memory[(implementation, recipe)] for recipe in data.common)
        for implementation, _ in IMPLEMENTATIONS
    }
    ordered = sorted(IMPLEMENTATIONS, key=lambda item: values[item[0]])
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
    for implementation, label in IMPLEMENTATIONS:
        versions = {
            r.runtime["library_version"]
            for (impl, _), group in data.groups.items()
            if impl == implementation
            for r in group
        }
        lines.append(rf"{label} & \texttt{{{_tex(_single(versions))}}} \\")
    _write_tex(path, lines)


def _csv_row(group: list[ResultRecord], recipe: RecipeSpec, data: _PaperData) -> dict[str, object]:
    implementation = group[0].cell.implementation
    key = (implementation, recipe.recipe_id)
    selected = any(
        row.implementation == implementation and row.recipe == recipe.recipe_id
        for rows in data.pairwise.values()
        for row in rows
    )
    row: dict[str, object] = {
        "run_id": RUN_ID,
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
                f"cell_id_{seed}": record.cell_id,
                f"completed_images_{seed}": record.throughput.completed_items,
                f"duration_seconds_{seed}": record.throughput.duration_seconds,
                f"throughput_images_s_{seed}": record.throughput.value,
                f"gpu_memory_peak_mib_{seed}": record.gpu_memory.peak_mib,
                f"gpu_memory_poll_ms_{seed}": record.gpu_memory.poll_interval_ms,
            }
        )
    return row


def _write_csv(path: Path, data: _PaperData) -> None:
    rows = [
        _csv_row(data.groups[(implementation, recipe.recipe_id)], recipe, data)
        for implementation, _ in IMPLEMENTATIONS
        for recipe in data.recipes
        if (implementation, recipe.recipe_id) in data.groups
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
    for implementation, label in IMPLEMENTATIONS:
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
