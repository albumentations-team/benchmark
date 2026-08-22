# ruff: noqa: INP001
"""Generate the benchmark paper's tables and plot data from validated cells."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import yaml

from augbench.frozen_rgb_run import build_frozen_rgb_run
from augbench.result_store import decode_result

if TYPE_CHECKING:
    from collections.abc import Iterable

    from augbench.run_records import ResultRecord

RUN_ID = "3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c"
GIT_COMMIT = "5fc35f6fdd177c286cbc4f5e39d1520576d6464a"
CODE_ARCHIVE_SHA256 = "61238619ace0dc471bc5df7a4f165e15052ffb67046ceae07c26fd7f370eac6d"

AX = "albumentationsx_cpu"
IMPLEMENTATIONS = (
    ("kornia_cpu", "Kornia CPU"),
    ("kornia_gpu", "Kornia GPU"),
    ("pillow_cpu", "Pillow CPU"),
    ("torchvision_cpu", "TorchVision CPU"),
    ("torchvision_gpu", "TorchVision GPU"),
    (AX, "AlbumentationsX"),
    ("dali_gpu", "DALI GPU"),
)
PAIRWISE = (
    ("Kornia", ("kornia_cpu", "kornia_gpu")),
    ("TorchVision", ("torchvision_cpu", "torchvision_gpu")),
    ("Pillow", ("pillow_cpu",)),
    ("DALI", ("dali_gpu",)),
)


@dataclass(frozen=True)
class _MetricInputs:
    cell_count: int
    speed: dict[tuple[str, str], float]
    memory: dict[tuple[str, str], float]
    common: set[str]
    pairwise: dict[str, list[tuple[str, float, float]]]
    coverage: dict[str, tuple[int, int]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()

    repository_root = Path(__file__).parents[1]
    records = _load_complete_run(args.cells, repository_root)
    speed, memory = _aggregate(records)
    support = _support_sets(speed)
    common = set.intersection(*(support[implementation] for implementation, _ in IMPLEMENTATIONS))
    pairwise = _pairwise(speed, memory, support)
    coverage = _coverage(repository_root / "catalog/operations.yaml")

    args.output.mkdir(parents=True, exist_ok=True)
    _write_metrics(
        args.output / "metrics.tex",
        _MetricInputs(
            cell_count=len(records),
            speed=speed,
            memory=memory,
            common=common,
            pairwise=pairwise,
            coverage=coverage,
        ),
    )
    _write_coverage_data(args.output / "coverage.dat", coverage)
    _write_common_data(args.output, speed, memory, common)
    _write_pairwise_data(args.output, pairwise)
    _write_common_recipes(args.output / "common-recipes.tex", common)


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
    return records


def _aggregate(
    records: Iterable[ResultRecord],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    throughput: dict[tuple[str, str], list[float]] = defaultdict(list)
    gpu_memory: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        key = (record.cell.implementation, record.cell.recipe_id)
        throughput[key].append(record.throughput.value)
        gpu_memory[key].append(record.gpu_memory.peak_mib)
    return (
        {key: median(values) for key, values in throughput.items()},
        {key: median(values) for key, values in gpu_memory.items()},
    )


def _support_sets(values: dict[tuple[str, str], float]) -> dict[str, set[str]]:
    support: dict[str, set[str]] = defaultdict(set)
    for implementation, recipe in values:
        support[implementation].add(recipe)
    return support


def _pairwise(
    speed: dict[tuple[str, str], float],
    memory: dict[tuple[str, str], float],
    support: dict[str, set[str]],
) -> dict[str, list[tuple[str, float, float]]]:
    rows: dict[str, list[tuple[str, float, float]]] = {}
    for family, implementations in PAIRWISE:
        available = set.union(*(support[implementation] for implementation in implementations))
        family_rows: list[tuple[str, float, float]] = []
        for recipe in sorted(support[AX] & available):
            candidates = [implementation for implementation in implementations if (implementation, recipe) in speed]
            fastest = max(candidates, key=lambda implementation: speed[(implementation, recipe)])
            family_rows.append(
                (
                    recipe,
                    speed[(AX, recipe)] / speed[(fastest, recipe)],
                    memory[(AX, recipe)] - memory[(fastest, recipe)],
                ),
            )
        rows[family] = family_rows
    return rows


def _coverage(catalog_path: Path) -> dict[str, tuple[int, int]]:
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    operations = catalog["operations"]
    counts = {"AlbumentationsX": (len(operations), 0)}
    for library, label in (("kornia", "Kornia"), ("torchvision", "TorchVision"), ("pillow", "Pillow")):
        direct = sum(operation["support"][library]["status"] == "direct" for operation in operations)
        partial = sum(operation["support"][library]["status"] == "partial" for operation in operations)
        counts[label] = (direct, partial)
    return counts


def _write_metrics(path: Path, inputs: _MetricInputs) -> None:
    lines = [
        "% Generated by paper/generate_results.py. Do not edit by hand.",
        f"\\newcommand{{\\ProductionRunId}}{{\\texttt{{{RUN_ID[:12]}\\ldots}}}}",
        f"\\newcommand{{\\ProductionCellCount}}{{{inputs.cell_count}}}",
        f"\\newcommand{{\\CommonRecipeCount}}{{{len(inputs.common)}}}",
    ]
    for label, (direct, partial) in inputs.coverage.items():
        command = label.replace(" ", "")
        lines.append(f"\\newcommand{{\\Coverage{command}}}{{{direct + partial}}}")
        lines.append(f"\\newcommand{{\\Coverage{command}Partial}}{{{partial}}}")
    for implementation, label in IMPLEMENTATIONS:
        command = label.replace(" ", "")
        common_speed = median(inputs.speed[(implementation, recipe)] for recipe in inputs.common)
        common_memory = median(inputs.memory[(implementation, recipe)] for recipe in inputs.common)
        common_memory_max = max(inputs.memory[(implementation, recipe)] for recipe in inputs.common)
        lines.append(f"\\newcommand{{\\CommonSpeed{command}}}{{{common_speed:,.0f}}}")
        lines.append(f"\\newcommand{{\\CommonMemory{command}}}{{{common_memory:,.0f}}}")
        lines.append(f"\\newcommand{{\\CommonMemoryMax{command}}}{{{common_memory_max:,.0f}}}")
    for family, rows in inputs.pairwise.items():
        ratios = [ratio for _, ratio, _ in rows]
        deltas = [delta for _, _, delta in rows]
        lines.append(f"\\newcommand{{\\Pair{family}Shared}}{{{len(rows)}}}")
        lines.append(f"\\newcommand{{\\Pair{family}Wins}}{{{sum(ratio > 1 for ratio in ratios)}}}")
        lines.append(f"\\newcommand{{\\Pair{family}Speedup}}{{{median(ratios):.2f}}}")
        lines.append(f"\\newcommand{{\\Pair{family}MemoryMagnitude}}{{{abs(median(deltas)):,.0f}}}")
        if family == "DALI":
            lines.append(f"\\newcommand{{\\PairDALIVsAXSpeedup}}{{{median(1 / ratio for ratio in ratios):.2f}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_coverage_data(path: Path, coverage: dict[str, tuple[int, int]]) -> None:
    lines = ["direct partial y"]
    for y, label in enumerate(("Pillow", "TorchVision", "Kornia", "AlbumentationsX")):
        direct, partial = coverage[label]
        lines.append(f"{direct} {partial} {y}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_common_data(
    output: Path,
    speed: dict[tuple[str, str], float],
    memory: dict[tuple[str, str], float],
    common: set[str],
) -> None:
    throughput_rows = ["x y"]
    throughput_medians = ["x y"]
    memory_rows = ["x y"]
    memory_medians = ["x y"]
    for y, (implementation, _) in enumerate(IMPLEMENTATIONS):
        values = [speed[(implementation, recipe)] for recipe in sorted(common)]
        memory_values = [memory[(implementation, recipe)] for recipe in sorted(common)]
        throughput_rows.extend(f"{value:.8f} {y}" for value in values)
        throughput_medians.append(f"{median(values):.8f} {y}")
        memory_rows.extend(f"{value:.8f} {y}" for value in memory_values)
        memory_medians.append(f"{median(memory_values):.8f} {y}")
    (output / "common-throughput.dat").write_text("\n".join(throughput_rows) + "\n", encoding="utf-8")
    (output / "common-throughput-medians.dat").write_text(
        "\n".join(throughput_medians) + "\n",
        encoding="utf-8",
    )
    (output / "common-memory.dat").write_text("\n".join(memory_rows) + "\n", encoding="utf-8")
    (output / "common-memory-medians.dat").write_text("\n".join(memory_medians) + "\n", encoding="utf-8")


def _write_pairwise_data(output: Path, pairwise: dict[str, list[tuple[str, float, float]]]) -> None:
    rows = ["x y"]
    medians = ["x y"]
    for y, (family, _) in enumerate(PAIRWISE):
        family_rows = pairwise[family]
        for index, (_, speedup, _) in enumerate(sorted(family_rows, key=lambda row: row[1])):
            jitter = ((index % 7) - 3) * 0.025
            rows.append(f"{speedup:.8f} {y + jitter:.3f}")
        medians.append(f"{median(speedup for _, speedup, _ in family_rows):.8f} {y}")
    (output / "pairwise-speedup.dat").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (output / "pairwise-speedup-medians.dat").write_text("\n".join(medians) + "\n", encoding="utf-8")


def _write_common_recipes(path: Path, common: set[str]) -> None:
    lines = ["% Generated by paper/generate_results.py. Do not edit by hand."]
    for recipe in sorted(common):
        escaped_recipe = recipe.replace("_", "\\_")
        lines.append(f"\\texttt{{{escaped_recipe}}} " + r"\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
