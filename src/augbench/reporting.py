"""Validated aggregation for the one production result schema."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from statistics import median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from augbench.result_store import ImmutableResultStore
    from augbench.run_records import CellKey, ResultRecord


@dataclass(frozen=True)
class ReportRow:
    implementation: str
    recipe_id: str
    seeds: tuple[int, ...]
    median_images_per_second: float
    median_peak_gpu_mib: float


@dataclass(frozen=True)
class MatrixReport:
    complete: bool
    missing_cell_ids: tuple[str, ...]
    rows: tuple[ReportRow, ...]


def build_report(*, results: ImmutableResultStore, expected_cells: tuple[CellKey, ...]) -> MatrixReport:
    records: list[ResultRecord] = []
    missing: list[str] = []
    for cell in expected_cells:
        try:
            records.append(results.get(cell.cell_id))
        except FileNotFoundError:
            missing.append(cell.cell_id)
    if missing:
        return MatrixReport(complete=False, missing_cell_ids=tuple(missing), rows=())
    return MatrixReport(complete=True, missing_cell_ids=(), rows=_summaries(records))


def write_report(report: MatrixReport, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "complete": report.complete,
        "missing_cell_ids": list(report.missing_cell_ids),
        "rows": [asdict(row) for row in report.rows],
    }
    (output_directory / "report.json").write_text(f"{json.dumps(payload, sort_keys=True)}\n", encoding="utf-8")
    with (output_directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ReportRow.__dataclass_fields__))
        writer.writeheader()
        for row in report.rows:
            encoded = asdict(row)
            encoded["seeds"] = ",".join(str(seed) for seed in row.seeds)
            writer.writerow(encoded)


def _summaries(records: list[ResultRecord]) -> tuple[ReportRow, ...]:
    grouped: dict[tuple[str, str], list[ResultRecord]] = {}
    for record in records:
        grouped.setdefault((record.cell.implementation, record.cell.recipe_id), []).append(record)
    rows: list[ReportRow] = []
    for (implementation, recipe_id), group in sorted(grouped.items()):
        throughputs = [record.throughput.value for record in group]
        peak_memory = [record.gpu_memory.peak_mib for record in group]
        rows.append(
            ReportRow(
                implementation=implementation,
                recipe_id=recipe_id,
                seeds=tuple(sorted(record.cell.seed for record in group)),
                median_images_per_second=median(throughputs),
                median_peak_gpu_mib=median(peak_memory),
            ),
        )
    return tuple(rows)
