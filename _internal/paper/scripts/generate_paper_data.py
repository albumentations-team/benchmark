# ruff: noqa: INP001, E501, PERF401
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "_internal" / "paper" / "generated"
DEFAULT_DRAFT = ROOT / "_internal" / "paper" / "draft.md"
PAPER_DIR = ROOT / "_internal" / "paper" / "neurips_2026_ed"


RUN_GROUPS = {
    "rgb_micro_cpu": [
        ROOT / "gcp_runs/prod-c4-standard-16-rgb-micro-cpu-r3/results/image-rgb/micro",
    ],
    "rgb_micro_gpu": [
        ROOT / "gcp_runs/prod-g2-standard-16-rgb-micro-gpu-r3/results/image-rgb/micro",
    ],
    "rgb_dataloader_cpu": [
        ROOT / "gcp_runs/prod-c4-standard-16-rgb-dataloader-cpu/results/image-rgb/pipeline",
        ROOT / "gcp_runs/prod-c4-standard-16-rgb-dataloader-cpu-topup-r2/results/image-rgb/pipeline",
    ],
    "rgb_dataloader_gpu": [
        ROOT / "results/published/paper-rgb-dataloader-gpu-memory-g2-standard-16-2026-05-07",
        ROOT / "gcp_runs/prod-g2-standard-16-rgb-dataloader-dali/results/image-rgb/pipeline",
    ],
}

REPLACED_RESULT_FILES: set[Path] = set()


LIBRARY_ORDER = ["albumentationsx", "torchvision", "kornia", "pillow", "dali"]
LIBRARY_DISPLAY = {
    "albumentationsx": "AlbumentationsX",
    "torchvision": "TorchVision",
    "kornia": "Kornia",
    "pillow": "Pillow",
    "dali": "DALI",
}
REGIME_LABELS = {
    "rgb_micro_cpu": "CPU micro",
    "rgb_micro_gpu": "GPU micro",
    "rgb_dataloader_cpu": "CPU DataLoader",
    "rgb_dataloader_gpu": "GPU DataLoader",
}
GENERATED_START = "<!-- GENERATED_RESULTS_START -->"
GENERATED_END = "<!-- GENERATED_RESULTS_END -->"
OPEN_DATALOADER_IMPLEMENTATION_ORDER = [
    ("rgb_dataloader_cpu", "albumentationsx"),
    ("rgb_dataloader_gpu", "dali"),
    ("rgb_dataloader_cpu", "torchvision"),
    ("rgb_dataloader_cpu", "pillow"),
    ("rgb_dataloader_gpu", "torchvision"),
    ("rgb_dataloader_cpu", "kornia"),
    ("rgb_dataloader_gpu", "kornia"),
]


@dataclass(frozen=True)
class SourceResult:
    regime: str
    library: str
    transform: str
    result: dict[str, Any]
    metadata: dict[str, Any]
    source: Path


@dataclass
class AggregateResult:
    regime: str
    library: str
    transform: str
    status: str
    supported: bool
    early_stopped: bool
    throughputs: list[float]
    times: list[float]
    median_throughput: float
    mean_throughput: float
    std_throughput: float
    cv_throughput: float
    ci95: float
    num_successful_runs: int
    reason: str
    gpu_peak_allocated_mb: float | None
    gpu_peak_reserved_mb: float | None
    slow_marker: str
    source_files: list[str]


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        msg = f"Unexpected result JSON shape: {path}"
        raise TypeError(msg)
    return payload


def _library_from_payload(path: Path, payload: dict[str, Any]) -> str:
    library = payload.get("metadata", {}).get("library")
    if isinstance(library, str) and library:
        return library
    stem = path.name
    for candidate in LIBRARY_ORDER:
        if stem.startswith(candidate):
            return candidate
    msg = f"Could not infer library for {path}"
    raise ValueError(msg)


def _iter_sources() -> list[SourceResult]:
    rows: list[SourceResult] = []
    for regime, dirs in RUN_GROUPS.items():
        for directory in dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*results.json")):
                if path in REPLACED_RESULT_FILES:
                    continue
                if path.name.endswith(".pyperf.json"):
                    continue
                payload = _load_payload(path)
                library = _library_from_payload(path, payload)
                metadata = payload.get("metadata", {})
                for transform, result in payload["results"].items():
                    rows.append(
                        SourceResult(
                            regime=regime,
                            library=library,
                            transform=transform,
                            result=result,
                            metadata=metadata,
                            source=path.relative_to(ROOT),
                        ),
                    )
    return rows


def _summarize(values: list[float]) -> tuple[float, float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    cv = std / mean if mean else 0.0
    ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return median, mean, std, cv, ci95


def _gpu_memory_mb(results: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for result in results:
        gpu_memory = result.get("gpu_memory")
        if isinstance(gpu_memory, dict) and isinstance(gpu_memory.get(key), int | float):
            values.append(float(gpu_memory[key]) / (1024 * 1024))
    return max(values) if values else None


def _aggregate_sources(sources: list[SourceResult]) -> list[AggregateResult]:
    grouped: dict[tuple[str, str, str], list[SourceResult]] = defaultdict(list)
    for source in sources:
        grouped[(source.regime, source.library, source.transform)].append(source)

    aggregates: list[AggregateResult] = []
    for (regime, library, transform), group in sorted(grouped.items()):
        result_dicts = [item.result for item in group]
        unsupported = [r for r in result_dicts if r.get("supported") is False or r.get("status") == "unsupported"]
        supported = not unsupported
        early_stopped = any(bool(r.get("early_stopped")) for r in result_dicts)
        throughputs = [
            float(value) for r in result_dicts for value in r.get("throughputs", []) if isinstance(value, int | float)
        ]
        times = [float(value) for r in result_dicts for value in r.get("times", []) if isinstance(value, int | float)]
        median, mean, std, cv, ci95 = _summarize(throughputs)
        if not throughputs:
            medians = [float(r.get("median_throughput", 0.0)) for r in result_dicts if r.get("median_throughput")]
            median = statistics.median(medians) if medians else 0.0
            mean = statistics.fmean(medians) if medians else 0.0
        reason = "; ".join(
            str(r.get("reason") or r.get("early_stop_reason") or r.get("unstable_reason") or "")
            for r in result_dicts
            if r.get("reason") or r.get("early_stop_reason") or r.get("unstable_reason")
        )
        status = (
            "unsupported"
            if not supported
            else "partial_early_stop"
            if early_stopped and throughputs
            else "early_stopped"
            if early_stopped
            else "ok"
        )
        aggregates.append(
            AggregateResult(
                regime=regime,
                library=library,
                transform=transform,
                status=status,
                supported=supported,
                early_stopped=early_stopped,
                throughputs=throughputs,
                times=times,
                median_throughput=median,
                mean_throughput=mean,
                std_throughput=std,
                cv_throughput=cv,
                ci95=ci95,
                num_successful_runs=len(throughputs),
                reason=reason,
                gpu_peak_allocated_mb=_gpu_memory_mb(result_dicts, "peak_allocated_bytes"),
                gpu_peak_reserved_mb=_gpu_memory_mb(result_dicts, "peak_reserved_bytes"),
                slow_marker=str(result_dicts[-1].get("slow_marker") or ""),
                source_files=sorted(str(item.source) for item in group),
            ),
        )
    return aggregates


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _status_cell(row: AggregateResult) -> str:
    if not row.supported:
        return "unsupported"
    if row.status == "partial_early_stop":
        return f"{_fmt(row.median_throughput)} (partial)"
    if row.early_stopped:
        return f"≤20 img/s ({_fmt(row.median_throughput)})"
    if row.num_successful_runs == 0:
        return "no full run"
    return _fmt(row.median_throughput)


def _write_csv(path: Path, rows: list[AggregateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regime",
        "regime_label",
        "library",
        "transform",
        "status",
        "supported",
        "early_stopped",
        "num_successful_runs",
        "median_throughput",
        "mean_throughput",
        "std_throughput",
        "cv_throughput",
        "ci95",
        "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
        "reason",
        "source_files",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "regime": row.regime,
                    "regime_label": REGIME_LABELS[row.regime],
                    "library": row.library,
                    "transform": row.transform,
                    "status": row.status,
                    "supported": row.supported,
                    "early_stopped": row.early_stopped,
                    "num_successful_runs": row.num_successful_runs,
                    "median_throughput": row.median_throughput,
                    "mean_throughput": row.mean_throughput,
                    "std_throughput": row.std_throughput,
                    "cv_throughput": row.cv_throughput,
                    "ci95": row.ci95,
                    "gpu_peak_allocated_mb": row.gpu_peak_allocated_mb,
                    "gpu_peak_reserved_mb": row.gpu_peak_reserved_mb,
                    "reason": row.reason,
                    "source_files": ";".join(row.source_files),
                },
            )


def _write_pivot_csv(path: Path, rows: list[AggregateResult]) -> None:
    transforms = sorted({row.transform for row in rows})
    libraries = [lib for lib in LIBRARY_ORDER if any(row.library == lib for row in rows)]
    by_key = {(row.transform, row.library): row for row in rows}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["transform", *libraries])
        for transform in transforms:
            values = [
                _status_cell(by_key[(transform, library)]) if (transform, library) in by_key else "-"
                for library in libraries
            ]
            writer.writerow([transform, *values])


def _regime_rows(rows: list[AggregateResult], regime: str) -> list[AggregateResult]:
    return [row for row in rows if row.regime == regime]


def _is_measured(row: AggregateResult) -> bool:
    return row.supported and not row.early_stopped and row.num_successful_runs > 0


def _implementation_label(regime: str, library: str) -> str:
    device = "GPU" if regime == "rgb_dataloader_gpu" else "CPU"
    return f"{LIBRARY_DISPLAY.get(library, library)} {device}"


def _recipe_display_name(recipe: str) -> str:
    label = recipe
    suffix = "+Normalize+ToTensor"
    label = label.removesuffix(suffix)
    prefix = "RandomCrop224+"
    label = label.removeprefix(prefix)
    return label or "RandomCrop224"


def _join_display_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _open_dataloader_stats(rows: list[AggregateResult]) -> dict[str, Any]:
    universe = sorted(
        {row.transform for row in rows if row.regime == "rgb_dataloader_cpu" and row.library == "albumentationsx"},
    )
    universe_set = set(universe)
    by_key = {
        (row.regime, row.library, row.transform): row
        for row in rows
        if row.regime in {"rgb_dataloader_cpu", "rgb_dataloader_gpu"}
    }
    wins: dict[tuple[str, str], int] = defaultdict(int)
    winner_rows: list[dict[str, Any]] = []
    for transform in universe:
        candidates = [
            row
            for (regime, library, candidate_transform), row in by_key.items()
            if candidate_transform == transform and _is_measured(row)
        ]
        if not candidates:
            continue
        winner = max(candidates, key=lambda row: row.median_throughput)
        wins[(winner.regime, winner.library)] += 1
        winner_rows.append(
            {
                "transform": transform,
                "display_transform": _recipe_display_name(transform),
                "regime": winner.regime,
                "regime_label": REGIME_LABELS[winner.regime],
                "library": winner.library,
                "implementation": _implementation_label(winner.regime, winner.library),
                "median_throughput": winner.median_throughput,
            },
        )

    leaderboard: list[dict[str, Any]] = []
    for regime, library in OPEN_DATALOADER_IMPLEMENTATION_ORDER:
        measured_rows = [
            row
            for row in rows
            if row.regime == regime and row.library == library and row.transform in universe_set and _is_measured(row)
        ]
        if not measured_rows and not any(row.regime == regime and row.library == library for row in rows):
            continue
        median = statistics.median([row.median_throughput for row in measured_rows]) if measured_rows else None
        leaderboard.append(
            {
                "implementation": _implementation_label(regime, library),
                "regime": regime,
                "regime_label": REGIME_LABELS[regime],
                "library": library,
                "full": len(measured_rows),
                "universe_rows": len(universe),
                "median_throughput": median,
                "wins": wins.get((regime, library), 0),
            },
        )
    leaderboard.sort(
        key=lambda item: (
            -(item["median_throughput"] if item["median_throughput"] is not None else -math.inf),
            str(item["implementation"]),
        ),
    )
    return {
        "universe_rows": len(universe),
        "leaderboard": leaderboard,
        "winner_rows": sorted(winner_rows, key=lambda item: item["display_transform"]),
    }


def _open_dataloader_leaderboard_table(rows: list[AggregateResult]) -> str:
    stats = _open_dataloader_stats(rows)
    lines = [
        "This is the open production DataLoader category: CPU and GPU DataLoader rows compete together over the "
        "57-recipe universe. Microbenchmarks remain separate.",
        "",
        "| Implementation | Regime | Full measured / 57 | Median measured-row throughput (img/s) | Open-category wins |",
        "|---|---|---:|---:|---:|",
    ]
    for item in stats["leaderboard"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["implementation"]),
                    str(item["regime_label"]),
                    f"{item['full']}/{item['universe_rows']}",
                    _fmt(item["median_throughput"]),
                    str(item["wins"]),
                ],
            )
            + " |",
        )

    exceptions: dict[str, list[str]] = defaultdict(list)
    for item in stats["winner_rows"]:
        if item["implementation"] != "AlbumentationsX CPU":
            exceptions[str(item["implementation"])].append(str(item["display_transform"]))
    exception_text = "; ".join(
        f"{implementation} wins {_join_display_list(transforms)}"
        for implementation, transforms in sorted(exceptions.items())
    )
    if exception_text:
        lines.extend(["", f"Open-category exception rows: {exception_text}."])
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _write_open_dataloader_files(output_dir: Path, rows: list[AggregateResult]) -> None:
    stats = _open_dataloader_stats(rows)
    leaderboard_fieldnames = [
        "implementation",
        "regime",
        "regime_label",
        "library",
        "full",
        "universe_rows",
        "median_throughput",
        "wins",
    ]
    with (output_dir / "open_dataloader_leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=leaderboard_fieldnames)
        writer.writeheader()
        writer.writerows(stats["leaderboard"])
    with (output_dir / "open_dataloader_winners.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "transform",
                "display_transform",
                "regime",
                "regime_label",
                "library",
                "implementation",
                "median_throughput",
            ],
        )
        writer.writeheader()
        writer.writerows(stats["winner_rows"])

    lines = [
        r"\begin{table}[t]",
        r"  \caption{Absolute/open production DataLoader leaderboard. CPU and GPU DataLoader implementations compete together over the same 57-recipe universe; microbenchmarks are intentionally excluded. Median throughput is computed over full measured rows only, while missing, unsupported, and early-stopped rows reduce coverage.}",
        r"  \label{tab:open-dataloader}",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Implementation & Regime & Measured & Median img/s & Wins \\",
        r"    \midrule",
    ]
    for item in stats["leaderboard"]:
        values = [
            _latex_escape(str(item["implementation"])),
            _latex_escape(str(item["regime_label"])),
            f"{item['full']}/{item['universe_rows']}",
            _fmt(item["median_throughput"]),
            str(item["wins"]),
        ]
        lines.append("    " + " & ".join(values) + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    table = "\n".join(lines)
    (output_dir / "open_dataloader_leaderboard_table.tex").write_text(table, encoding="utf-8")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "open_dataloader_leaderboard_table.tex").write_text(table, encoding="utf-8")


def _library_summary(rows: list[AggregateResult]) -> str:
    lines = [
        "| Regime | Library | Rows | Full runs | Early-stopped | Unsupported | Median of measured rows (img/s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for regime, regime_label in REGIME_LABELS.items():
        for library in LIBRARY_ORDER:
            subset = [r for r in rows if r.regime == regime and r.library == library]
            if not subset:
                continue
            measured = [
                r.median_throughput for r in subset if r.supported and not r.early_stopped and r.num_successful_runs
            ]
            lines.append(
                "| "
                + " | ".join(
                    [
                        regime_label,
                        library,
                        str(len(subset)),
                        str(sum(1 for r in subset if r.num_successful_runs)),
                        str(sum(1 for r in subset if r.early_stopped)),
                        str(sum(1 for r in subset if not r.supported)),
                        _fmt(statistics.median(measured) if measured else None),
                    ],
                )
                + " |",
            )
    return "\n".join(lines)


def _best_library_table(rows: list[AggregateResult], regime: str) -> str:
    subset = _regime_rows(rows, regime)
    by_transform: dict[str, list[AggregateResult]] = defaultdict(list)
    for row in subset:
        if row.supported and not row.early_stopped and row.num_successful_runs:
            by_transform[row.transform].append(row)
    wins: dict[str, int] = defaultdict(int)
    speedups: list[tuple[str, str, float, str]] = []
    for transform, candidates in by_transform.items():
        if len(candidates) < 2:
            continue
        ordered = sorted(candidates, key=lambda r: r.median_throughput, reverse=True)
        wins[ordered[0].library] += 1
        if ordered[1].median_throughput > 0:
            speedups.append(
                (
                    transform,
                    ordered[0].library,
                    ordered[0].median_throughput / ordered[1].median_throughput,
                    ordered[1].library,
                ),
            )
    lines = [
        f"Measured winner counts for {REGIME_LABELS[regime]}:",
        "",
        "| Library | Wins |",
        "|---|---:|",
    ]
    for library, count in sorted(wins.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {library} | {count} |")
    lines.extend(
        [
            "",
            "Largest measured winner gaps:",
            "",
            "| Transform | Winner | Gap over second | Second |",
            "|---|---|---:|---|",
        ],
    )
    for transform, winner, gap, second in sorted(speedups, key=lambda item: item[2], reverse=True)[:10]:
        lines.append(f"| {transform} | {winner} | {_fmt(gap)}x | {second} |")
    return "\n".join(lines)


def _cpu_vs_gpu_table(rows: list[AggregateResult]) -> str:
    cpu = {(r.library, r.transform): r for r in rows if r.regime == "rgb_dataloader_cpu"}
    gpu = {(r.library, r.transform): r for r in rows if r.regime == "rgb_dataloader_gpu"}
    comparisons: list[tuple[str, str, float, float, float]] = []
    for key, gpu_row in gpu.items():
        cpu_row = cpu.get(key)
        if (
            cpu_row
            and cpu_row.supported
            and gpu_row.supported
            and not cpu_row.early_stopped
            and not gpu_row.early_stopped
            and cpu_row.median_throughput > 0
            and gpu_row.median_throughput > 0
        ):
            comparisons.append(
                (
                    key[0],
                    key[1],
                    cpu_row.median_throughput,
                    gpu_row.median_throughput,
                    gpu_row.median_throughput / cpu_row.median_throughput,
                ),
            )
    lines = [
        "| Library | Compared rows | GPU faster | CPU faster/equal | Median GPU/CPU ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for library in ["torchvision", "kornia", "dali"]:
        subset = [row for row in comparisons if row[0] == library]
        ratios = [row[4] for row in subset]
        lines.append(
            f"| {library} | {len(subset)} | {sum(1 for ratio in ratios if ratio > 1)} | "
            f"{sum(1 for ratio in ratios if ratio <= 1)} | {_fmt(statistics.median(ratios) if ratios else None, 2)}x |",
        )
    lines.extend(
        [
            "",
            "Largest GPU wins:",
            "",
            "| Library | Transform | CPU img/s | GPU img/s | GPU/CPU |",
            "|---|---|---:|---:|---:|",
        ],
    )
    for library, transform, cpu_tp, gpu_tp, ratio in sorted(comparisons, key=lambda item: item[4], reverse=True)[:10]:
        lines.append(f"| {library} | {transform} | {_fmt(cpu_tp)} | {_fmt(gpu_tp)} | {_fmt(ratio, 2)}x |")
    lines.extend(
        [
            "",
            "Largest CPU wins:",
            "",
            "| Library | Transform | CPU img/s | GPU img/s | GPU/CPU |",
            "|---|---|---:|---:|---:|",
        ],
    )
    for library, transform, cpu_tp, gpu_tp, ratio in sorted(comparisons, key=lambda item: item[4])[:10]:
        lines.append(f"| {library} | {transform} | {_fmt(cpu_tp)} | {_fmt(gpu_tp)} | {_fmt(ratio, 2)}x |")
    return "\n".join(lines)


def _albumentations_vs_gpu_table(rows: list[AggregateResult]) -> str:
    alb = {r.transform: r for r in rows if r.regime == "rgb_dataloader_cpu" and r.library == "albumentationsx"}
    gpu_rows = [
        r
        for r in rows
        if r.regime == "rgb_dataloader_gpu" and r.supported and not r.early_stopped and r.num_successful_runs
    ]
    comparisons: list[tuple[str, str, float, float, float]] = []
    for gpu_row in gpu_rows:
        cpu_row = alb.get(gpu_row.transform)
        if cpu_row and cpu_row.supported and not cpu_row.early_stopped and cpu_row.median_throughput > 0:
            comparisons.append(
                (
                    gpu_row.library,
                    gpu_row.transform,
                    cpu_row.median_throughput,
                    gpu_row.median_throughput,
                    gpu_row.median_throughput / cpu_row.median_throughput,
                ),
            )
    lines = [
        "| GPU library | Compared rows | GPU faster than AlbumentationsX CPU | AlbumentationsX CPU faster/equal | "
        "Median GPU/AlbumentationsX ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for library in ["torchvision", "kornia", "dali"]:
        subset = [row for row in comparisons if row[0] == library]
        ratios = [row[4] for row in subset]
        lines.append(
            f"| {library} | {len(subset)} | {sum(1 for ratio in ratios if ratio > 1)} | "
            f"{sum(1 for ratio in ratios if ratio <= 1)} | {_fmt(statistics.median(ratios) if ratios else None, 2)}x |",
        )
    return "\n".join(lines)


def _gpu_memory_table(rows: list[AggregateResult]) -> str:
    gpu_rows = [
        row
        for row in rows
        if row.regime == "rgb_dataloader_gpu" and row.gpu_peak_allocated_mb is not None and row.supported
    ]
    lines = [
        "| Library | Transform | Throughput img/s | Peak allocated MB | Peak reserved MB | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines.extend(
        (
            f"| {row.library} | {row.transform} | {_status_cell(row)} | {_fmt(row.gpu_peak_allocated_mb)} | "
            f"{_fmt(row.gpu_peak_reserved_mb)} | {row.status} |"
        )
        for row in sorted(gpu_rows, key=lambda r: r.gpu_peak_allocated_mb or 0, reverse=True)[:20]
    )
    return "\n".join(lines)


def _limitations_table(rows: list[AggregateResult]) -> str:
    interesting = [row for row in rows if not row.supported or row.early_stopped]
    lines = ["| Regime | Library | Transform | Status | Reason |", "|---|---|---|---|---|"]
    for row in sorted(interesting, key=lambda r: (r.regime, r.library, r.transform)):
        reason = row.reason.replace("\n", " ").replace("|", "\\|")
        if len(reason) > 180:
            reason = reason[:177] + "..."
        lines.append(
            f"| {REGIME_LABELS[row.regime]} | {row.library} | {row.transform} | {row.status} | {reason} |",
        )
    return "\n".join(lines)


def _limitations_summary_table(rows: list[AggregateResult]) -> str:
    lines = [
        "| Regime | Library | Full measured | Early-stopped | Unsupported |",
        "|---|---|---:|---:|---:|",
    ]
    for regime, regime_label in REGIME_LABELS.items():
        for library in LIBRARY_ORDER:
            subset = [r for r in rows if r.regime == regime and r.library == library]
            if not subset:
                continue
            full = sum(1 for r in subset if r.supported and not r.early_stopped and r.num_successful_runs)
            early = sum(1 for r in subset if r.supported and r.early_stopped)
            unsupported = sum(1 for r in subset if not r.supported)
            lines.append(f"| {regime_label} | {library} | {full} | {early} | {unsupported} |")
    return "\n".join(lines)


def _write_limitations_files(output_dir: Path, rows: list[AggregateResult]) -> None:
    interesting = [row for row in rows if not row.supported or row.early_stopped]
    fieldnames = ["regime", "regime_label", "library", "transform", "status", "reason", "source_files"]
    with (output_dir / "unsupported_and_early_stopped.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(interesting, key=lambda r: (r.regime, r.library, r.transform)):
            writer.writerow(
                {
                    "regime": row.regime,
                    "regime_label": REGIME_LABELS[row.regime],
                    "library": row.library,
                    "transform": row.transform,
                    "status": row.status,
                    "reason": row.reason,
                    "source_files": ";".join(row.source_files),
                },
            )
    (output_dir / "unsupported_and_early_stopped.md").write_text(
        "\n".join(
            [
                "# Unsupported And Early-Stopped Rows",
                "",
                "This supplement table preserves the full unsupported and slow-row detail used by the paper.",
                "",
                _limitations_table(rows),
                "",
            ],
        ),
        encoding="utf-8",
    )


def _headline_metrics(rows: list[AggregateResult]) -> dict[str, Any]:
    def median_for(regime: str, library: str) -> float | None:
        values = [r.median_throughput for r in rows if r.regime == regime and r.library == library and _is_measured(r)]
        return statistics.median(values) if values else None

    def winner_counts(regime: str) -> dict[str, int]:
        by_transform: dict[str, list[AggregateResult]] = defaultdict(list)
        for row in rows:
            if row.regime == regime and _is_measured(row):
                by_transform[row.transform].append(row)
        wins: dict[str, int] = defaultdict(int)
        for candidates in by_transform.values():
            if len(candidates) < 2:
                continue
            winner = max(candidates, key=lambda r: r.median_throughput)
            wins[winner.library] += 1
        return dict(sorted(wins.items()))

    alb_cpu = {
        r.transform: r
        for r in rows
        if r.regime == "rgb_dataloader_cpu" and r.library == "albumentationsx" and _is_measured(r)
    }
    gpu_vs_alb: dict[str, dict[str, float | int | None]] = {}
    for library in ["torchvision", "kornia", "dali"]:
        ratios: list[float] = []
        for row in rows:
            if row.regime != "rgb_dataloader_gpu" or row.library != library or not _is_measured(row):
                continue
            cpu_row = alb_cpu.get(row.transform)
            if cpu_row and cpu_row.median_throughput > 0:
                ratios.append(row.median_throughput / cpu_row.median_throughput)
        gpu_vs_alb[library] = {
            "compared_rows": len(ratios),
            "gpu_faster": sum(1 for ratio in ratios if ratio > 1),
            "median_ratio": statistics.median(ratios) if ratios else None,
        }

    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for regime in REGIME_LABELS:
        coverage[regime] = {}
        for library in LIBRARY_ORDER:
            subset = [r for r in rows if r.regime == regime and r.library == library]
            if subset:
                coverage[regime][library] = {
                    "rows": len(subset),
                    "full": sum(1 for r in subset if _is_measured(r)),
                    "early_stopped": sum(1 for r in subset if r.supported and r.early_stopped),
                    "unsupported": sum(1 for r in subset if not r.supported),
                }

    return {
        "median_throughput": {
            regime: {
                library: median_for(regime, library)
                for library in LIBRARY_ORDER
                if median_for(regime, library) is not None
            }
            for regime in REGIME_LABELS
        },
        "winner_counts": {regime: winner_counts(regime) for regime in REGIME_LABELS},
        "gpu_vs_albumentationsx_cpu": gpu_vs_alb,
        "open_dataloader": _open_dataloader_stats(rows),
        "coverage": coverage,
    }


def _write_markdown(path: Path, rows: list[AggregateResult]) -> str:
    sections = [
        "# Generated Paper Data",
        "",
        "Generated from committed `results/published/*` snapshots plus local `gcp_runs/prod-*` artifacts that have not yet been published. Throughput units are images/second.",
        "",
        "## Coverage Summary",
        "",
        _library_summary(rows),
        "",
        "## Absolute/Open Production DataLoader Category",
        "",
        _open_dataloader_leaderboard_table(rows),
        "",
        "## Winner Counts",
        "",
        _best_library_table(rows, "rgb_micro_cpu"),
        "",
        _best_library_table(rows, "rgb_dataloader_cpu"),
        "",
        _best_library_table(rows, "rgb_micro_gpu"),
        "",
        _best_library_table(rows, "rgb_dataloader_gpu"),
        "",
        "## CPU DataLoader vs GPU DataLoader",
        "",
        _cpu_vs_gpu_table(rows),
        "",
        "## AlbumentationsX CPU DataLoader vs GPU DataLoader",
        "",
        _albumentations_vs_gpu_table(rows),
        "",
        "## GPU Memory",
        "",
        _gpu_memory_table(rows),
        "",
        "## Unsupported And Early-Stopped Summary",
        "",
        _limitations_summary_table(rows),
        "",
        "Full row-level reasons are generated in the public paper-data supplement after figure generation.",
        "",
    ]
    content = "\n".join(sections)
    path.write_text(content, encoding="utf-8")
    return content


def _update_draft(draft_path: Path, generated_markdown: str) -> None:
    text = draft_path.read_text(encoding="utf-8")
    block = f"{GENERATED_START}\n\n{generated_markdown}\n{GENERATED_END}"
    if GENERATED_START in text and GENERATED_END in text:
        start = text.index(GENERATED_START)
        end = text.index(GENERATED_END) + len(GENERATED_END)
        text = f"{text[:start]}{block}{text[end:]}"
    else:
        insert_after = "## 4. Results\n"
        if insert_after not in text:
            text = f"{text.rstrip()}\n\n{block}\n"
        else:
            text = text.replace(insert_after, f"{insert_after}\n{block}\n", 1)
    draft_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables from production benchmark JSON artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--update-draft", action="store_true")
    args = parser.parse_args()

    sources = _iter_sources()
    rows = _aggregate_sources(sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "all_results.csv", rows)
    for regime in REGIME_LABELS:
        regime_rows = _regime_rows(rows, regime)
        _write_csv(args.output_dir / f"{regime}.csv", regime_rows)
        _write_pivot_csv(args.output_dir / f"{regime}_pivot.csv", regime_rows)
    _write_limitations_files(args.output_dir, rows)
    _write_open_dataloader_files(args.output_dir, rows)
    markdown = _write_markdown(args.output_dir / "summary.md", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_source_rows": len(sources),
                "num_aggregate_rows": len(rows),
                "regimes": {regime: len(_regime_rows(rows, regime)) for regime in REGIME_LABELS},
                "headlines": _headline_metrics(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.update_draft:
        _update_draft(args.draft, markdown)
    sys.stdout.write(f"Wrote {args.output_dir}\n")
    if args.update_draft:
        sys.stdout.write(f"Updated {args.draft}\n")


if __name__ == "__main__":
    main()
