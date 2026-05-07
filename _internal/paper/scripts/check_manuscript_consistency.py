# ruff: noqa: INP001, E501
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DRAFT = ROOT / "_internal" / "paper" / "draft.md"
DEFAULT_LATEX = ROOT / "_internal" / "paper" / "neurips_2026_ed" / "main.tex"
DEFAULT_SUMMARY = ROOT / "_internal" / "paper" / "generated" / "summary.json"
DEFAULT_ALL_RESULTS = ROOT / "_internal" / "paper" / "generated" / "all_results.csv"
DEFAULT_SUPPORT_MATRIX = ROOT / "_internal" / "paper" / "generated" / "production_support_matrix.csv"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return payload


def _find_required(text: str, pattern: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        msg = f"Missing manuscript claim: {label}"
        raise AssertionError(msg)
    return match.group(0)


def _assert_contains(text: str, expected: str) -> None:
    if expected not in text:
        msg = f"Expected manuscript to contain: {expected}"
        raise AssertionError(msg)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _kornia_gpu_measured_dali_not(path: Path) -> int:
    gpu_rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["regime"] != "rgb_dataloader_gpu":
                continue
            gpu_rows[(row["library"], row["transform"])] = row

    kornia_measured = {
        transform
        for (library, transform), row in gpu_rows.items()
        if library == "kornia"
        and _as_bool(row["supported"])
        and not _as_bool(row["early_stopped"])
        and int(row["num_successful_runs"]) > 0
    }
    dali_measured = {
        transform
        for (library, transform), row in gpu_rows.items()
        if library == "dali"
        and _as_bool(row["supported"])
        and not _as_bool(row["early_stopped"])
        and int(row["num_successful_runs"]) > 0
    }
    return len(kornia_measured - dali_measured)


def _load_support_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    draft = DEFAULT_DRAFT.read_text(encoding="utf-8") if DEFAULT_DRAFT.exists() else ""
    latex = DEFAULT_LATEX.read_text(encoding="utf-8")
    manuscript = f"{draft}\n{latex}"
    required_targets = [latex]
    if draft:
        required_targets.append(draft)
    summary = _load_json(DEFAULT_SUMMARY)
    headlines = summary["headlines"]

    cpu_wins = headlines["winner_counts"]["rgb_dataloader_cpu"]["albumentationsx"]
    cpu_rows = headlines["coverage"]["rgb_dataloader_cpu"]["albumentationsx"]["full"]
    alb_cpu_median = headlines["median_throughput"]["rgb_dataloader_cpu"]["albumentationsx"]
    tv_gpu = headlines["gpu_vs_albumentationsx_cpu"]["torchvision"]
    kornia_gpu = headlines["gpu_vs_albumentationsx_cpu"]["kornia"]
    dali_gpu = headlines["gpu_vs_albumentationsx_cpu"]["dali"]
    open_dataloader = headlines["open_dataloader"]
    open_leaderboard = open_dataloader["leaderboard"]
    alb_open = next(row for row in open_leaderboard if row["implementation"] == "AlbumentationsX CPU")
    dali_open = next(row for row in open_leaderboard if row["implementation"] == "DALI GPU")
    kornia_cpu_open = next(row for row in open_leaderboard if row["implementation"] == "Kornia CPU")
    open_exceptions = {
        row["display_transform"]: row["implementation"]
        for row in open_dataloader["winner_rows"]
        if row["implementation"] != "AlbumentationsX CPU"
    }
    alb_cpu_coverage = headlines["coverage"]["rgb_dataloader_cpu"]["albumentationsx"]
    kornia_cpu_coverage = headlines["coverage"]["rgb_dataloader_cpu"]["kornia"]
    tv_cpu_coverage = headlines["coverage"]["rgb_dataloader_cpu"]["torchvision"]
    pillow_cpu_coverage = headlines["coverage"]["rgb_dataloader_cpu"]["pillow"]
    kornia_gpu_coverage = headlines["coverage"]["rgb_dataloader_gpu"]["kornia"]
    tv_gpu_coverage = headlines["coverage"]["rgb_dataloader_gpu"]["torchvision"]
    dali_coverage = headlines["coverage"]["rgb_dataloader_gpu"]["dali"]
    kornia_measured_dali_not = _kornia_gpu_measured_dali_not(DEFAULT_ALL_RESULTS)
    support_matrix = _load_support_matrix(DEFAULT_SUPPORT_MATRIX)
    if len(support_matrix) != alb_cpu_coverage["rows"]:
        msg = f"Expected {alb_cpu_coverage['rows']} support-matrix rows, found {len(support_matrix)}"
        raise AssertionError(msg)

    _assert_contains(manuscript, f"AlbumentationsX is ranked first for {cpu_wins} of {cpu_rows}")
    _assert_contains(manuscript, f"median throughput of {alb_cpu_median:.1f} images/s")
    _assert_contains(
        manuscript,
        f"AlbumentationsX CPU is ranked first for {alb_open['wins']} of {alb_open['universe_rows']}",
    )
    _assert_contains(manuscript, f"DALI GPU is ranked first for {dali_open['wins']} of {dali_open['universe_rows']}")
    _assert_contains(
        manuscript,
        f"Kornia CPU is ranked first for {kornia_cpu_open['wins']} of {kornia_cpu_open['universe_rows']}",
    )
    for transform, implementation in open_exceptions.items():
        _assert_contains(manuscript, transform)
        _assert_contains(manuscript, implementation)
    _assert_contains(
        manuscript,
        f"TorchVision GPU has higher throughput for {tv_gpu['gpu_faster']} of {tv_gpu['compared_rows']}",
    )
    _assert_contains(
        manuscript,
        f"Kornia GPU for {kornia_gpu['gpu_faster']} of {kornia_gpu['compared_rows']}",
    )
    _assert_contains(
        manuscript,
        f"DALI for {dali_gpu['gpu_faster']} of {dali_gpu['compared_rows']}",
    )
    for text in required_targets:
        _assert_contains(
            text,
            f"AlbumentationsX supports and measures {alb_cpu_coverage['full']} of {alb_cpu_coverage['rows']} CPU DataLoader recipes",
        )
        _assert_contains(
            text,
            f"Kornia supports and measures {kornia_cpu_coverage['full']} of {alb_cpu_coverage['rows']} CPU DataLoader recipes",
        )
        _assert_contains(
            text,
            f"TorchVision supports and measures {tv_cpu_coverage['full']} of {alb_cpu_coverage['rows']}",
        )
        _assert_contains(
            text,
            f"Pillow supports and measures {pillow_cpu_coverage['full']} of {alb_cpu_coverage['rows']}",
        )
        _assert_contains(
            text,
            f"Kornia GPU supports and measures {kornia_gpu_coverage['full']} of {alb_cpu_coverage['rows']}",
        )
        _assert_contains(
            text,
            f"TorchVision GPU supports and measures {tv_gpu_coverage['full']} of {alb_cpu_coverage['rows']}",
        )
        _assert_contains(
            text,
            f"DALI supports and measures {dali_coverage['full']} of {alb_cpu_coverage['rows']}",
        )
        _assert_contains(
            text,
            "Recipes not supported by a given implementation and early-stopped recipe-implementation pairs are not assigned zero throughput",
        )
        _assert_contains(
            text,
            f"Kornia GPU supports and measures {kornia_measured_dali_not} recipes that DALI does not",
        )
        _assert_contains(text, "open production DataLoader category")
        _assert_contains(text, "Elastic")
        _assert_contains(text, "production DataLoader support")
        _assert_contains(text, "production_support_matrix")

    for label in [
        r"\label{tab:benchmarking-pain-points}",
        r"\label{fig:open-dataloader-leaderboard}",
        r"\label{fig:coverage-throughput}",
        r"\label{fig:gpu-vs-cpu}",
        r"\label{fig:gpu-memory}",
    ]:
        _assert_contains(latex, label)
    if draft:
        for title in [
            "Open production DataLoader category",
            "Coverage breadth versus measured throughput",
            "GPU DataLoader pipelines versus AlbumentationsX CPU",
        ]:
            _assert_contains(draft, title)

    forbidden = [
        "TODO:",
        "No DALI RGB baseline",
        "1 of 29 compared",
        "none of the 29 compared",
        "0 of 29",
        "DALI fully measures",
        "DALI coverage limits",
        "not full transform parity",
        "direct transform coverage",
        "DALI GPU covers",
        "recipe-library pairs",
        "recipe-library-pair",
        "row-level provenance",
        "row winner",
        "GPU microbenchmark rows",
        "absolute fastest library",
        "fig:benchmarking-pitfalls",
        "figures/benchmarking_pitfalls",
        "fig:abstract-claims",
        "figures/abstract_claims",
        "gpu_vs_albumentationsx_cpu_boxplot",
        "baseline to beat",
        "beats GPU augmentation",
        "beat GPU augmentation",
        "rescue an inefficient implementation",
        "strongest production-path",
    ]
    for phrase in forbidden:
        if phrase in manuscript:
            msg = f"Forbidden stale manuscript phrase found: {phrase}"
            raise AssertionError(msg)

    if "_internal/" in latex:
        msg = "Internal-only paths must not appear in the LaTeX manuscript."
        raise AssertionError(msg)

    _find_required(manuscript, r"NeurIPS 2026 Evaluations and Datasets", "target track")
    _find_required(manuscript, r"evaluation card", "evaluation-card appendix")
    _find_required(manuscript, r"anonymized artifact", "artifact anonymization")
    sys.stdout.write("Manuscript consistency checks passed.\n")


if __name__ == "__main__":
    main()
