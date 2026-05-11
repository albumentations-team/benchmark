# ruff: noqa: INP001, E402, E501, S603, RUF059, PD010, ARG001, PERF401, T201, PLW0603
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from common import (
    LIBRARY_DISPLAY,
    LIBRARY_ORDER,
    PALETTE,
    REGIME_ORDER,
    fmt_ratio,
    latex_escape,
    recipe_display_name,
)

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/benchmark-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/benchmark-cache")

import matplotlib.pyplot as plt

GENERATED = ROOT / "_internal" / "paper" / "generated"
FIGURES = ROOT / "_internal" / "paper" / "figures"
PUBLIC_DATA = ROOT / "docs" / "paper_data"
PUBLIC_FIGURES = ROOT / "docs" / "paper_figures"
PAPER_DIR = ROOT / "_internal" / "paper" / "neurips_2026_ed"
PAPER_FIGURES = PAPER_DIR / "figures"
README = ROOT / "README.md"
DRAFT = ROOT / "_internal" / "paper" / "draft.md"

PAPER_UNIVERSE_REGIME = "rgb_dataloader_cpu"
PAPER_UNIVERSE_LIBRARY = "albumentationsx"
DATALOADER_REGIMES = ["rgb_dataloader_cpu", "rgb_dataloader_gpu"]
DATALOADER_FACETS = {
    "CPU DataLoader": ["albumentationsx", "kornia", "torchvision", "pillow"],
    "GPU DataLoader": ["kornia", "torchvision", "dali"],
}
PRODUCTION_MATRIX_COLUMNS = [
    ("albumentationsx_cpu", "AlbX CPU", "rgb_dataloader_cpu", "albumentationsx"),
    ("kornia_cpu", "Kornia CPU", "rgb_dataloader_cpu", "kornia"),
    ("kornia_gpu", "Kornia GPU", "rgb_dataloader_gpu", "kornia"),
    ("torchvision_cpu", "TV CPU", "rgb_dataloader_cpu", "torchvision"),
    ("torchvision_gpu", "TV GPU", "rgb_dataloader_gpu", "torchvision"),
    ("pillow_cpu", "Pillow CPU", "rgb_dataloader_cpu", "pillow"),
    ("dali_gpu", "DALI GPU", "rgb_dataloader_gpu", "dali"),
]
MAIN_FIGURES = [
    {
        "path": "open_dataloader_leaderboard.png",
        "title": "Figure 1. Open production DataLoader category",
        "caption": (
            "CPU and GPU DataLoader implementations compete together over the same 57-recipe universe. "
            "Bars show median measured-row throughput; labels show full measured coverage and open-category wins. "
            "AlbumentationsX CPU wins 52 of 57 recipes and has the highest median throughput."
        ),
    },
    {
        "path": "coverage_vs_throughput.png",
        "title": "Figure 2. Coverage breadth versus measured throughput",
        "caption": (
            "DataLoader coverage and throughput are distinct benchmark axes. The x-axis is the count of full "
            "measured recipes over the canonical 57 CPU DataLoader recipes, and the y-axis is median throughput "
            "over measured rows only. The Elastic drill-down shows that GPU execution does not rescue a slow "
            "implementation of a hard transform."
        ),
    },
    {
        "path": "gpu_vs_albumentationsx_cpu_ratios.png",
        "title": "Figure 3. GPU DataLoader pipelines versus AlbumentationsX CPU",
        "caption": (
            "Each point is a paired GPU DataLoader recipe divided by the AlbumentationsX CPU DataLoader throughput "
            "for the same recipe. The dashed line marks parity. Most GPU rows fall below parity once the full "
            "DataLoader path is measured."
        ),
    },
    {
        "path": "gpu_memory_vs_throughput.png",
        "title": "Figure 4. GPU memory consumed by augmentation pipelines",
        "caption": (
            "GPU augmentation also consumes accelerator memory that would otherwise be available to model "
            "parameters, activations, optimizer state, or larger batches. Each point is a measured GPU "
            "DataLoader row with peak allocated memory recorded during the benchmark."
        ),
    },
]
APPENDIX_FIGURES = [
    {
        "path": "winner_counts.png",
        "title": "Appendix Figure. Winner counts by benchmark regime",
        "caption": (
            "Measured winner counts among comparable measured transforms by regime. The conclusion changes when "
            "moving from augmentation-only microbenchmarks to production-style DataLoader measurements."
        ),
    },
]


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _run_data_generator(extra_args: list[str]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/paper/generate_paper_data.py"),
            "--output-dir",
            str(GENERATED),
            *extra_args,
        ],
        check=True,
        cwd=ROOT,
    )


def _load_results() -> pd.DataFrame:
    df = pd.read_csv(GENERATED / "all_results.csv")
    df["supported"] = df["supported"].astype(bool)
    df["early_stopped"] = df["early_stopped"].astype(bool)
    df["measured"] = df["supported"] & ~df["early_stopped"] & (df["num_successful_runs"] > 0)
    df["library"] = pd.Categorical(df["library"], categories=LIBRARY_ORDER, ordered=True)
    df["regime_label"] = pd.Categorical(df["regime_label"], categories=REGIME_ORDER, ordered=True)
    return df


def _savefig(path: Path, *, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _bar_label(ax: plt.Axes, fmt: str = "{:.0f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not math.isfinite(height) or height <= 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_y() + height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _paper_recipe_universe(df: pd.DataFrame) -> list[str]:
    universe = sorted(
        df[(df["regime"] == PAPER_UNIVERSE_REGIME) & (df["library"] == PAPER_UNIVERSE_LIBRARY)]["transform"]
        .astype(str)
        .unique()
        .tolist(),
    )
    if len(universe) != 57:
        msg = (
            f"Expected 57 canonical paper recipes from "
            f"{PAPER_UNIVERSE_LIBRARY}/{PAPER_UNIVERSE_REGIME}, found {len(universe)}"
        )
        raise RuntimeError(msg)
    return universe


def _open_dataloader_leaderboard_plot() -> pd.DataFrame:
    leaderboard = pd.read_csv(GENERATED / "open_dataloader_leaderboard.csv")
    leaderboard["library"] = leaderboard["library"].astype(str)
    plot_df = leaderboard.sort_values("median_throughput", ascending=True).reset_index(drop=True)
    plot_df.to_csv(GENERATED / "figure_open_dataloader_leaderboard.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y_positions = list(range(len(plot_df)))
    colors = [PALETTE.get(str(row.library), "#777777") for row in plot_df.itertuples()]
    labels = [str(row.implementation) for row in plot_df.itertuples()]
    values = [float(row.median_throughput) for row in plot_df.itertuples()]
    ax.barh(y_positions, values, color=colors, height=0.62)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Median measured-row throughput (img/s)")
    ax.set_title("Open production DataLoader category: CPU and GPU pipelines compete together")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.75)
    ax.set_xlim(0, max(values) * 1.36)
    for y, row in zip(y_positions, plot_df.itertuples(), strict=True):
        ax.text(
            float(row.median_throughput) + max(values) * 0.02,
            y,
            f"{float(row.median_throughput):.0f} img/s  |  {int(row.full)}/{int(row.universe_rows)}  |  {int(row.wins)} wins",
            va="center",
            ha="left",
            fontsize=7.7,
        )
    _savefig(FIGURES / "open_dataloader_leaderboard.png")
    return leaderboard


def _dataloader_coverage_vs_throughput(df: pd.DataFrame) -> pd.DataFrame:
    universe = _paper_recipe_universe(df)
    universe_set = set(universe)
    rows: list[dict[str, object]] = []
    for regime in DATALOADER_REGIMES:
        regime_label = str(df.loc[df["regime"] == regime, "regime_label"].iloc[0])
        for library in DATALOADER_FACETS[regime_label]:
            subset = df[
                (df["regime"] == regime) & (df["library"] == library) & (df["transform"].astype(str).isin(universe_set))
            ].copy()
            present_transforms = set(subset["transform"].astype(str))
            measured = subset[subset["measured"]]
            full = int(measured["transform"].astype(str).nunique())
            early_stopped = int(
                subset.loc[subset["supported"] & subset["early_stopped"], "transform"].astype(str).nunique(),
            )
            explicit_unsupported = int(
                subset.loc[~subset["supported"], "transform"].astype(str).nunique(),
            )
            absent = len(universe_set - present_transforms)
            median = float(measured["median_throughput"].median()) if not measured.empty else math.nan
            rows.append(
                {
                    "regime": regime,
                    "regime_label": regime_label,
                    "library": library,
                    "full": full,
                    "early_stopped": early_stopped,
                    "explicit_unsupported": explicit_unsupported,
                    "absent_or_not_in_result": absent,
                    "universe_rows": len(universe),
                    "coverage_pct": full / len(universe),
                    "median_throughput": median,
                },
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(GENERATED / "figure_coverage_vs_throughput.csv", index=False)

    elastic = df[
        df["regime"].isin(DATALOADER_REGIMES)
        & (df["transform"].astype(str) == "RandomCrop224+Elastic+Normalize+ToTensor")
    ].copy()
    elastic.to_csv(GENERATED / "figure_elastic_row.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.15), sharey=True)
    label_offsets = {
        ("CPU DataLoader", "albumentationsx"): (-14, -32),
        ("CPU DataLoader", "kornia"): (8, 8),
        ("CPU DataLoader", "torchvision"): (8, 10),
        ("CPU DataLoader", "pillow"): (-42, -16),
        ("GPU DataLoader", "kornia"): (-20, 8),
        ("GPU DataLoader", "torchvision"): (8, -16),
        ("GPU DataLoader", "dali"): (-10, 10),
    }
    for ax, (regime_label, libraries) in zip(axes, DATALOADER_FACETS.items(), strict=True):
        facet = summary[summary["regime_label"] == regime_label]
        for library in libraries:
            row = facet[facet["library"] == library].iloc[0]
            x_value = row["full"]
            y_value = row["median_throughput"]
            ax.scatter(
                x_value,
                y_value,
                s=95,
                color=PALETTE.get(library, "#777777"),
                edgecolor="#222222",
                linewidth=0.6,
                zorder=3,
            )
            dx, dy = label_offsets.get((regime_label, library), (8, 8))
            ax.annotate(
                f"{LIBRARY_DISPLAY.get(library, library)}\n{int(row['full'])}/{int(row['universe_rows'])}",
                xy=(x_value, y_value),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.0,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
            )
        ax.set_title(regime_label)
        ax.set_xlabel("Full measured recipes (of 57)")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 5100)
        ax.grid(axis="both", color="#dddddd", linewidth=0.6, alpha=0.7)
    elastic_lookup = {(str(row.regime_label), str(row.library)): row for row in elastic.itertuples()}
    elastic_line = (
        "Elastic drill-down: "
        f"AlbX CPU {elastic_lookup[('CPU DataLoader', 'albumentationsx')].median_throughput:.0f} img/s; "
        f"Kornia CPU/GPU {elastic_lookup[('CPU DataLoader', 'kornia')].median_throughput:.0f}/"
        f"{elastic_lookup[('GPU DataLoader', 'kornia')].median_throughput:.0f}; "
        f"TorchVision CPU/GPU {elastic_lookup[('CPU DataLoader', 'torchvision')].median_throughput:.0f}/"
        f"{elastic_lookup[('GPU DataLoader', 'torchvision')].median_throughput:.0f}; "
        "DALI GPU unsupported.\nGPU does not rescue slow implementation."
    )
    axes[0].set_ylabel("Median measured-row throughput (img/s)")
    fig.suptitle("Coverage breadth and measured throughput are separate axes", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        elastic_line,
        ha="center",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.28"},
    )
    fig.text(
        0.5,
        -0.105,
        "Missing, unsupported, and early-stopped rows reduce coverage; they are not assigned zero throughput.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.26, wspace=0.10)
    _savefig(FIGURES / "coverage_vs_throughput.png", tight=False)
    return summary


def _format_ci_cell(row: pd.Series, *, latex: bool = False, bold: bool = False) -> str:
    median = float(row["median_throughput"])
    ci95 = float(row["ci95"]) if pd.notna(row["ci95"]) else 0.0
    if latex:
        body = rf"\({median:.0f}{{\pm}}{ci95:.0f}\)"
        return rf"\textbf{{{body}}}" if bold else body
    body = f"{median:.0f} +/- {ci95:.0f}"
    return f"**{body}**" if bold else body


def _matrix_cell(row: pd.Series | None, *, latex: bool = False, bold: bool = False) -> str:
    if row is None:
        return "-"
    if not bool(row["supported"]):
        return "-"
    if bool(row["early_stopped"]):
        return "-"
    if bool(row["measured"]):
        return _format_ci_cell(row, latex=latex, bold=bold)
    return "-"


def _dominant_columns(cells: dict[str, pd.Series | None]) -> set[str]:
    measured: list[tuple[str, float, float]] = []
    for column, row in cells.items():
        if row is None or not bool(row["measured"]):
            continue
        median = float(row["median_throughput"])
        ci95 = float(row["ci95"]) if pd.notna(row["ci95"]) else 0.0
        measured.append((column, median, ci95))
    if len(measured) < 2:
        return set()
    measured.sort(key=lambda item: item[1], reverse=True)
    best_column, best_median, best_ci = measured[0]
    strongest_competitor_upper = max(median + ci for _, median, ci in measured[1:])
    if best_median - best_ci > strongest_competitor_upper:
        return {best_column}
    return set()


def _write_markdown_table(matrix: pd.DataFrame) -> None:
    headers = ["Transform", *[title for _, title, _, _ in PRODUCTION_MATRIX_COLUMNS]]
    lines = [
        "# RGB Production DataLoader Support Matrix",
        "",
        "Rows are the canonical 57 AlbumentationsX CPU DataLoader recipes. Numeric cells are median "
        "measured-row throughput +/- 95% CI in images/s. `-` means absent, unsupported, early-stopped, "
        "or otherwise not fully measured. Bold marks the recipe leader only when its lower 95% confidence "
        "bound is above every other measured implementation's upper bound.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *["---:" for _ in PRODUCTION_MATRIX_COLUMNS]]) + " |",
    ]
    for row in matrix.itertuples(index=False):
        values = [row.transform, *[getattr(row, column) for column, _, _, _ in PRODUCTION_MATRIX_COLUMNS]]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    (GENERATED / "production_support_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_support_table(matrix: pd.DataFrame) -> None:
    header = "Transform & AlbX CPU & Kornia CPU & Kornia GPU & TV CPU & TV GPU & Pillow CPU & DALI GPU \\\\"
    lines = [
        r"{\tiny",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\begin{longtable}{p{0.27\linewidth}rrrrrrr}",
        r"\caption{RGB production DataLoader support matrix. Rows are the canonical 57 AlbumentationsX CPU DataLoader recipes. Numeric cells are median throughput \(\pm\) 95\% CI in images/s for full measured rows. A dash means absent, unsupported, early-stopped, or otherwise not fully measured. Bold marks a recipe leader only when its lower 95\% confidence bound is above every other measured implementation's upper bound.}\label{tab:production-support-matrix}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]
    for row in matrix.itertuples(index=False):
        values = [
            latex_escape(row.transform),
            *[str(getattr(row, column)) for column, _, _, _ in PRODUCTION_MATRIX_COLUMNS],
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}"])
    table = "\n".join(lines) + "\n"
    (GENERATED / "production_support_matrix_table.tex").write_text(table, encoding="utf-8")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "production_support_matrix_table.tex").write_text(table, encoding="utf-8")


def _production_support_matrix(df: pd.DataFrame) -> pd.DataFrame:
    universe = _paper_recipe_universe(df)
    indexed = {(str(row.regime), str(row.library), str(row.transform)): row for row in df.itertuples()}
    display_rows: list[dict[str, str]] = []
    latex_rows: list[dict[str, str]] = []
    for recipe in universe:
        display_row = {"transform": recipe_display_name(recipe), "recipe": recipe}
        latex_row = {"transform": recipe_display_name(recipe), "recipe": recipe}
        cells: dict[str, pd.Series | None] = {}
        for column, _, regime, library in PRODUCTION_MATRIX_COLUMNS:
            row = indexed.get((regime, library, recipe))
            cells[column] = pd.Series(row._asdict()) if row is not None else None
        dominant = _dominant_columns(cells)
        for column, _, _, _ in PRODUCTION_MATRIX_COLUMNS:
            display_row[column] = _matrix_cell(cells[column], bold=column in dominant)
            latex_row[column] = _matrix_cell(cells[column], latex=True, bold=column in dominant)
        display_rows.append(display_row)
        latex_rows.append(latex_row)

    matrix = pd.DataFrame(display_rows)
    latex_matrix = pd.DataFrame(latex_rows)
    matrix.to_csv(GENERATED / "production_support_matrix.csv", index=False)
    _write_markdown_table(matrix.drop(columns=["recipe"]))
    _write_latex_support_table(latex_matrix.drop(columns=["recipe"]))
    return matrix


def _coverage_plot(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.assign(
            full=df["measured"].astype(int),
            early=(df["supported"] & df["early_stopped"]).astype(int),
            unsupported=(~df["supported"]).astype(int),
        )
        .groupby(["regime_label", "library"], observed=True)[["full", "early", "unsupported"]]
        .sum()
        .reset_index()
    )
    summary.to_csv(GENERATED / "figure_coverage_summary.csv", index=False)

    labels = [f"{row.regime_label}\n{row.library}" for row in summary.itertuples()]
    x = range(len(summary))
    plt.figure(figsize=(12, 4.8))
    bottom = [0] * len(summary)
    for column, color in [("full", "#2c7a4b"), ("early", "#d9a441"), ("unsupported", "#b85c5c")]:
        values = summary[column].tolist()
        plt.bar(x, values, bottom=bottom, label=column.replace("_", " "), color=color)
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Transform rows")
    plt.title("Coverage by regime and library")
    plt.legend(frameon=False, ncols=3)
    _savefig(FIGURES / "coverage_by_regime.png")
    return summary


def _winner_counts(df: pd.DataFrame) -> pd.DataFrame:
    winners: list[dict[str, object]] = []
    for regime, group in df[df["measured"]].groupby("regime_label", observed=True):
        for transform, transform_group in group.groupby("transform"):
            if len(transform_group) < 2:
                continue
            winner = transform_group.sort_values("median_throughput", ascending=False).iloc[0]
            second = transform_group.sort_values("median_throughput", ascending=False).iloc[1]
            winners.append(
                {
                    "regime_label": regime,
                    "transform": transform,
                    "winner": winner["library"],
                    "winner_throughput": winner["median_throughput"],
                    "second": second["library"],
                    "second_throughput": second["median_throughput"],
                    "gap": winner["median_throughput"] / second["median_throughput"],
                },
            )
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(GENERATED / "figure_winner_rows.csv", index=False)
    counts = winners_df.groupby(["regime_label", "winner"], observed=True).size().reset_index(name="wins")
    counts.to_csv(GENERATED / "figure_winner_counts.csv", index=False)

    pivot = counts.pivot(index="regime_label", columns="winner", values="wins").fillna(0)
    pivot = pivot[[lib for lib in LIBRARY_ORDER if lib in pivot.columns]]
    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 4.5),
        color=[PALETTE.get(col, "#777777") for col in pivot.columns],
    )
    ax.set_ylabel("Measured winner count")
    ax.set_xlabel("")
    ax.set_title("Winner counts among comparable measured transforms")
    ax.legend(frameon=False, ncols=3)
    plt.xticks(rotation=20, ha="right")
    _savefig(FIGURES / "winner_counts.png")
    return winners_df


def _albumentations_vs_gpu(df: pd.DataFrame) -> pd.DataFrame:
    cpu_alb = df[(df["regime"] == "rgb_dataloader_cpu") & (df["library"] == "albumentationsx") & df["measured"]][
        ["transform", "median_throughput"]
    ].rename(columns={"median_throughput": "albumentationsx_cpu"})
    gpu = df[(df["regime"] == "rgb_dataloader_gpu") & df["measured"]][["library", "transform", "median_throughput"]]
    merged = gpu.merge(cpu_alb, on="transform", how="inner")
    merged["gpu_to_albumentationsx_cpu"] = merged["median_throughput"] / merged["albumentationsx_cpu"]
    merged.to_csv(GENERATED / "figure_gpu_vs_albumentationsx_cpu.csv", index=False)

    gpu_micro_winners: list[str] = []
    gpu_micro = df[(df["regime"] == "rgb_micro_gpu") & df["measured"]]
    for _, transform_group in gpu_micro.groupby("transform"):
        if len(transform_group) < 2:
            continue
        winner = transform_group.sort_values("median_throughput", ascending=False).iloc[0]
        gpu_micro_winners.append(str(winner["library"]))
    gpu_micro_counts = pd.Series(gpu_micro_winners).value_counts()

    fig, (ax_micro, ax) = plt.subplots(
        1,
        2,
        figsize=(7.6, 3.65),
        gridspec_kw={"width_ratios": [0.95, 3.4]},
    )
    micro_libraries = ["torchvision", "kornia"]
    micro_values = [int(gpu_micro_counts.get(library, 0)) for library in micro_libraries]
    ax_micro.bar(
        micro_libraries,
        micro_values,
        color=[PALETTE.get(library, "#777777") for library in micro_libraries],
        width=0.62,
    )
    ax_micro.set_title("GPU micro\nwinners")
    ax_micro.set_ylabel("Winner count")
    ax_micro.set_ylim(0, max(micro_values) + 4)
    ax_micro.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.75)
    ax_micro.set_xticks(
        range(len(micro_libraries)),
        [LIBRARY_DISPLAY[library] for library in micro_libraries],
        rotation=25,
        ha="right",
    )
    for index, value in enumerate(micro_values):
        ax_micro.text(index, value + 0.5, str(value), ha="center", va="bottom", fontsize=9, fontweight="bold")

    libraries = ["torchvision", "kornia", "dali"]
    x_positions = list(range(len(libraries)))
    for x_position, library in zip(x_positions, libraries, strict=True):
        subset = (
            merged.loc[merged["library"] == library, "gpu_to_albumentationsx_cpu"].sort_values().reset_index(drop=True)
        )
        if subset.empty:
            continue
        jitter = pd.Series([0.0] * len(subset))
        if len(subset) > 1:
            jitter = pd.Series([(-0.16 + 0.32 * index / (len(subset) - 1)) for index in range(len(subset))])
        ax.scatter(
            x_position + jitter,
            subset,
            s=38,
            color=PALETTE.get(library, "#777777"),
            edgecolor="#222222",
            linewidth=0.35,
            alpha=0.82,
            zorder=3,
        )
        median = float(subset.median())
        ax.scatter(
            [x_position],
            [median],
            marker="D",
            s=72,
            color="#111111",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        wins = int((subset > 1.0).sum())
        ax.text(
            x_position,
            2.35,
            f"{wins}/{len(subset)} wins",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            x_position,
            0.035,
            f"median {median:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color="#444444",
        )

    ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 3.0)
    ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    ax.set_yticklabels(["0.05x", "0.10x", "0.25x", "0.50x", "1.00x", "2.00x"])
    ax.set_xticks(x_positions, [LIBRARY_DISPLAY[library] for library in libraries])
    ax.set_ylabel("Throughput ratio")
    ax.set_title("GPU DataLoader / AlbumentationsX CPU", fontsize=10.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.75, which="both")
    ax.text(
        0.99,
        0.64,
        "parity",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    fig.suptitle("GPU micro does not predict DataLoader throughput", fontsize=11.5, y=1.02)
    _savefig(FIGURES / "gpu_vs_albumentationsx_cpu_ratios.png")
    return merged


def _dataloader_libraries(df: pd.DataFrame) -> pd.DataFrame:
    dataloader = df[df["regime"].isin(["rgb_dataloader_cpu", "rgb_dataloader_gpu"]) & df["measured"]]
    medians = (
        dataloader.groupby(["regime_label", "library"], observed=True)["median_throughput"]
        .median()
        .reset_index()
        .sort_values(["regime_label", "library"])
    )
    medians.to_csv(GENERATED / "figure_dataloader_library_medians.csv", index=False)

    plt.figure(figsize=(8.5, 4.8))
    labels = []
    values = []
    colors = []
    for row in medians.itertuples():
        labels.append(f"{row.regime_label}\n{row.library}")
        values.append(row.median_throughput)
        colors.append(PALETTE.get(str(row.library), "#777777"))
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(values)), labels, rotation=40, ha="right")
    plt.ylabel("Median row throughput (img/s)")
    plt.title("End-to-end DataLoader throughput")
    _savefig(FIGURES / "dataloader_library_medians.png")
    return medians


def _gpu_memory_plot(df: pd.DataFrame) -> pd.DataFrame:
    gpu_mem = df[(df["regime"] == "rgb_dataloader_gpu") & df["measured"] & df["gpu_peak_allocated_mb"].notna()].copy()
    gpu_mem["gpu_peak_allocated_gb"] = gpu_mem["gpu_peak_allocated_mb"] / 1024.0
    gpu_mem["gpu_peak_reserved_gb"] = gpu_mem["gpu_peak_reserved_mb"] / 1024.0
    gpu_mem.to_csv(GENERATED / "figure_gpu_memory_rows.csv", index=False)
    if gpu_mem.empty:
        return gpu_mem

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for library in ["torchvision", "kornia", "dali"]:
        subset = gpu_mem[gpu_mem["library"] == library]
        if subset.empty:
            continue
        ax.scatter(
            subset["median_throughput"],
            subset["gpu_peak_allocated_gb"],
            label=LIBRARY_DISPLAY.get(library, library),
            color=PALETTE.get(library, "#777777"),
            alpha=0.85,
            s=46,
            edgecolor="#222222",
            linewidth=0.35,
        )
    ax.set_xlabel("Throughput (img/s)")
    ax.set_ylabel("Peak allocated GPU memory (GB)")
    ax.set_title("GPU augmentation consumes training GPU memory")
    ax.grid(axis="both", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#cccccc",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    fig.text(
        0.5,
        -0.02,
        "The benchmark records allocated and reserved memory; this figure plots peak allocated memory in GB.",
        ha="center",
        fontsize=8.5,
    )
    _savefig(FIGURES / "gpu_memory_vs_throughput.png")
    return gpu_mem


def _abstract_claims_plot(
    df: pd.DataFrame,
    winners: pd.DataFrame,
    gpu_vs_alb: pd.DataFrame,
    dataloader_medians: pd.DataFrame,
    coverage: pd.DataFrame,
    dataloader_coverage: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    ax_winners, ax_cpu, ax_gpu, ax_coverage = axes.flatten()

    winner_counts = winners.groupby(["regime_label", "winner"], observed=True).size().reset_index(name="wins")
    winner_pivot = winner_counts.pivot(index="regime_label", columns="winner", values="wins").fillna(0)
    winner_pivot = winner_pivot.reindex(REGIME_ORDER).dropna(how="all")
    winner_pivot = winner_pivot[[lib for lib in LIBRARY_ORDER if lib in winner_pivot.columns]]
    bottom = [0.0] * len(winner_pivot)
    x = list(range(len(winner_pivot)))
    for library in winner_pivot.columns:
        values = winner_pivot[library].tolist()
        ax_winners.bar(x, values, bottom=bottom, label=library, color=PALETTE.get(str(library), "#777777"))
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    ax_winners.set_xticks(x, [str(index) for index in winner_pivot.index], rotation=20, ha="right")
    ax_winners.set_ylabel("Winner count")
    ax_winners.set_title("A. Winners change by regime")
    ax_winners.legend(frameon=False, fontsize=8, ncols=2)

    cpu = (
        dataloader_medians[dataloader_medians["regime_label"] == "CPU DataLoader"]
        .sort_values("median_throughput", ascending=False)
        .copy()
    )
    cpu_labels = cpu["library"].astype(str).tolist()
    cpu_values = cpu["median_throughput"].tolist()
    ax_cpu.bar(cpu_labels, cpu_values, color=[PALETTE.get(label, "#777777") for label in cpu_labels])
    ax_cpu.set_ylabel("Median throughput (img/s)")
    ax_cpu.set_title("B. CPU DataLoader median throughput")
    ax_cpu.tick_params(axis="x", rotation=20)
    _bar_label(ax_cpu)
    ax_cpu.text(
        0.02,
        0.92,
        "AlbumentationsX wins 56/57 CPU recipes",
        transform=ax_cpu.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.25"},
    )

    gpu_summary = (
        gpu_vs_alb.groupby("library", observed=True)["gpu_to_albumentationsx_cpu"]
        .agg(["count", "median", lambda s: int((s > 1.0).sum())])
        .rename(columns={"<lambda_0>": "wins"})
        .reset_index()
    )
    gpu_summary["library"] = gpu_summary["library"].astype(str)
    gpu_summary = gpu_summary.set_index("library").reindex(["torchvision", "kornia", "dali"]).dropna().reset_index()
    ax_gpu.bar(
        gpu_summary["library"],
        gpu_summary["median"],
        color=[PALETTE.get(str(library), "#777777") for library in gpu_summary["library"]],
    )
    ax_gpu.axhline(1.0, color="#222222", linewidth=1, linestyle="--")
    ax_gpu.set_ylim(0, max(1.1, float(gpu_summary["median"].max()) * 1.25))
    ax_gpu.set_ylabel("Median GPU / AlbumentationsX CPU")
    ax_gpu.set_title("C. GPU DataLoader rarely beats CPU baseline")
    for index, row in enumerate(gpu_summary.itertuples()):
        ax_gpu.text(
            index,
            row.median + 0.035,
            f"{int(row.wins)}/{int(row.count)} wins",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_gpu.tick_params(axis="x", rotation=20)

    gpu_cov = dataloader_coverage[dataloader_coverage["regime_label"] == "GPU DataLoader"].copy()
    gpu_cov["library"] = gpu_cov["library"].astype(str)
    gpu_cov = gpu_cov.set_index("library").reindex(["kornia", "torchvision", "dali"]).dropna().reset_index()
    x_cov = list(range(len(gpu_cov)))
    bottom_cov = [0] * len(gpu_cov)
    for column, color, label in [
        ("full", "#2c7a4b", "full"),
        ("early_stopped", "#d9a441", "early-stopped"),
        ("explicit_unsupported", "#b85c5c", "unsupported"),
        ("absent_or_not_in_result", "#bbbbbb", "absent"),
    ]:
        values = gpu_cov[column].tolist()
        ax_coverage.bar(x_cov, values, bottom=bottom_cov, color=color, label=label)
        bottom_cov = [a + b for a, b in zip(bottom_cov, values, strict=True)]
    ax_coverage.set_xticks(x_cov, gpu_cov["library"].tolist(), rotation=20, ha="right")
    ax_coverage.set_ylabel("Recipe rows")
    ax_coverage.set_title("D. GPU coverage breadth varies")
    for index, row in enumerate(gpu_cov.itertuples()):
        ax_coverage.text(index, row.universe_rows + 1, f"{row.full}/{row.universe_rows}", ha="center", fontsize=8)
    ax_coverage.legend(frameon=False, fontsize=8, ncols=3)

    fig.suptitle("Abstract claims are regime- and coverage-dependent", fontsize=14, y=1.01)
    _savefig(FIGURES / "abstract_claims.png")


def _write_insights(
    df: pd.DataFrame,
    winners: pd.DataFrame,
    gpu_vs_alb: pd.DataFrame,
    dataloader_medians: pd.DataFrame,
    coverage: pd.DataFrame,
    dataloader_coverage: pd.DataFrame,
) -> None:
    gpu_vs_alb_summary = (
        gpu_vs_alb.groupby("library", observed=True)["gpu_to_albumentationsx_cpu"]
        .agg(["count", "median", lambda s: int((s > 1.0).sum())])
        .rename(columns={"<lambda_0>": "wins_over_alb_cpu"})
        .reset_index()
    )
    cpu_pipeline = dataloader_medians[dataloader_medians["regime_label"] == "CPU DataLoader"]
    gpu_pipeline = dataloader_medians[dataloader_medians["regime_label"] == "GPU DataLoader"]

    gpu_winners = winners[winners["regime_label"] == "GPU DataLoader"].copy()
    gpu_winner_counts = gpu_winners.groupby("winner", observed=True).size().sort_values(ascending=False)
    gpu_rows = df[df["regime"] == "rgb_dataloader_gpu"]
    kornia_measured = set(
        gpu_rows[(gpu_rows["library"] == "kornia") & gpu_rows["measured"]]["transform"].astype(str),
    )
    dali_measured = set(
        gpu_rows[(gpu_rows["library"] == "dali") & gpu_rows["measured"]]["transform"].astype(str),
    )
    kornia_measured_dali_not = len(kornia_measured - dali_measured)

    coverage_lookup = {(str(row.regime_label), str(row.library)): row for row in dataloader_coverage.itertuples()}
    alb_cpu_cov = coverage_lookup[("CPU DataLoader", "albumentationsx")]
    kornia_cpu_cov = coverage_lookup[("CPU DataLoader", "kornia")]
    tv_cpu_cov = coverage_lookup[("CPU DataLoader", "torchvision")]
    pillow_cpu_cov = coverage_lookup[("CPU DataLoader", "pillow")]
    kornia_gpu_cov = coverage_lookup[("GPU DataLoader", "kornia")]
    tv_gpu_cov = coverage_lookup[("GPU DataLoader", "torchvision")]
    dali_gpu_cov = coverage_lookup[("GPU DataLoader", "dali")]
    open_leaderboard = pd.read_csv(GENERATED / "open_dataloader_leaderboard.csv")
    open_winners = pd.read_csv(GENERATED / "open_dataloader_winners.csv")
    alb_open = open_leaderboard[open_leaderboard["implementation"] == "AlbumentationsX CPU"].iloc[0]
    open_exceptions = open_winners[open_winners["implementation"] != "AlbumentationsX CPU"]
    open_exception_text = "; ".join(
        (f"{implementation} wins " + ", ".join(group["display_transform"].astype(str).tolist()))
        for implementation, group in open_exceptions.groupby("implementation", sort=True)
    )

    lines = [
        "# Paper Insights",
        "",
        "## Main Claims Supported By Current RGB Data",
        "",
        "- CPU DataLoader throughput is the strongest AlbumentationsX result: it wins 56 of 57 comparable CPU pipeline rows, with median measured row throughput around 4.6k img/s.",
        f"- In the open production DataLoader category, AlbumentationsX CPU wins {int(alb_open.wins)} of {int(alb_open.universe_rows)} recipes across CPU and GPU DataLoader implementations, with the highest median measured-row throughput ({float(alb_open.median_throughput):.1f} img/s).",
        "- GPU micro speed does not predict end-to-end GPU DataLoader speed. TorchVision wins most GPU micro rows, but its GPU DataLoader median is below its CPU DataLoader median.",
        "- Coverage breadth is a first-class result: CPU DataLoader coverage is "
        f"AlbumentationsX {alb_cpu_cov.full}/{alb_cpu_cov.universe_rows}, Kornia {kornia_cpu_cov.full}/{kornia_cpu_cov.universe_rows}, "
        f"TorchVision {tv_cpu_cov.full}/{tv_cpu_cov.universe_rows}, and Pillow {pillow_cpu_cov.full}/{pillow_cpu_cov.universe_rows}; "
        f"GPU DataLoader coverage is Kornia {kornia_gpu_cov.full}/{kornia_gpu_cov.universe_rows}, "
        f"TorchVision {tv_gpu_cov.full}/{tv_gpu_cov.universe_rows}, and DALI {dali_gpu_cov.full}/{dali_gpu_cov.universe_rows}.",
        f"- Kornia GPU measures {kornia_measured_dali_not} recipes that DALI does not measure. Broad GPU support can lower a measured-row throughput summary because it includes harder recipes; missing rows are coverage gaps, not zero-throughput speed failures.",
        "- DALI remains a useful specialized native graph backend on supported recipes, while AlbumentationsX remains the CPU DataLoader winner under the 57-recipe protocol.",
        "",
        "## Figure Recommendations For NeurIPS",
        "",
        "1. Use a native LaTeX table, not a PNG figure, for the benchmarking pain points and protocol guardrails.",
        "2. Use `docs/paper_figures/open_dataloader_leaderboard.png` as the headline empirical result: it shows the open production DataLoader category directly.",
        "3. Use `docs/paper_figures/coverage_vs_throughput.png` as the coverage figure: it separates measured-row throughput from coverage over the same 57-recipe universe and includes the Elastic implementation drill-down.",
        "4. Use `docs/paper_figures/gpu_vs_albumentationsx_cpu_ratios.png` to support the paired claim that GPU DataLoader pipelines rarely beat the AlbumentationsX CPU baseline.",
        "5. Use `docs/paper_figures/gpu_memory_vs_throughput.png` in the main paper because GPU augmentation consumes memory that could otherwise be used by training.",
        "",
        "## Data Tables To Put In The Paper",
        "",
        "- Main text figure/table source: `docs/paper_data/open_dataloader_leaderboard.csv` / `figure_open_dataloader_leaderboard.csv` for the open production DataLoader category.",
        "- Main text figure source: `docs/paper_data/figure_elastic_row.csv` for the Elastic implementation drill-down.",
        "- Main text table: `docs/paper_data/summary.md` sections `Coverage Summary`, `AlbumentationsX CPU DataLoader vs GPU DataLoader`, and `GPU Memory`.",
        "- Appendix table: `docs/paper_data/production_support_matrix.md` for the 57-row production DataLoader support/performance matrix.",
        "- Supporting pivot tables: `docs/paper_data/*_pivot.csv` for each generated RGB and 9-channel regime.",
        "- Reproducibility table: `docs/paper_data/all_results.csv` with source file provenance.",
        "",
        "## Current Data Summary",
        "",
        "### DataLoader median throughput by library",
        "",
        "| Regime | Library | Median row throughput img/s |",
        "|---|---|---:|",
    ]
    for row in pd.concat([cpu_pipeline, gpu_pipeline]).itertuples():
        lines.append(f"| {row.regime_label} | {row.library} | {row.median_throughput:.1f} |")

    lines.extend(
        [
            "",
            "### Open production DataLoader category",
            "",
            "| Implementation | Regime | Full measured / 57 | Median measured-row throughput img/s | Open wins |",
            "|---|---|---:|---:|---:|",
        ],
    )
    for row in open_leaderboard.itertuples(index=False):
        lines.append(
            f"| {row.implementation} | {row.regime_label} | {int(row.full)}/{int(row.universe_rows)} | "
            f"{float(row.median_throughput):.1f} | {int(row.wins)} |",
        )
    if open_exception_text:
        lines.append("")
        lines.append(f"Open-category exception rows: {open_exception_text}.")

    lines.extend(
        [
            "",
            "### DataLoader coverage versus throughput",
            "",
            "| Regime | Library | Full measured / 57 | Explicit unsupported | Absent/not in result | Median measured-row throughput img/s |",
            "|---|---|---:|---:|---:|---:|",
        ],
    )
    for row in dataloader_coverage.itertuples():
        lines.append(
            f"| {row.regime_label} | {row.library} | {row.full}/{row.universe_rows} | "
            f"{row.explicit_unsupported} | {row.absent_or_not_in_result} | {row.median_throughput:.1f} |",
        )

    lines.extend(
        [
            "",
            "### GPU versus AlbumentationsX CPU DataLoader",
            "",
            "| GPU library | Compared rows | Wins over AlbumentationsX CPU | Median ratio |",
            "|---|---:|---:|---:|",
        ],
    )
    for row in gpu_vs_alb_summary.itertuples():
        lines.append(f"| {row.library} | {int(row.count)} | {int(row.wins_over_alb_cpu)} | {fmt_ratio(row.median)} |")

    lines.extend(
        [
            "",
            "### GPU DataLoader winner counts",
            "",
            "| Library | Wins |",
            "|---|---:|",
        ],
    )
    for library, count in gpu_winner_counts.items():
        lines.append(f"| {library} | {int(count)} |")

    lines.extend(
        [
            "",
            "### Coverage",
            "",
            "| Regime | Library | Full | Early-stopped | Unsupported |",
            "|---|---|---:|---:|---:|",
        ],
    )
    for row in coverage.itertuples():
        lines.append(f"| {row.regime_label} | {row.library} | {row.full} | {row.early} | {row.unsupported} |")

    lines.extend(
        [
            "",
            "## Unexpected Or Paper-Worthy Observations",
            "",
            "- TorchVision GPU micro results look excellent, but the production path has to apply per-sample random transforms in a Python loop; the DataLoader result is therefore much weaker than the micro result.",
            "- The open production DataLoader category is unexpectedly CPU-led: AlbumentationsX CPU beats the GPU implementations on most recipes once the benchmark includes DataLoader work, transfer/synchronization scope, random semantics, and coverage.",
            "- Coverage breadth changes summary statistics. Kornia GPU measures many recipes that DALI does not, so a lower Kornia median can reflect both implementation speed and the inclusion of harder rows.",
            "- The GPU DataLoader story is not `GPU wins`: it is `GPU can be strong when the pipeline, transform support, batching semantics, and graph/runtime fit the recipe; otherwise CPU DataLoader can be faster and simpler`.",
            "- Memory is a legitimate axis: DALI uses roughly 3 GB in this run, while Kornia can reserve substantially more on heavy GPU rows and also hits unsupported/early-stop cases.",
            "",
            "## What To Avoid Claiming",
            "",
            "- Do not treat missing or unsupported rows as zero throughput.",
            "- Do not claim any GPU backend has full recipe parity unless the coverage table shows it for that regime.",
            "- Do not compare DALI micro performance; this benchmark has only DALI pipeline measurements.",
            "- Do not claim GPU augmentation is slower in principle. The supported DALI rows show that GPU pipelines can be fast when implemented as a native graph.",
            "- Do not mix 9-channel or video results into the main NeurIPS story unless the paper explicitly scopes them as separate preliminary appendices.",
            "",
        ],
    )
    (GENERATED / "insights.md").write_text("\n".join(lines), encoding="utf-8")


def _patch_between_markers(path: Path, start: str, end: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    start_index = text.find(start)
    end_index = text.find(end)
    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise RuntimeError(f"Could not find marker block {start!r} / {end!r} in {path}")
    patched = text[: start_index + len(start)] + "\n\n" + replacement.strip() + "\n\n" + text[end_index:]
    path.write_text(patched, encoding="utf-8")


def _figure_markdown(figures: list[dict[str, str]], prefix: str) -> str:
    blocks = []
    for figure in figures:
        image_path = f"{prefix}{figure['path']}"
        blocks.extend(
            [
                f"### {figure['title']}",
                "",
                f"![{figure['title']}]({image_path})",
                "",
                figure["caption"],
            ],
        )
        blocks.append("")
    return "\n".join(blocks).strip()


README_SCENARIO_TABLES = [
    (
        "RGB benchmark table",
        "rgb_",
        ["rgb_micro_cpu", "rgb_dataloader_cpu", "rgb_micro_gpu", "rgb_dataloader_gpu"],
    ),
    (
        "9-channel benchmark table",
        "image9ch_",
        ["image9ch_micro_cpu", "image9ch_dataloader_cpu", "image9ch_micro_gpu", "image9ch_dataloader_gpu"],
    ),
    (
        "Video benchmark table",
        "video16f_",
        ["video16f_micro_cpu", "video16f_dataloader_cpu", "video16f_micro_gpu", "video16f_dataloader_gpu"],
    ),
]


def _readme_transform_tables_markdown() -> str:
    df = pd.read_csv(GENERATED / "all_results.csv")
    blocks = [
        "### Scenario benchmark tables",
        "",
        "Rows are transforms. Columns are benchmark regimes. Each cell shows the fastest full measured implementation for that transform and regime, formatted as `Library throughput`; `-` means no full measured row is available.",
    ]
    for title, regime_prefix, regimes in README_SCENARIO_TABLES:
        subset = df[df["regime"].astype(str).str.startswith(regime_prefix)].copy()
        blocks.extend(["", f"#### {title}", ""])
        if subset.empty:
            blocks.append("No published snapshots are available for this scenario yet.")
            continue
        subset["full"] = (
            subset["supported"].astype(bool)
            & ~subset["early_stopped"].astype(bool)
            & (subset["num_successful_runs"].astype(int) > 0)
        )
        subset["display_transform"] = subset["transform"].astype(str).map(recipe_display_name)
        headers = [
            "Transform",
            *[
                str(subset.loc[subset["regime"] == regime, "regime_label"].iloc[0])
                for regime in regimes
                if (subset["regime"] == regime).any()
            ],
        ]
        active_regimes = [regime for regime in regimes if (subset["regime"] == regime).any()]
        blocks.append("| " + " | ".join(headers) + " |")
        blocks.append("| " + " | ".join(["---", *["---:" for _ in active_regimes]]) + " |")
        for transform in sorted(subset["display_transform"].unique()):
            cells = [transform]
            for regime in active_regimes:
                candidates = subset[(subset["regime"] == regime) & (subset["display_transform"] == transform)]
                measured = candidates[candidates["full"]].sort_values("median_throughput", ascending=False)
                if measured.empty:
                    cells.append("-")
                    continue
                best = measured.iloc[0]
                library = LIBRARY_DISPLAY.get(str(best["library"]), str(best["library"]))
                cells.append(f"{library} {float(best['median_throughput']):.1f}")
            blocks.append("| " + " | ".join(cells) + " |")
    return "\n".join(blocks)


def _write_figure_markdown() -> None:
    readme_block = "\n".join(
        [
            "The paper figures below are generated from the checked-in data under `docs/paper_data/`.",
            "",
            _figure_markdown(MAIN_FIGURES, "docs/paper_figures/"),
            "",
            _figure_markdown(APPENDIX_FIGURES, "docs/paper_figures/"),
            "",
            _readme_transform_tables_markdown(),
        ],
    )
    _patch_between_markers(
        README,
        "<!-- PAPER_FIGURES_START -->",
        "<!-- PAPER_FIGURES_END -->",
        readme_block,
    )

    if DRAFT.exists():
        draft_block = "\n".join(
            [
                "The figures in this section are generated from the same row-level CSV as the tables above.",
                "",
                _figure_markdown(MAIN_FIGURES, "figures/"),
                "",
                _figure_markdown(APPENDIX_FIGURES, "figures/"),
            ],
        )
        _patch_between_markers(
            DRAFT,
            "<!-- GENERATED_FIGURES_START -->",
            "<!-- GENERATED_FIGURES_END -->",
            draft_block,
        )


def _write_support_matrix_markdown() -> None:
    if not DRAFT.exists():
        return
    matrix_md = (GENERATED / "production_support_matrix.md").read_text(encoding="utf-8")
    _patch_between_markers(
        DRAFT,
        "<!-- GENERATED_PRODUCTION_SUPPORT_MATRIX_START -->",
        "<!-- GENERATED_PRODUCTION_SUPPORT_MATRIX_END -->",
        matrix_md,
    )


def _sync_public_figures() -> None:
    PUBLIC_FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for figure in [*MAIN_FIGURES, *APPENDIX_FIGURES]:
        source = FIGURES / figure["path"]
        _copy_if_different(source, PUBLIC_FIGURES / source.name)
        _copy_if_different(source, PAPER_FIGURES / source.name)
        pdf_source = source.with_suffix(".pdf")
        if pdf_source.exists():
            _copy_if_different(pdf_source, PUBLIC_FIGURES / pdf_source.name)
            _copy_if_different(pdf_source, PAPER_FIGURES / pdf_source.name)


def _copy_if_different(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _sync_public_data() -> None:
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    names = [
        "all_results.csv",
        "summary.md",
        "summary.json",
        "open_dataloader_leaderboard.csv",
        "open_dataloader_winners.csv",
        "figure_open_dataloader_leaderboard.csv",
        "figure_coverage_vs_throughput.csv",
        "figure_elastic_row.csv",
        "figure_gpu_vs_albumentationsx_cpu.csv",
        "figure_gpu_memory_rows.csv",
        "figure_winner_counts.csv",
        "figure_winner_rows.csv",
        "production_support_matrix.csv",
        "production_support_matrix.md",
        "unsupported_and_early_stopped.csv",
        "unsupported_and_early_stopped.md",
    ]
    names.extend(path.name for path in GENERATED.glob("*_pivot.csv"))
    names.extend(path.name for path in GENERATED.glob("*_cpu.csv"))
    names.extend(path.name for path in GENERATED.glob("*_gpu.csv"))
    for name in names:
        source = GENERATED / name
        if source.exists():
            _copy_if_different(source, PUBLIC_DATA / name)


def _remove_stale_generated_files() -> None:
    for path in [
        FIGURES / "benchmarking_pitfalls.png",
        FIGURES / "benchmarking_pitfalls.pdf",
        FIGURES / "gpu_vs_albumentationsx_cpu_boxplot.png",
        FIGURES / "gpu_vs_albumentationsx_cpu_boxplot.pdf",
        FIGURES / "abstract_claims.png",
        FIGURES / "abstract_claims.pdf",
        PUBLIC_FIGURES / "benchmarking_pitfalls.png",
        PUBLIC_FIGURES / "benchmarking_pitfalls.pdf",
        PUBLIC_FIGURES / "gpu_vs_albumentationsx_cpu_boxplot.png",
        PUBLIC_FIGURES / "gpu_vs_albumentationsx_cpu_boxplot.pdf",
        PUBLIC_FIGURES / "abstract_claims.png",
        PUBLIC_FIGURES / "abstract_claims.pdf",
        PUBLIC_FIGURES / "coverage_by_regime.png",
        PUBLIC_FIGURES / "coverage_by_regime.pdf",
        PAPER_FIGURES / "benchmarking_pitfalls.png",
        PAPER_FIGURES / "benchmarking_pitfalls.pdf",
        PAPER_FIGURES / "gpu_vs_albumentationsx_cpu_boxplot.png",
        PAPER_FIGURES / "gpu_vs_albumentationsx_cpu_boxplot.pdf",
        PAPER_FIGURES / "abstract_claims.png",
        PAPER_FIGURES / "abstract_claims.pdf",
        PAPER_FIGURES / "coverage_by_regime.png",
        PAPER_FIGURES / "coverage_by_regime.pdf",
        GENERATED / "figure_benchmarking_pitfalls.csv",
    ]:
        path.unlink(missing_ok=True)


def main() -> None:
    global DRAFT, FIGURES, GENERATED, PAPER_DIR, PAPER_FIGURES, PUBLIC_DATA, PUBLIC_FIGURES, README

    parser = argparse.ArgumentParser(description="Regenerate paper figures and derived insight files.")
    parser.add_argument("--data", type=Path, default=GENERATED, help="Directory containing/generated paper CSV data.")
    parser.add_argument("--output", type=Path, default=FIGURES, help="Directory for primary generated figures.")
    parser.add_argument("--public-data", type=Path, default=PUBLIC_DATA, help="Directory to sync public data CSVs.")
    parser.add_argument("--public-figures", type=Path, default=PUBLIC_FIGURES, help="Directory to sync public figures.")
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=PAPER_DIR,
        help="Paper directory for synced LaTeX tables/figures.",
    )
    parser.add_argument("--readme", type=Path, default=README, help="README path to patch.")
    parser.add_argument("--draft", type=Path, default=DRAFT, help="Optional draft markdown path to patch.")
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Use existing CSVs in --data instead of regenerating from raw JSON artifacts.",
    )
    parser.add_argument(
        "--extra-run-dir",
        action="append",
        default=[],
        metavar="REGIME=PATH",
        help="Forwarded to generate_paper_data.py when data generation is enabled.",
    )
    args = parser.parse_args()

    GENERATED = _resolve_path(args.data)
    FIGURES = _resolve_path(args.output)
    PUBLIC_DATA = _resolve_path(args.public_data)
    PUBLIC_FIGURES = _resolve_path(args.public_figures)
    PAPER_DIR = _resolve_path(args.paper_dir)
    PAPER_FIGURES = PAPER_DIR / "figures"
    README = _resolve_path(args.readme)
    DRAFT = _resolve_path(args.draft)

    if not args.skip_data_generation:
        forwarded = [item for value in args.extra_run_dir for item in ("--extra-run-dir", value)]
        _run_data_generator(forwarded)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = _load_results()
    _remove_stale_generated_files()
    _open_dataloader_leaderboard_plot()
    coverage = _coverage_plot(df)
    dataloader_coverage = _dataloader_coverage_vs_throughput(df)
    _production_support_matrix(df)
    winners = _winner_counts(df)
    gpu_vs_alb = _albumentations_vs_gpu(df)
    dataloader_medians = _dataloader_libraries(df)
    _gpu_memory_plot(df)
    _sync_public_figures()
    _write_insights(df, winners, gpu_vs_alb, dataloader_medians, coverage, dataloader_coverage)
    _sync_public_data()
    _write_figure_markdown()
    _write_support_matrix_markdown()
    print(f"Wrote figures to {FIGURES}")
    print(f"Wrote public figures to {PUBLIC_FIGURES}")
    print(f"Wrote public data to {PUBLIC_DATA}")
    print(f"Wrote paper figures to {PAPER_FIGURES}")
    print(f"Wrote insights to {GENERATED / 'insights.md'}")


if __name__ == "__main__":
    main()
