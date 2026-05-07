# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "docs" / "paper_data"
DEFAULT_OUTPUT = ROOT / "docs" / "paper_figures"

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/benchmark-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/benchmark-cache")

import matplotlib.pyplot as plt

PALETTE = {
    "albumentationsx": "#177245",
    "torchvision": "#2f6fbd",
    "kornia": "#8a4fb5",
    "pillow": "#c47a1b",
    "dali": "#5f6b2f",
}
MARKERS = {
    "albumentationsx": "o",
    "torchvision": "s",
    "kornia": "^",
    "pillow": "v",
    "dali": "P",
}
LIBRARY_DISPLAY = {
    "albumentationsx": "AlbumentationsX",
    "torchvision": "TorchVision",
    "kornia": "Kornia",
    "pillow": "Pillow",
    "dali": "DALI",
}
LIBRARY_ORDER = ["albumentationsx", "torchvision", "kornia", "pillow", "dali"]
DATALOADER_FACETS = {
    "CPU DataLoader": ["albumentationsx", "kornia", "torchvision", "pillow"],
    "GPU DataLoader": ["kornia", "torchvision", "dali"],
}


def _savefig(output_dir: Path, name: str, *, tight: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    png_path = output_dir / name
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _require(path: Path) -> Path:
    if not path.exists():
        msg = f"Missing required paper data file: {path}"
        raise FileNotFoundError(msg)
    return path


def _open_dataloader_leaderboard_plot(data_dir: Path, output_dir: Path) -> None:
    leaderboard = pd.read_csv(_require(data_dir / "open_dataloader_leaderboard.csv"))
    leaderboard["library"] = leaderboard["library"].astype(str)
    plot_df = leaderboard.sort_values("median_throughput", ascending=True).reset_index(drop=True)
    plot_df.to_csv(data_dir / "figure_open_dataloader_leaderboard.csv", index=False)

    _, ax = plt.subplots(figsize=(7.2, 4.2))
    y_positions = list(range(len(plot_df)))
    colors = [PALETTE.get(str(row.library), "#777777") for row in plot_df.itertuples()]
    values = [float(row.median_throughput) for row in plot_df.itertuples()]
    labels = [str(row.implementation) for row in plot_df.itertuples()]
    ax.barh(y_positions, values, color=colors, height=0.62)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Median measured-recipe throughput (img/s)")
    ax.set_title("Open production DataLoader category: CPU and GPU pipelines compete together")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.75)
    ax.set_xlim(0, max(values) * 1.36)
    for y, row in zip(y_positions, plot_df.itertuples(), strict=True):
        label = (
            f"{float(row.median_throughput):.0f} img/s  |  "
            f"{int(row.full)}/{int(row.universe_rows)}  |  rank-1 {int(row.wins)}"
        )
        ax.text(
            float(row.median_throughput) + max(values) * 0.02,
            y,
            label,
            va="center",
            ha="left",
            fontsize=7.7,
        )
    _savefig(output_dir, "open_dataloader_leaderboard.png")


def _coverage_vs_throughput_plot(data_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(_require(data_dir / "figure_coverage_vs_throughput.csv"))
    elastic = pd.read_csv(_require(data_dir / "figure_elastic_row.csv"))

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
            ax.scatter(
                row["full"],
                row["median_throughput"],
                s=95,
                marker=MARKERS.get(library, "o"),
                color=PALETTE.get(library, "#777777"),
                edgecolor="#222222",
                linewidth=0.6,
                zorder=3,
            )
            dx, dy = label_offsets.get((regime_label, library), (8, 8))
            ax.annotate(
                f"{LIBRARY_DISPLAY.get(library, library)}\n{int(row['full'])}/{int(row['universe_rows'])}",
                xy=(row["full"], row["median_throughput"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.0,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
            )
        ax.set_title(regime_label)
        ax.set_xlabel("Supported and measured recipes (of 57)")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 5100)
        ax.grid(axis="both", color="#dddddd", linewidth=0.6, alpha=0.7)

    elastic_lookup = {(str(row.regime_label), str(row.library)): row for row in elastic.itertuples()}
    elastic_line = (
        "Elastic production recipe: "
        f"AlbX CPU {elastic_lookup[('CPU DataLoader', 'albumentationsx')].median_throughput:.0f} img/s; "
        f"Kornia GPU {elastic_lookup[('GPU DataLoader', 'kornia')].median_throughput:.0f}; "
        f"TorchVision GPU {elastic_lookup[('GPU DataLoader', 'torchvision')].median_throughput:.0f}; "
        "DALI not supported."
    )
    axes[0].set_ylabel("Median measured-recipe throughput (img/s)")
    fig.suptitle("Supported recipes and measured throughput are separate axes", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        elastic_line,
        ha="center",
        fontsize=8.3,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.28"},
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.20, wspace=0.10)
    _savefig(output_dir, "coverage_vs_throughput.png", tight=False)


def _gpu_vs_cpu_plot(data_dir: Path, output_dir: Path) -> None:
    merged = pd.read_csv(_require(data_dir / "figure_gpu_vs_albumentationsx_cpu.csv"))
    df = pd.read_csv(_require(data_dir / "all_results.csv"))
    df["supported"] = df["supported"].astype(bool)
    df["early_stopped"] = df["early_stopped"].astype(bool)
    df["measured"] = df["supported"] & ~df["early_stopped"] & (df["num_successful_runs"] > 0)

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
        figsize=(7.6, 3.9),
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
    ax_micro.set_title("GPU micro\nrank-1")
    ax_micro.set_ylabel("Rank-one count")
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
            marker=MARKERS.get(library, "o"),
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
            3.35,
            f"{wins}/{len(subset)} above parity",
            ha="center",
            va="top",
            fontsize=8.4,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "boxstyle": "round,pad=0.16"},
        )
        ax.text(x_position, 0.035, f"median {median:.2f}x", ha="center", va="bottom", fontsize=8.4, color="#444444")

    ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 4.0)
    ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_yticklabels(["0.05x", "0.10x", "0.25x", "0.50x", "1.00x", "2.00x", "4.00x"])
    ax.set_xticks(x_positions, [LIBRARY_DISPLAY[library] for library in libraries])
    ax.set_ylabel("Throughput ratio")
    ax.set_title("GPU DataLoader / paired AlbumentationsX CPU", fontsize=10.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.75, which="both")
    ax.text(0.99, 0.64, "parity", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#333333")
    fig.suptitle("GPU micro does not predict DataLoader throughput", fontsize=11.5, y=0.97)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.20, wspace=0.30)
    _savefig(output_dir, "gpu_vs_albumentationsx_cpu_ratios.png", tight=False)


def _gpu_memory_plot(data_dir: Path, output_dir: Path) -> None:
    gpu_mem = pd.read_csv(_require(data_dir / "figure_gpu_memory_rows.csv"))
    if gpu_mem.empty:
        return
    if "gpu_peak_allocated_gb" not in gpu_mem.columns:
        gpu_mem["gpu_peak_allocated_gb"] = gpu_mem["gpu_peak_allocated_mb"] / 1024.0
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for library in ["torchvision", "kornia", "dali"]:
        subset = gpu_mem[gpu_mem["library"] == library]
        if subset.empty:
            continue
        ax.scatter(
            subset["median_throughput"],
            subset["gpu_peak_allocated_gb"],
            label=LIBRARY_DISPLAY.get(library, library),
            marker=MARKERS.get(library, "o"),
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
        "Allocated memory is plotted; reserved memory is retained in the source CSV for production measurements.",
        ha="center",
        fontsize=8.5,
    )
    _savefig(output_dir, "gpu_memory_vs_throughput.png")


def _winner_counts_plot(data_dir: Path, output_dir: Path) -> None:
    counts = pd.read_csv(_require(data_dir / "figure_winner_counts.csv"))
    pivot = counts.pivot_table(index="regime_label", columns="winner", values="wins").fillna(0)
    regime_order = ["CPU micro", "CPU DataLoader", "GPU micro", "GPU DataLoader"]
    pivot = pivot.reindex([label for label in regime_order if label in pivot.index])
    pivot = pivot[[library for library in LIBRARY_ORDER if library in pivot.columns]]
    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 4.5),
        color=[PALETTE.get(col, "#777777") for col in pivot.columns],
    )
    ax.set_ylabel("Rank-one recipe count")
    ax.set_xlabel("")
    ax.set_title("Rank-one counts among comparable measured transforms")
    ax.legend(frameon=False, ncols=3)
    plt.xticks(rotation=20, ha="right")
    _savefig(output_dir, "winner_counts.png")


def _remove_stale(output_dir: Path) -> None:
    for name in [
        "abstract_claims.png",
        "abstract_claims.pdf",
        "benchmarking_pitfalls.png",
        "benchmarking_pitfalls.pdf",
        "coverage_by_regime.png",
        "coverage_by_regime.pdf",
        "gpu_vs_albumentationsx_cpu_boxplot.png",
        "gpu_vs_albumentationsx_cpu_boxplot.pdf",
    ]:
        (output_dir / name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate public paper figures from docs/paper_data CSVs.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_dir = args.data if args.data.is_absolute() else ROOT / args.data
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    _remove_stale(output_dir)
    _open_dataloader_leaderboard_plot(data_dir, output_dir)
    _coverage_vs_throughput_plot(data_dir, output_dir)
    _gpu_vs_cpu_plot(data_dir, output_dir)
    _gpu_memory_plot(data_dir, output_dir)
    _winner_counts_plot(data_dir, output_dir)
    sys.stdout.write(f"Wrote paper figures to {output_dir}\n")


if __name__ == "__main__":
    main()
