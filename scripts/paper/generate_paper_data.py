from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "docs" / "paper_raw"
DEFAULT_OUTPUT = ROOT / "docs" / "paper_data"

REGIME_LABELS = {
    "rgb_micro_cpu": "CPU micro",
    "rgb_micro_gpu": "GPU micro",
    "rgb_dataloader_cpu": "CPU DataLoader",
    "rgb_dataloader_gpu": "GPU DataLoader",
}
LIBRARY_DISPLAY = {
    "albumentationsx": "AlbumentationsX",
    "torchvision": "TorchVision",
    "kornia": "Kornia",
    "pillow": "Pillow",
    "dali": "DALI",
}
LIBRARY_ORDER = ["albumentationsx", "torchvision", "kornia", "pillow", "dali"]
ALL_RESULTS_COLUMNS = [
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
]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path} does not contain a JSON object"
        raise TypeError(msg)
    return payload


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _result_throughputs(result: dict[str, Any]) -> list[float]:
    raw = result.get("throughputs")
    if not isinstance(raw, list):
        return []
    return [value for item in raw if (value := _float_or_none(item)) is not None]


def _result_reason(result: dict[str, Any]) -> str:
    for key in ("reason", "unsupported_reason", "early_stop_reason", "error"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _gpu_memory_mb(result: dict[str, Any], key: str) -> float | None:
    gpu_memory = result.get("gpu_memory")
    if not isinstance(gpu_memory, dict):
        return None
    bytes_value = _float_or_none(gpu_memory.get(key))
    if bytes_value is None:
        return None
    return bytes_value / (1024 * 1024)


def _raw_rows(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*/*_results.json")):
        regime = path.parent.name
        payload = _read_json(path)
        metadata = payload.get("metadata", {})
        results = payload.get("results", {})
        if regime not in REGIME_LABELS:
            msg = f"Unknown paper raw regime directory: {path.parent}"
            raise ValueError(msg)
        if not isinstance(metadata, dict) or not isinstance(results, dict):
            msg = f"{path} is not a benchmark result JSON"
            raise TypeError(msg)
        library = str(metadata.get("library") or path.name.split("_", maxsplit=1)[0])
        for transform, raw_result in results.items():
            if not isinstance(raw_result, dict):
                continue
            status = str(raw_result.get("status") or "ok")
            supported = bool(raw_result.get("supported", status != "unsupported"))
            early_stopped = bool(raw_result.get("early_stopped", status == "early_stopped"))
            throughputs = _result_throughputs(raw_result)
            rows.append(
                {
                    "regime": regime,
                    "regime_label": REGIME_LABELS[regime],
                    "library": library,
                    "transform": str(transform),
                    "status": status,
                    "supported": supported,
                    "early_stopped": early_stopped,
                    "throughputs": throughputs,
                    "fallback_median": _float_or_none(raw_result.get("median_throughput")),
                    "gpu_peak_allocated_mb": _gpu_memory_mb(raw_result, "peak_allocated_bytes"),
                    "gpu_peak_reserved_mb": _gpu_memory_mb(raw_result, "peak_reserved_bytes"),
                    "reason": _result_reason(raw_result),
                },
            )
    if not rows:
        msg = f"No raw result JSONs found under {raw_dir}"
        raise FileNotFoundError(msg)
    return pd.DataFrame(rows)


def _ci95(values: list[float], std: float) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(len(values))


def _aggregate_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["regime", "regime_label", "library", "transform"]
    for group_key, group in raw_df.groupby(group_cols, sort=True):
        regime, regime_label, library, transform = group_key
        throughputs = [value for values in group["throughputs"] for value in values]
        supported = bool(group["supported"].any())
        early_stopped = bool(group["early_stopped"].any()) and not throughputs
        if throughputs:
            median = statistics.median(throughputs)
            mean = statistics.fmean(throughputs)
            std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
            status = "ok"
        else:
            fallback_values = [value for value in group["fallback_median"].tolist() if value is not None]
            median = fallback_values[0] if fallback_values else 0.0
            mean = median
            std = 0.0
            if early_stopped:
                status = "early_stopped"
            elif not supported:
                status = "unsupported"
            else:
                status = str(group["status"].iloc[0])
        gpu_allocated_values = [value for value in group["gpu_peak_allocated_mb"].tolist() if value is not None]
        gpu_reserved_values = [value for value in group["gpu_peak_reserved_mb"].tolist() if value is not None]
        reasons = [reason for reason in group["reason"].tolist() if reason]
        rows.append(
            {
                "regime": regime,
                "regime_label": regime_label,
                "library": library,
                "transform": transform,
                "status": status,
                "supported": supported,
                "early_stopped": early_stopped,
                "num_successful_runs": len(throughputs),
                "median_throughput": median,
                "mean_throughput": mean,
                "std_throughput": std,
                "cv_throughput": std / mean if mean else 0.0,
                "ci95": _ci95(throughputs, std),
                "gpu_peak_allocated_mb": max(gpu_allocated_values) if gpu_allocated_values else None,
                "gpu_peak_reserved_mb": max(gpu_reserved_values) if gpu_reserved_values else None,
                "reason": "; ".join(dict.fromkeys(reasons)),
            },
        )
    return pd.DataFrame(rows, columns=ALL_RESULTS_COLUMNS).sort_values(["regime", "library", "transform"])


def _measured(df: pd.DataFrame) -> pd.Series:
    return df["supported"].astype(bool) & ~df["early_stopped"].astype(bool) & (df["num_successful_runs"] > 0)


def _winner_rows(df: pd.DataFrame) -> pd.DataFrame:
    measured = df[_measured(df)].copy()
    rows = []
    for (regime, transform), group in measured.groupby(["regime", "transform"], sort=True):
        if len(group) < 2:
            continue
        winner = group.sort_values("median_throughput", ascending=False).iloc[0]
        rows.append(
            {
                "regime": regime,
                "regime_label": winner["regime_label"],
                "transform": transform,
                "winner": winner["library"],
                "winner_throughput": winner["median_throughput"],
                "num_compared": len(group),
            },
        )
    return pd.DataFrame(rows)


def _winner_counts(winner_rows: pd.DataFrame) -> pd.DataFrame:
    if winner_rows.empty:
        return pd.DataFrame(columns=["regime", "regime_label", "winner", "wins"])
    return (
        winner_rows.groupby(["regime", "regime_label", "winner"], sort=True)
        .size()
        .reset_index(name="wins")
        .sort_values(["regime", "winner"])
    )


def _coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    dataloader = df[df["regime"].isin(["rgb_dataloader_cpu", "rgb_dataloader_gpu"])].copy()
    universe_rows = int(dataloader[dataloader["regime"] == "rgb_dataloader_cpu"]["transform"].nunique())
    rows = []
    for (regime, regime_label, library), group in dataloader.groupby(["regime", "regime_label", "library"], sort=True):
        measured = group[_measured(group)]
        rows.append(
            {
                "regime": regime,
                "regime_label": regime_label,
                "library": library,
                "full": len(measured),
                "universe_rows": universe_rows,
                "median_throughput": float(measured["median_throughput"].median()) if not measured.empty else 0.0,
            },
        )
    return pd.DataFrame(rows)


def _open_dataloader_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    dataloader = df[df["regime"].isin(["rgb_dataloader_cpu", "rgb_dataloader_gpu"]) & _measured(df)].copy()
    if dataloader.empty:
        return
    dataloader["implementation"] = dataloader.apply(
        lambda row: (
            f"{LIBRARY_DISPLAY.get(row['library'], row['library'])} "
            f"{'CPU' if row['regime'] == 'rgb_dataloader_cpu' else 'GPU'}"
        ),
        axis=1,
    )
    winners = []
    for transform, group in dataloader.groupby("transform", sort=True):
        winner = group.sort_values("median_throughput", ascending=False).iloc[0]
        winners.append(
            {
                "transform": transform,
                "library": winner["library"],
                "implementation": winner["implementation"],
                "median_throughput": winner["median_throughput"],
            },
        )
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(output_dir / "open_dataloader_winners.csv", index=False)

    coverage = _coverage_summary(df)
    coverage["implementation"] = coverage.apply(
        lambda row: (
            f"{LIBRARY_DISPLAY.get(row['library'], row['library'])} "
            f"{'CPU' if row['regime'] == 'rgb_dataloader_cpu' else 'GPU'}"
        ),
        axis=1,
    )
    win_counts = winners_df.groupby("implementation").size().to_dict()
    coverage["wins"] = coverage["implementation"].map(win_counts).fillna(0).astype(int)
    coverage = coverage[["library", "implementation", "median_throughput", "full", "universe_rows", "wins"]]
    coverage.to_csv(output_dir / "open_dataloader_leaderboard.csv", index=False)


def _gpu_vs_cpu(df: pd.DataFrame) -> pd.DataFrame:
    measured = df[_measured(df)]
    cpu = measured[(measured["regime"] == "rgb_dataloader_cpu") & (measured["library"] == "albumentationsx")]
    gpu = measured[measured["regime"] == "rgb_dataloader_gpu"]
    cpu_lookup = cpu.set_index("transform")["median_throughput"].to_dict()
    rows = []
    for row in gpu.itertuples(index=False):
        cpu_throughput = cpu_lookup.get(row.transform)
        if not cpu_throughput:
            continue
        rows.append(
            {
                "library": row.library,
                "transform": row.transform,
                "gpu_median_throughput": row.median_throughput,
                "albumentationsx_cpu_median_throughput": cpu_throughput,
                "gpu_to_albumentationsx_cpu": row.median_throughput / cpu_throughput,
            },
        )
    return pd.DataFrame(rows)


def _pivot(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    measured = df[(df["regime"] == regime) & _measured(df)]
    if measured.empty:
        return pd.DataFrame()
    return measured.pivot_table(index="transform", columns="library", values="median_throughput", aggfunc="first")


def _write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    summary: dict[str, Any] = {}
    for regime, group in df.groupby("regime", sort=True):
        measured = group[_measured(group)]
        summary[regime] = {
            "regime_label": group["regime_label"].iloc[0],
            "rows": len(group),
            "measured_rows": len(measured),
            "libraries": sorted(group["library"].unique().tolist()),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Paper Data Summary", "", "Generated from sanitized raw benchmark JSONs.", ""]
    for regime, item in summary.items():
        lines.append(
            f"- {item['regime_label']} (`{regime}`): {item['measured_rows']} measured rows "
            f"across {', '.join(item['libraries'])}.",
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def generate(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _aggregate_rows(_raw_rows(raw_dir))
    df.to_csv(output_dir / "all_results.csv", index=False)

    support = df[df["regime"].isin(["rgb_dataloader_cpu", "rgb_dataloader_gpu"])].copy()
    support.to_csv(output_dir / "production_support_matrix.csv", index=False)
    support.to_markdown(output_dir / "production_support_matrix.md", index=False)

    unsupported = df[(~df["supported"].astype(bool)) | df["early_stopped"].astype(bool)].copy()
    unsupported.to_csv(output_dir / "unsupported_and_early_stopped.csv", index=False)
    unsupported.to_markdown(output_dir / "unsupported_and_early_stopped.md", index=False)

    winners = _winner_rows(df)
    winners.to_csv(output_dir / "figure_winner_rows.csv", index=False)
    _winner_counts(winners).to_csv(output_dir / "figure_winner_counts.csv", index=False)
    coverage = _coverage_summary(df)
    coverage.to_csv(output_dir / "figure_coverage_vs_throughput.csv", index=False)
    _open_dataloader_outputs(df, output_dir)
    df[
        df["regime"].isin(["rgb_dataloader_cpu", "rgb_dataloader_gpu"])
        & (df["transform"] == "RandomCrop224+Elastic+Normalize+ToTensor")
    ].to_csv(output_dir / "figure_elastic_row.csv", index=False)
    _gpu_vs_cpu(df).to_csv(output_dir / "figure_gpu_vs_albumentationsx_cpu.csv", index=False)

    gpu_memory = df[(df["regime"] == "rgb_dataloader_gpu") & _measured(df) & df["gpu_peak_allocated_mb"].notna()].copy()
    gpu_memory["measured"] = True
    gpu_memory["gpu_peak_allocated_gb"] = gpu_memory["gpu_peak_allocated_mb"] / 1024.0
    gpu_memory["gpu_peak_reserved_gb"] = gpu_memory["gpu_peak_reserved_mb"] / 1024.0
    gpu_memory.to_csv(output_dir / "figure_gpu_memory_rows.csv", index=False)

    _pivot(df, "rgb_dataloader_cpu").to_csv(output_dir / "rgb_dataloader_cpu_pivot.csv")
    _pivot(df, "rgb_dataloader_gpu").to_csv(output_dir / "rgb_dataloader_gpu_pivot.csv")
    _write_summaries(df, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper CSV inputs from sanitized raw result JSONs.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_dir = args.raw if args.raw.is_absolute() else ROOT / args.raw
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    generate(raw_dir, output_dir)
    sys.stdout.write(f"Wrote paper data to {output_dir}\n")


if __name__ == "__main__":
    main()
