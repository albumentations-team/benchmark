from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmark.reliability import dataset_fingerprint, environment_snapshot, gpu_snapshot, timing_metadata
from benchmark.utils import get_library_versions, get_system_info, verify_thread_settings

if TYPE_CHECKING:
    from pathlib import Path


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def summarize_runs(
    throughputs: list[float],
    times: list[float],
    *,
    unstable_cv_threshold: float = 0.15,
) -> dict[str, Any]:
    if not throughputs:
        return {
            "status": "unsupported",
            "supported": False,
            "throughputs": [],
            "median_throughput": 0.0,
            "mean_throughput": 0.0,
            "std_throughput": 0.0,
            "cv_throughput": 0.0,
            "throughput_ci95": 0.0,
            "p50_throughput": 0.0,
            "p75_throughput": 0.0,
            "p90_throughput": 0.0,
            "p95_throughput": 0.0,
            "times": times,
            "mean_time": 0.0,
            "std_time": 0.0,
            "num_successful_runs": 0,
            "unstable": False,
        }

    mean_throughput = float(np.mean(throughputs))
    std_throughput = float(np.std(throughputs, ddof=1)) if len(throughputs) > 1 else 0.0
    cv_throughput = std_throughput / mean_throughput if mean_throughput > 0 else 0.0
    throughput_ci95 = 1.96 * std_throughput / math.sqrt(len(throughputs)) if len(throughputs) > 1 else 0.0
    mean_time = float(np.mean(times)) if times else 0.0
    std_time = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    return {
        "status": "ok",
        "supported": True,
        "throughputs": throughputs,
        "median_throughput": float(np.median(throughputs)),
        "mean_throughput": mean_throughput,
        "std_throughput": std_throughput,
        "cv_throughput": cv_throughput,
        "throughput_ci95": throughput_ci95,
        "p50_throughput": _percentile(throughputs, 50),
        "p75_throughput": _percentile(throughputs, 75),
        "p90_throughput": _percentile(throughputs, 90),
        "p95_throughput": _percentile(throughputs, 95),
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "num_successful_runs": len(throughputs),
        "unstable": cv_throughput > unstable_cv_threshold,
        "unstable_reason": f"cv_throughput {cv_throughput:.3f} > {unstable_cv_threshold:.3f}"
        if cv_throughput > unstable_cv_threshold
        else None,
    }


def unsupported_result(reason: str) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "supported": False,
        "reason": reason,
        "throughputs": [],
        "median_throughput": 0.0,
        "mean_throughput": 0.0,
        "std_throughput": 0.0,
        "cv_throughput": 0.0,
        "throughput_ci95": 0.0,
        "p50_throughput": 0.0,
        "p75_throughput": 0.0,
        "p90_throughput": 0.0,
        "p95_throughput": 0.0,
        "times": [],
        "mean_time": 0.0,
        "std_time": 0.0,
        "num_successful_runs": 0,
        "unstable": False,
    }


def build_metadata(
    *,
    scenario: str,
    mode: str,
    library: str | None = None,
    decoder: str | None = None,
    benchmark_params: dict[str, Any] | None = None,
    timing_backend: str = "perf_counter",
    measurement_scope: str = "augmentation_only",
    data_source: str = "memory",
    data_dir: Path | None = None,
    media: str | None = None,
    includes_decode: bool = False,
    includes_collate: bool = False,
    includes_gpu_transfer: bool = False,
    includes_dataloader_workers: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    package_key = library or decoder or "benchmark"
    return {
        "system_info": get_system_info(),
        "library_versions": get_library_versions(package_key),
        "thread_settings": verify_thread_settings(),
        "environment": environment_snapshot(repo_root),
        "gpu": gpu_snapshot(),
        "dataset": dataset_fingerprint(data_dir, media=media),
        "timing": timing_metadata(
            timing_backend=timing_backend,
            measurement_scope=measurement_scope,
            data_source=data_source,
            includes_decode=includes_decode,
            includes_collate=includes_collate,
            includes_gpu_transfer=includes_gpu_transfer,
            includes_dataloader_workers=includes_dataloader_workers,
        ),
        "scenario": scenario,
        "mode": mode,
        "library": library,
        "decoder": decoder,
        "benchmark_params": benchmark_params or {},
    }


def write_results(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
