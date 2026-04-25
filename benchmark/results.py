from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmark.utils import get_library_versions, get_system_info, verify_thread_settings

if TYPE_CHECKING:
    from pathlib import Path


def summarize_runs(throughputs: list[float], times: list[float]) -> dict[str, Any]:
    if not throughputs:
        return {
            "supported": False,
            "throughputs": [],
            "median_throughput": 0.0,
            "std_throughput": 0.0,
            "times": times,
            "mean_time": 0.0,
            "std_time": 0.0,
        }

    return {
        "supported": True,
        "throughputs": throughputs,
        "median_throughput": float(np.median(throughputs)),
        "std_throughput": float(np.std(throughputs, ddof=1)) if len(throughputs) > 1 else 0.0,
        "times": times,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
    }


def unsupported_result(reason: str) -> dict[str, Any]:
    return {
        "supported": False,
        "reason": reason,
        "throughputs": [],
        "median_throughput": 0.0,
        "std_throughput": 0.0,
        "times": [],
        "mean_time": 0.0,
        "std_time": 0.0,
    }


def build_metadata(
    *,
    scenario: str,
    mode: str,
    library: str | None = None,
    decoder: str | None = None,
    benchmark_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package_key = library or decoder or "benchmark"
    return {
        "system_info": get_system_info(),
        "library_versions": get_library_versions(package_key),
        "thread_settings": verify_thread_settings(),
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
