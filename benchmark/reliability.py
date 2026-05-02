from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class AuditReport:
    files_checked: int
    issues: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "files_checked": self.files_checked,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def git_state(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    git_executable = shutil.which("git")
    if git_executable is None:
        return {"commit": None, "branch": None, "dirty": None}

    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(  # noqa: S603
                [git_executable, *args],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run_git(["status", "--porcelain"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
    }


def environment_snapshot(repo_root: Path | None = None) -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "argv": sys.argv,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "thread_env": {name: os.environ.get(name) for name in _THREAD_ENV_VARS},
        "git": git_state(repo_root),
    }


def gpu_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False, "cuda_available": False}

    info: dict[str, Any] = {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }
    if not torch.cuda.is_available():
        return info

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    info.update(
        {
            "device_index": device,
            "device_name": torch.cuda.get_device_name(device),
            "total_memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
        },
    )
    return info


def dataset_fingerprint(data_dir: Path | None, *, media: str | None = None) -> dict[str, Any]:
    if data_dir is None:
        return {"available": False, "reason": "data_dir not provided"}

    extensions = {
        "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
        "video": {".mp4", ".avi", ".mov", ".mkv", ".webm"},
    }.get(media or "")

    paths = sorted(path for path in data_dir.rglob("*") if path.is_file())
    if extensions is not None:
        paths = [path for path in paths if path.suffix.lower() in extensions]

    suffix_counts = Counter(path.suffix.lower() or "<none>" for path in paths)
    total_bytes = sum(path.stat().st_size for path in paths)
    return {
        "available": True,
        "path": str(data_dir),
        "media": media,
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "extensions": dict(sorted(suffix_counts.items())),
        "sample_paths": [str(path.relative_to(data_dir)) for path in paths[:10]],
    }


def timing_metadata(
    *,
    timing_backend: str,
    measurement_scope: str,
    data_source: str,
    includes_decode: bool,
    includes_collate: bool,
    includes_gpu_transfer: bool,
    includes_dataloader_workers: bool,
) -> dict[str, Any]:
    return {
        "timing_backend": timing_backend,
        "measurement_scope": measurement_scope,
        "data_source": data_source,
        "includes_decode": includes_decode,
        "includes_collate": includes_collate,
        "includes_gpu_transfer": includes_gpu_transfer,
        "includes_dataloader_workers": includes_dataloader_workers,
    }


def iter_result_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        if not path.name.endswith(".pyperf.json"):
            yield path
        return
    yield from sorted(path for path in path.rglob("*.json") if not path.name.endswith(".pyperf.json"))


def audit_result_payload(payload: dict[str, Any], *, source: Path) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []

    metadata = payload.get("metadata")
    results = payload.get("results")
    if not isinstance(metadata, dict):
        return [f"{source}: missing metadata object"], warnings
    if not isinstance(results, dict) or not results:
        return [f"{source}: missing or empty results object"], warnings

    issues.extend(
        f"{source}: metadata missing {key!r}"
        for key in ["system_info", "library_versions", "thread_settings", "benchmark_params", "timing"]
        if key not in metadata
    )

    timing = metadata.get("timing", {})
    if isinstance(timing, dict):
        issues.extend(
            f"{source}: timing metadata missing {key!r}"
            for key in ["timing_backend", "measurement_scope", "data_source"]
            if not timing.get(key)
        )
    else:
        issues.append(f"{source}: metadata.timing is not an object")

    benchmark_params = metadata.get("benchmark_params", {})
    if isinstance(benchmark_params, dict):
        num_runs = benchmark_params.get("num_runs")
        if isinstance(num_runs, int) and num_runs < 3:
            warnings.append(f"{source}: num_runs={num_runs}; OK for production-path checks, weak for paper results")

    unsupported = 0
    unstable = 0
    failed = 0
    for name, result in results.items():
        if not isinstance(result, dict):
            issues.append(f"{source}: result {name!r} is not an object")
            continue
        if result.get("supported") is False:
            unsupported += 1
        if result.get("unstable"):
            unstable += 1
        if result.get("status") == "error":
            failed += 1
        if result.get("supported") is True and not result.get("throughputs"):
            warnings.append(f"{source}: supported result {name!r} has no throughput samples")

    if unsupported:
        warnings.append(f"{source}: {unsupported} unsupported results")
    if unstable:
        warnings.append(f"{source}: {unstable} unstable results")
    if failed:
        issues.append(f"{source}: {failed} failed results")

    return issues, warnings


def audit_results(path: Path) -> AuditReport:
    issues: list[str] = []
    warnings: list[str] = []
    files_checked = 0
    for result_file in iter_result_files(path):
        files_checked += 1
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            issues.append(f"{result_file}: invalid JSON: {e}")
            continue
        file_issues, file_warnings = audit_result_payload(payload, source=result_file)
        issues.extend(file_issues)
        warnings.extend(file_warnings)

    if files_checked == 0:
        issues.append(f"{path}: no JSON result files found")
    return AuditReport(files_checked=files_checked, issues=issues, warnings=warnings)


def doctor_report(repo_root: Path | None = None) -> dict[str, Any]:
    env = environment_snapshot(repo_root)
    warnings: list[str] = []

    for name, value in env["thread_env"].items():
        if value not in {"1", None}:
            warnings.append(f"{name}={value}; single-thread CPU benchmarks should usually set it to 1")

    try:
        import pyperf  # noqa: F401
    except ImportError:
        warnings.append("pyperf is not installed; micro benchmarks require pyperf")

    if env["git"].get("dirty"):
        warnings.append("git working tree is dirty; record this in paper/release benchmark artifacts")

    return {
        "ok": not warnings,
        "warnings": warnings,
        "environment": env,
        "gpu": gpu_snapshot(),
    }
