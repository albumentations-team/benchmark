from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pyperf

from benchmark.results import build_metadata, summarize_runs, write_results
from benchmark.runner import BenchmarkRunner, MediaType, load_from_python_file
from benchmark.utils import materialize_transform_output

_SLOW_DEFAULTS: dict[MediaType, dict[str, float | int]] = {
    MediaType.IMAGE: {
        "threshold_sec_per_item": 0.1,
        "preflight_items": 10,
        "max_preflight_secs": 60,
    },
    MediaType.VIDEO: {
        "threshold_sec_per_item": 2.0,
        "preflight_items": 3,
        "max_preflight_secs": 120,
    },
}


def _make_micro_output_contiguous(output: Any) -> Any:
    """Materialize micro-benchmark outputs so views/lazy buffers are not counted as finished work."""
    return materialize_transform_output(output)


def _time_transform_loop(loops: int, transform: Any, media: list[Any], call_fn: Any) -> float:
    start = pyperf.perf_counter()
    for _ in range(loops):
        for item in media:
            _ = _make_micro_output_contiguous(call_fn(transform, item))
    return pyperf.perf_counter() - start


def _pyperf_value_throughputs(values: list[float]) -> list[float]:
    """Convert pyperf-normalized per-item seconds into items/second."""
    return [1.0 / value for value in values if value > 0]


def _add_worker_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--specs-file",
            str(args.specs_file),
            "--data-dir",
            str(args.data_dir),
            "--json-output",
            str(args.json_output),
            "--media",
            args.media,
            "--scenario",
            args.scenario,
            "--num-channels",
            str(args.num_channels),
        ],
    )
    if args.num_items is not None:
        cmd.extend(["--num-items", str(args.num_items)])
    if args.media_cache is not None:
        cmd.extend(["--media-cache", str(args.media_cache)])
    if args.transforms:
        cmd.extend(["--transforms", args.transforms])
    if args.slow_threshold_sec_per_item is not None:
        cmd.extend(["--slow-threshold-sec-per-item", str(args.slow_threshold_sec_per_item)])
    if args.slow_preflight_items is not None:
        cmd.extend(["--slow-preflight-items", str(args.slow_preflight_items)])
    if args.disable_slow_skip:
        cmd.append("--disable-slow-skip")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CPU micro benchmarks with pyperf")
    parser.add_argument("--specs-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--media", choices=["image", "video"], default="image")
    parser.add_argument("--scenario", default="manual")
    parser.add_argument("--num-items", type=int)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--transforms", default="")
    parser.add_argument("--media-cache", type=Path)
    parser.add_argument("--slow-threshold-sec-per-item", type=float)
    parser.add_argument("--slow-preflight-items", type=int)
    parser.add_argument("--disable-slow-skip", action="store_true")
    return parser


def _slow_skip_config(args: argparse.Namespace, media_type: MediaType) -> tuple[float, int, float]:
    defaults = _SLOW_DEFAULTS[media_type]
    threshold = (
        args.slow_threshold_sec_per_item
        if args.slow_threshold_sec_per_item is not None
        else float(defaults["threshold_sec_per_item"])
    )
    preflight_items = (
        args.slow_preflight_items if args.slow_preflight_items is not None else int(defaults["preflight_items"])
    )
    return threshold, preflight_items, float(defaults["max_preflight_secs"])


def _preflight_slow_transform(
    *,
    transform: Any,
    transform_name: str,
    media: list[Any],
    call_fn: Any,
    media_type: MediaType,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.disable_slow_skip:
        return None
    threshold, preflight_items, max_preflight_secs = _slow_skip_config(args, media_type)
    subset = media[: max(1, min(preflight_items, len(media)))]
    if not subset:
        return None

    start = time.perf_counter()
    elapsed = _time_transform_loop(1, transform, subset, call_fn)
    wall_time = time.perf_counter() - start
    time_per_item = elapsed / len(subset)
    throughput = len(subset) / elapsed if elapsed > 0 else 0.0
    item_label = "video" if media_type == MediaType.VIDEO else "image"
    unit = "vid/s" if media_type == MediaType.VIDEO else "img/s"

    reason: str | None = None
    if time_per_item >= threshold:
        reason = f"{transform_name} slower than threshold: {time_per_item:.3f} sec/{item_label} >= {threshold:.3f}"
    elif wall_time > max_preflight_secs:
        reason = f"{transform_name} preflight timeout: {wall_time:.1f} sec > {max_preflight_secs:.1f} sec"

    if reason is None:
        return None

    return {
        "status": "ok",
        "supported": True,
        "throughputs": [],
        "median_throughput": throughput,
        "mean_throughput": throughput,
        "std_throughput": 0.0,
        "cv_throughput": 0.0,
        "throughput_ci95": 0.0,
        "p50_throughput": throughput,
        "p75_throughput": throughput,
        "p90_throughput": throughput,
        "p95_throughput": throughput,
        "times": [],
        "mean_time": time_per_item,
        "std_time": 0.0,
        "num_successful_runs": 0,
        "variance_stable": False,
        "unstable": True,
        "unstable_reason": "early stopped during pyperf preflight",
        "early_stopped": True,
        "early_stop_reason": reason,
        "slow_threshold_sec_per_item": threshold,
        "slow_threshold_throughput": 1.0 / threshold if threshold > 0 else 0.0,
        "slow_threshold_unit": unit,
        "slow_marker": f"≤{1.0 / threshold:.0f} {unit}" if threshold > 0 else "slow-skipped",
        "preflight_items": len(subset),
        "preflight_elapsed": elapsed,
        "pyperf": None,
    }


def _run_transform_subprocesses(
    *,
    runner: pyperf.Runner,
    args: argparse.Namespace,
    library: str,
    transforms: list[dict[str, Any]],
) -> None:
    """Run one transform per pyperf process; pyperf worker mode assumes one benchmark target."""
    payload: dict[str, Any] | None = None
    combined_pyperf: dict[str, Any] = {"benchmarks": []}

    with tempfile.TemporaryDirectory(prefix="pyperf-micro-") as tmp:
        tmp_dir = Path(tmp)
        media_cache = tmp_dir / "media.pkl"
        _write_media_cache(args, media_cache, library)
        for transform_dict in transforms:
            transform_name = str(transform_dict["name"])
            result_path = tmp_dir / f"{transform_name}.json"
            pyperf_path = tmp_dir / f"{transform_name}.pyperf.json"
            cmd = [
                sys.executable,
                "-m",
                "benchmark.pyperf_micro_runner",
                "--specs-file",
                str(args.specs_file),
                "--data-dir",
                str(args.data_dir),
                "--json-output",
                str(result_path),
                "--output",
                str(pyperf_path),
                "--media",
                args.media,
                "--scenario",
                args.scenario,
                "--num-channels",
                str(args.num_channels),
                "--transforms",
                transform_name,
                "--media-cache",
                str(media_cache),
                "--processes",
                str(runner.args.processes),
                "--values",
                str(runner.args.values),
                "--warmups",
                str(runner.args.warmups),
                "--min-time",
                str(runner.args.min_time),
            ]
            if args.slow_threshold_sec_per_item is not None:
                cmd.extend(["--slow-threshold-sec-per-item", str(args.slow_threshold_sec_per_item)])
            if args.slow_preflight_items is not None:
                cmd.extend(["--slow-preflight-items", str(args.slow_preflight_items)])
            if args.disable_slow_skip:
                cmd.append("--disable-slow-skip")
            if args.num_items is not None:
                cmd.extend(["--num-items", str(args.num_items)])

            env = {**os.environ, "BENCHMARK_TRANSFORMS_FILTER": transform_name}
            subprocess.run(cmd, check=True, env=env)  # noqa: S603 - cmd is built from this runner's own executable/args.

            transform_payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload is None:
                transform_payload["results"] = {}
                payload = transform_payload
            payload_dict = payload
            if payload_dict is None:
                msg = "Combined pyperf payload was not initialized"
                raise RuntimeError(msg)
            results = payload_dict["results"]
            if not isinstance(results, dict):
                results = {}
                payload_dict["results"] = results
            results.update(transform_payload["results"])

            _merge_pyperf_payload(combined_pyperf, pyperf_path)

    if payload is None:
        payload = {"metadata": {}, "results": {}}
    write_results(args.json_output, payload)
    output_path = getattr(runner.args, "output", None)
    if output_path:
        Path(output_path).write_text(json.dumps(combined_pyperf, separators=(",", ":")), encoding="utf-8")


def _merge_pyperf_payload(combined_pyperf: dict[str, Any], pyperf_path: Path) -> None:
    if not pyperf_path.exists():
        return

    pyperf_payload = json.loads(pyperf_path.read_text(encoding="utf-8"))
    combined_pyperf["benchmarks"].extend(pyperf_payload.get("benchmarks", []))
    if "metadata" in pyperf_payload:
        combined_pyperf.setdefault("metadata", pyperf_payload["metadata"])
    if "version" in pyperf_payload:
        combined_pyperf.setdefault("version", pyperf_payload["version"])


def _write_media_cache(args: argparse.Namespace, media_cache: Path, library: str) -> None:
    media = _load_media(args, library)
    with media_cache.open("wb") as f:
        pickle.dump(media, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_media_cache(media_cache: Path) -> list[Any]:
    with media_cache.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def _load_media(args: argparse.Namespace, library: str) -> list[Any]:
    media_type = MediaType(args.media)
    media_loader = BenchmarkRunner(
        library=library,
        data_dir=args.data_dir,
        transforms=[],
        call_fn=lambda _transform, item: item,
        media_type=media_type,
        num_items=args.num_items,
        num_channels=args.num_channels,
    )
    return media_loader.load_media()


def _run_filtered_transforms(
    *,
    runner: pyperf.Runner,
    args: argparse.Namespace,
    library: str,
    call_fn: Any,
    transforms: list[dict[str, Any]],
) -> None:
    media_type = MediaType(args.media)
    media = _read_media_cache(args.media_cache) if args.media_cache is not None else _load_media(args, library)

    results: dict[str, Any] = {}
    for transform_dict in transforms:
        transform_name = str(transform_dict["name"])
        slow_result = None
        if not runner.args.worker:
            slow_result = _preflight_slow_transform(
                transform=transform_dict["transform"],
                transform_name=transform_name,
                media=media,
                call_fn=call_fn,
                media_type=media_type,
                args=args,
            )
        if slow_result is not None:
            results[transform_name] = slow_result
            continue

        bench = runner.bench_time_func(
            transform_name,
            _time_transform_loop,
            transform_dict["transform"],
            media,
            call_fn,
            inner_loops=len(media),
        )
        if runner.args.worker:
            return
        if bench is None:
            continue
        times = [float(value) for value in bench.get_values()]
        throughputs = _pyperf_value_throughputs(times)
        result = summarize_runs(throughputs, times)
        result["pyperf"] = {
            "name": bench.get_name(),
            "nrun": bench.get_nrun(),
            "nvalue": bench.get_nvalue(),
            "total_duration": bench.get_total_duration(),
            "loops": bench.get_loops(),
            "inner_loops": bench.get_inner_loops(),
        }
        results[transform_name] = result

    num_key = "num_videos" if media_type == MediaType.VIDEO else "num_images"
    output = {
        "metadata": build_metadata(
            scenario=args.scenario,
            mode="micro",
            library=library,
            benchmark_params={
                num_key: len(media),
                "num_runs": runner.args.values,
                "num_channels": args.num_channels,
                "timer_backend": "pyperf",
                "slow_skip_enabled": not args.disable_slow_skip,
                "slow_threshold_sec_per_item": _slow_skip_config(args, media_type)[0],
                "slow_preflight_items": _slow_skip_config(args, media_type)[1],
            },
            timing_backend="pyperf",
            measurement_scope="augmentation_only",
            data_source="memory",
            data_dir=args.data_dir,
            media=args.media,
            repo_root=Path(__file__).parent.parent,
        ),
        "results": results,
    }
    write_results(args.json_output, output)


def main() -> None:
    runner = pyperf.Runner(
        processes=1,
        values=3,
        warmups=1,
        min_time=0.001,
        program_args=("-m", "benchmark.pyperf_micro_runner"),
        add_cmdline_args=_add_worker_args,
        _argparser=_parser(),
    )
    args = runner.parse_args()

    if args.transforms:
        os.environ["BENCHMARK_TRANSFORMS_FILTER"] = args.transforms

    library, call_fn, transforms = load_from_python_file(args.specs_file)
    filter_names = [name.strip() for name in args.transforms.split(",") if name.strip()]
    transforms = BenchmarkRunner.filter_transforms(transforms, filter_names or None)

    if not runner.args.worker and len(transforms) > 1:
        _run_transform_subprocesses(runner=runner, args=args, library=library, transforms=transforms)
        return

    if not runner.args.worker and args.media_cache is None:
        with tempfile.TemporaryDirectory(prefix="pyperf-micro-") as tmp:
            args.media_cache = Path(tmp) / "media.pkl"
            _write_media_cache(args, args.media_cache, library)
            _run_filtered_transforms(runner=runner, args=args, library=library, call_fn=call_fn, transforms=transforms)
        return

    _run_filtered_transforms(runner=runner, args=args, library=library, call_fn=call_fn, transforms=transforms)


if __name__ == "__main__":
    main()
