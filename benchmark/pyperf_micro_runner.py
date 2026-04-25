from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyperf

from benchmark.results import build_metadata, summarize_runs, write_results
from benchmark.runner import BenchmarkRunner, MediaType, load_from_python_file


def _time_transform_loop(loops: int, transform: Any, media: list[Any], call_fn: Any) -> float:
    start = pyperf.perf_counter()
    for _ in range(loops):
        for item in media:
            _ = call_fn(transform, item)
    return pyperf.perf_counter() - start


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
    if args.transforms:
        cmd.extend(["--transforms", args.transforms])


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
    return parser


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

    library, call_fn, transforms = load_from_python_file(args.specs_file)
    filter_names = [name.strip() for name in args.transforms.split(",") if name.strip()]
    transforms = BenchmarkRunner.filter_transforms(transforms, filter_names or None)

    media_type = MediaType(args.media)
    media_loader = BenchmarkRunner(
        library=library,
        data_dir=args.data_dir,
        transforms=transforms,
        call_fn=call_fn,
        media_type=media_type,
        num_items=args.num_items,
        num_channels=args.num_channels,
    )
    media = media_loader.load_media()

    results: dict[str, Any] = {}
    for transform_dict in transforms:
        transform_name = str(transform_dict["name"])
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
        throughputs = [len(media) / value for value in times if value > 0]
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


if __name__ == "__main__":
    main()
