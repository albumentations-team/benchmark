"""Run DALI pipeline benchmarks in the library venv (subprocess entry point).

`benchmark.orchestrator.execute_job` delegates `backend=dali_pipeline` jobs here so
`envs.ensure_venv` installs NVIDIA DALI into `.venv_*` before any DALI import runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.jobs import BenchmarkJob
from benchmark.pipeline_runner import PipelineBenchmarkRunner
from benchmark.transforms.image_recipe_specs import recipe_augmentation_specs, recipe_name
from benchmark.transforms.video_recipe_specs import recipe_augmentation_specs as video_recipe_augmentation_specs
from benchmark.transforms.video_recipe_specs import recipe_name as video_recipe_name


def _dali_transforms_from_specs(*, media: str, transforms_filter: tuple[str, ...] = ()) -> list[dict[str, object]]:
    if media == "image":
        image_transforms: list[dict[str, object]] = [
            {
                "name": recipe_name(spec),
                "transform": {"name": spec.name, "params": spec.params},
            }
            for spec in recipe_augmentation_specs(num_channels=3)
        ]
        if not transforms_filter:
            return image_transforms
        allowed = set(transforms_filter)
        filtered: list[dict[str, object]] = []
        for transform in image_transforms:
            transform_spec = transform["transform"]
            spec_name = transform_spec["name"] if isinstance(transform_spec, dict) else ""
            if str(transform["name"]) in allowed or str(spec_name) in allowed:
                filtered.append(transform)
        return filtered

    video_transforms: list[dict[str, object]] = [
        {
            "name": video_recipe_name(spec),
            "transform": {"name": spec.name, "params": spec.params},
        }
        for spec in video_recipe_augmentation_specs()
    ]
    if not transforms_filter:
        return video_transforms
    allowed = set(transforms_filter)
    filtered = []
    for transform in video_transforms:
        transform_spec = transform["transform"]
        spec_name = transform_spec["name"] if isinstance(transform_spec, dict) else ""
        if str(transform["name"]) in allowed or str(spec_name) in allowed:
            filtered.append(transform)
    return filtered


def run_dali_pipeline_job(job: BenchmarkJob) -> None:
    runner = PipelineBenchmarkRunner(
        library=job.library,
        data_dir=job.data_dir,
        output_file=job.output_file,
        transforms=_dali_transforms_from_specs(media=job.media, transforms_filter=job.transforms_filter),
        call_fn=lambda _transform, item: item,
        media=job.media,
        scenario=job.scenario,
        num_items=job.num_items,
        num_runs=job.num_runs,
        batch_size=job.batch_size,
        workers=job.workers,
        min_time=job.min_time,
        min_batches=job.min_batches,
        num_channels=job.num_channels,
        clip_length=job.clip_length,
        pipeline_scope=job.pipeline_scope,  # type: ignore[arg-type]
        device=job.device,  # type: ignore[arg-type]
        thread_policy=job.thread_policy,  # type: ignore[arg-type]
        slow_threshold_sec_per_item=job.slow_threshold_sec_per_item,
        slow_preflight_items=job.slow_preflight_items,
        disable_slow_skip=job.disable_slow_skip,
    )
    runner.run()


def benchmark_job_from_json_dict(data: dict[str, Any]) -> BenchmarkJob:
    """Rebuild a frozen `BenchmarkJob` from `json.loads` + `dataclasses.asdict` output."""
    fixed: dict[str, Any] = dict(data)
    for key in ("data_dir", "output_file", "spec_file"):
        value = fixed.get(key)
        if value is not None:
            fixed[key] = Path(str(value))
    tf = fixed.get("transforms_filter", ())
    fixed["transforms_filter"] = tuple(str(x) for x in tf) if isinstance(tf, (list, tuple)) else ()
    return BenchmarkJob(**fixed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DALI pipeline benchmark job (internal subprocess)")
    parser.add_argument(
        "--job-file",
        type=Path,
        required=True,
        help="JSON serialized BenchmarkJob (from dataclasses.asdict).",
    )
    args = parser.parse_args()
    raw = json.loads(args.job_file.read_text(encoding="utf-8"))
    job = benchmark_job_from_json_dict(raw)
    if job.backend != "dali_pipeline":
        parser.error(f"expected backend=dali_pipeline, got {job.backend!r}")
    run_dali_pipeline_job(job)


if __name__ == "__main__":
    main()
