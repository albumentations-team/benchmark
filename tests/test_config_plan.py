from __future__ import annotations

from pathlib import Path

from benchmark.config import BenchmarkRunConfig, build_run_plan, load_run_config


def test_plan_lists_micro_jobs_and_outputs() -> None:
    config = load_run_config(Path("configs/examples/local_rgb_micro_cpu.yaml"))

    plan = build_run_plan(config, Path.cwd())

    assert len(plan.jobs) == 4
    assert plan.jobs[0].scenario == "image-rgb"
    assert plan.jobs[0].mode == "micro"
    assert plan.jobs[0].backend == "pyperf"
    assert "output/rgb_micro/image-rgb/micro/albumentationsx_micro_results.json" in plan.expected_outputs


def test_plan_lists_gpu_pipeline_device_outputs_and_cloud() -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_dataloader_gpu_smoke.yaml"))

    plan = build_run_plan(config, Path.cwd())

    assert [job.library for job in plan.jobs] == ["torchvision", "kornia"]
    assert all(job.backend == "pipeline" for job in plan.jobs)
    assert all(job.device == "cuda" for job in plan.jobs)
    assert all(job.data_dir == "/root/benchmark-data/val" for job in plan.jobs)
    assert any("_dev-cuda_results.json" in output for output in plan.expected_outputs)
    assert plan.cloud is not None
    assert plan.cloud["machine_type"] == "g2-standard-16"
    assert plan.cloud["execution_output_dir"] == "/root/benchmark-work/results"


def test_plan_lists_production_gpu_pipeline_outputs() -> None:
    config = load_run_config(Path("configs/paper/prod_g2_rgb_dataloader_gpu.yaml"))

    plan = build_run_plan(config, Path.cwd())

    assert [job.library for job in plan.jobs] == ["torchvision", "kornia"]
    assert all(job.device == "cuda" for job in plan.jobs)
    assert all("_n10000_r1_w8_b256_dev-cuda_results.json" in job.output_file for job in plan.jobs)
    assert (
        "/root/benchmark-work/results/image-rgb/pipeline/"
        "torchvision_memory_dataloader_augment_n10000_r1_w8_b256_dev-cuda_results.json"
    ) in plan.expected_outputs
    assert (
        "/root/benchmark-work/results/image-rgb/pipeline/"
        "kornia_memory_dataloader_augment_n10000_r1_w8_b256_dev-cuda_results.json"
    ) in plan.expected_outputs


def test_plan_lists_dali_rgb_pipeline_job() -> None:
    config = load_run_config(Path("configs/paper/prod_g2_rgb_dataloader_dali.yaml"))

    plan = build_run_plan(config, Path.cwd())

    assert [job.library for job in plan.jobs] == ["dali"]
    assert plan.jobs[0].backend == "dali_pipeline"
    assert plan.jobs[0].media == "image"
    assert plan.jobs[0].spec_file is None
    assert plan.jobs[0].device == "cuda"
    assert (
        "/root/benchmark-work/results/image-rgb/pipeline/"
        "dali_decode_dataloader_augment_n10000_r1_w8_b256_dev-cuda_results.json"
    ) in plan.expected_outputs


def test_plan_lists_decode_sidecar_and_combined_outputs() -> None:
    data = load_run_config(Path("configs/paper/gcp_g2_video_smoke.yaml")).model_dump()
    data["selection"] = {"scenario": "video-decode-16f", "mode": "decode", "decoders": ["opencv", "pyav"]}
    config = BenchmarkRunConfig.model_validate(data)

    plan = build_run_plan(config, Path.cwd())

    assert "/root/benchmark-work/results/video-decode-16f/decode/opencv_decode_results.json" in plan.expected_outputs
    assert "/root/benchmark-work/results/video-decode-16f/decode/video_decode_results.json" in plan.expected_outputs
