from __future__ import annotations

from pathlib import Path

from benchmark.output_naming import manual_micro_output_file, micro_output_file, pipeline_output_file


def test_micro_output_file_uses_explicit_device_suffix() -> None:
    assert micro_output_file(Path("out"), "kornia", device="cuda") == Path("out/kornia_micro_dev-cuda_results.json")
    assert micro_output_file(Path("out"), "kornia", device="none") == Path("out/kornia_micro_results.json")


def test_pipeline_output_file_encodes_scope_size_runs_workers_batch_and_device() -> None:
    assert pipeline_output_file(
        Path("out"),
        "torchvision",
        pipeline_scope="decode_dataloader_augment",
        num_items=100,
        num_runs=1,
        workers=2,
        batch_size=32,
        device="cuda",
    ) == Path("out/torchvision_decode_dataloader_augment_n100_r1_w2_b32_dev-cuda_results.json")


def test_manual_micro_output_file_preserves_legacy_video_suffix() -> None:
    assert manual_micro_output_file(Path("out"), "kornia", media="video", device="none") == Path(
        "out/kornia_video_results.json",
    )
