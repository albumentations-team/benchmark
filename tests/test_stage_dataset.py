from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from benchmark.cloud.stage_dataset import build_stage_plan, extract_dataset_tar
from benchmark.config import load_run_config


def _add_file(tf: tarfile.TarFile, name: str, payload: bytes = b"x") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def test_video_tar_extract_filters_macos_junk_and_limits_items(tmp_path: Path) -> None:
    tar_path = tmp_path / "ucf101.tar"
    with tarfile.open(tar_path, "w") as tf:
        _add_file(tf, "UCF101/ApplyEyeMakeup/v_0001.avi")
        _add_file(tf, "UCF101/ApplyEyeMakeup/v_0002.mp4")
        _add_file(tf, "__MACOSX/UCF101/._v_0003.avi")
        _add_file(tf, "UCF101/.DS_Store")
        _add_file(tf, "UCF101/ApplyEyeMakeup/notes.txt")

    out_dir = tmp_path / "data"
    count = extract_dataset_tar(tar_path, out_dir, media="video", limit=1)

    assert count == 1
    assert (out_dir / "UCF101" / "ApplyEyeMakeup" / "v_0001.avi").exists()
    assert not (out_dir / "__MACOSX").exists()
    assert not (out_dir / "UCF101" / ".DS_Store").exists()


def test_image_tar_extract_accepts_non_imagenet_layout(tmp_path: Path) -> None:
    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as tf:
        _add_file(tf, "class-a/image-001.jpg")
        _add_file(tf, "class-a/image-002.png")
        _add_file(tf, "class-a/._image-003.jpg")

    out_dir = tmp_path / "data"
    count = extract_dataset_tar(tar_path, out_dir, media="image", limit=0)

    assert count == 2
    assert (out_dir / "class-a" / "image-001.jpg").exists()
    assert (out_dir / "class-a" / "image-002.png").exists()


def test_micro_gcp_requires_tar_source() -> None:
    job = {
        "gcs_data_uri": "gs://bucket/ucf101-dir",
        "run_config": {
            "selection": {"scenario": "video-16f", "mode": "micro"},
            "data": {"num_items": 10},
        },
    }

    with pytest.raises(SystemExit, match="must point to a tarball"):
        build_stage_plan(job)


def test_video_micro_plan_uses_video_default_limit() -> None:
    job = {
        "gcs_data_uri": "gs://bucket/ucf101.tar",
        "run_config": {"selection": {"scenario": "video-16f", "mode": "micro"}, "data": {}},
    }

    plan = build_stage_plan(job)

    assert plan.media == "video"
    assert plan.limit == 50


def test_stage_plan_requires_typed_run_config() -> None:
    job = {"gcs_data_uri": "gs://bucket/ucf101.tar"}

    with pytest.raises(SystemExit, match="missing typed run_config"):
        build_stage_plan(job)


def test_stage_plan_accepts_typed_run_config_payload() -> None:
    job = {
        "gcs_data_uri": "gs://bucket/imagenet-val.tar",
        "run_config": {
            "selection": {"scenario": "image-rgb", "mode": "micro"},
            "data": {"num_items": 123},
        },
    }

    plan = build_stage_plan(job)

    assert plan.media == "image"
    assert plan.mode == "micro"
    assert plan.limit == 123


def test_stage_plan_uses_validated_run_config_when_available() -> None:
    config = load_run_config(Path("configs/paper/gcp_g2_rgb_gpu_smoke.yaml"))
    job = {
        "gcs_data_uri": "gs://bucket/imagenet/val.tar",
        "run_config": config.model_dump(mode="json", exclude_none=True),
    }

    plan = build_stage_plan(job)

    assert plan.media == "image"
    assert plan.mode == "pipeline"
    assert plan.limit == 0
