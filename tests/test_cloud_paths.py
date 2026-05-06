from __future__ import annotations

from benchmark.cloud.paths import VM_DATADIR, VM_RESULTS, staged_data_dir_for_gcs_uri


def test_vm_path_constants_match_startup_script_contract() -> None:
    assert VM_DATADIR == "/root/benchmark-data"
    assert VM_RESULTS == "/root/benchmark-work/results"


def test_staged_data_dir_matches_detached_gcp_archive_layout() -> None:
    assert staged_data_dir_for_gcs_uri("gs://bucket/imagenet/val.tar") == "/root/benchmark-data/val"
    assert staged_data_dir_for_gcs_uri("gs://bucket/datasets/ucf101.tar") == "/root/benchmark-data"
    assert staged_data_dir_for_gcs_uri("gs://bucket/imagenet/train") == "/root/benchmark-data/train"
    assert staged_data_dir_for_gcs_uri(None) == "/root/benchmark-data"
