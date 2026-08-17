from augbench.guest_request import GuestRequest
from augbench.run_records import build_run_record


def test_guest_request_contains_one_run_and_only_missing_cells() -> None:
    run = build_run_record(
        family_config={"family": "rgb"},
        git_commit="a" * 40,
        code_archive_sha256="b" * 64,
        dataset_archive_sha256="d" * 64,
        recipe_catalog_sha256="e" * 64,
        environment_lock_sha256={"rgb": "f" * 64},
        hardware={"machine_type": "g2-standard-16"},
    )
    request = GuestRequest(
        run=run,
        gcs_base_uri="gs://imagenet_validation/augmentation-benchmark",
        code_archive_uri="gs://imagenet_validation/augmentation-benchmark/code/source.tar.gz",
        dataset_archive_uri="gs://imagenet_validation/imagenet/val.tar",
        dataset_archive_sha256="d" * 64,
        environment_cache_uri="gs://imagenet_validation/augmentation-benchmark/environments/rgb.tar.gz",
        environment_lock_path="environments/rgb/lock.txt",
        environment_lock_sha256="f" * 64,
        environment_python_version="3.13.14",
        pending_cell_ids=("1" * 64,),
    )

    assert GuestRequest.model_validate_json(request.model_dump_json()) == request
