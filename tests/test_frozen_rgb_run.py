from pathlib import Path

from augbench.frozen_rgb_run import build_frozen_rgb_run


def test_frozen_rgb_run_resolves_one_config_lock_and_matrix() -> None:
    root = Path(__file__).parents[1]

    frozen = build_frozen_rgb_run(
        repository_root=root,
        git_commit="a" * 40,
        code_archive_sha256="b" * 64,
    )

    assert frozen.request.run == frozen.run
    assert frozen.request.pending_cell_ids == tuple(cell.cell_id for cell in frozen.cells)
    assert frozen.request.environment_lock_path == "environments/rgb/lock.txt"
    assert frozen.run.family_config["normalization"] == "gpu"
    assert frozen.run.hardware == {
        "machine_type": "g2-standard-16",
        "accelerator": "nvidia-l4",
        "image": "projects/deeplearning-platform-release/global/images/common-cu129-ubuntu-2204-nvidia-580-v20260730",
        "provisioning_model": "STANDARD",
        "boot_disk_type": "pd-balanced",
        "boot_disk_size_gib": "350",
        "python_version": "3.13.14",
    }
    assert len(frozen.cells) == 759
