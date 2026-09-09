"""Build the sole immutable input record for an RGB production run."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - this module reads and hashes frozen files at runtime.

from augbench.guest_request import GuestRequest
from augbench.matrix import build_matrix
from augbench.recipes.load import load_recipe_catalog
from augbench.run_config import load_family_config, load_gcp_config
from augbench.run_records import CellKey, RunInputs, RunRecord, build_run_record


@dataclass(frozen=True)
class FrozenRgbRun:
    run: RunRecord
    request: GuestRequest
    cells: tuple[CellKey, ...]


def build_frozen_rgb_run(
    *,
    repository_root: Path,
    git_commit: str,
    code_archive_sha256: str,
) -> FrozenRgbRun:
    """Resolve repository inputs once; no historic epoch or suite is involved."""
    config = load_family_config(repository_root / "configs/families/rgb.yaml")
    cloud = load_gcp_config(repository_root / config.cloud)
    recipes_path = repository_root / config.recipes
    lock_path = repository_root / "environments/rgb/lock.txt"
    recipes = load_recipe_catalog(recipes_path)
    recipe_sha256 = _sha256_file(recipes_path)
    lock_sha256 = _sha256_file(lock_path)
    run = build_run_record(
        family_config=config.model_dump(mode="json"),
        inputs=RunInputs(
            git_commit=git_commit,
            code_archive_sha256=code_archive_sha256,
            dataset_archive_sha256=config.dataset.archive_sha256,
            recipe_catalog_sha256=recipe_sha256,
            environment_lock_sha256={"rgb": lock_sha256},
        ),
        hardware={
            "machine_type": cloud.machine_type,
            "accelerator": cloud.accelerator,
            "image": cloud.image,
            "provisioning_model": cloud.provisioning_model,
            "boot_disk_type": "pd-balanced",
            "boot_disk_size_gib": str(cloud.boot_disk_size_gib),
            "python_version": cloud.python_version,
        },
    )
    cells = build_matrix(run_id=run.run_id, config=config, recipes=recipes)
    request = GuestRequest(
        run=run,
        gcs_base_uri=cloud.gcs_base_uri,
        code_archive_uri=f"{cloud.gcs_base_uri.rstrip('/')}/code/{code_archive_sha256}/source.tar.gz",
        dataset_archive_uri=config.dataset.archive_uri,
        dataset_archive_sha256=config.dataset.archive_sha256,
        environment_cache_uri=(f"{cloud.gcs_base_uri.rstrip('/')}/environments/rgb/{lock_sha256}/environment.tar.gz"),
        environment_lock_path="environments/rgb/lock.txt",
        environment_lock_sha256=lock_sha256,
        environment_python_version=cloud.python_version,
        pending_cell_ids=tuple(cell.cell_id for cell in cells),
    )
    return FrozenRgbRun(run=run, request=request, cells=cells)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
