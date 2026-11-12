"""Run the missing RGB cells on one staged GCE VM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from augbench.cloud.gcp.storage import GcloudObjectStore
from augbench.dataset_access import prewarm_files
from augbench.dataset_materialize import materialize_rgb_dataset
from augbench.guest_request import GuestRequest
from augbench.matrix import build_matrix
from augbench.recipes.load import load_recipe_catalog
from augbench.result_publisher import ResultPublisher
from augbench.rgb_executor import RGBCellExecutor, pillow_allsimd_runtime
from augbench.run_config import load_family_config

_AX_THREAD_ENV = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "VECLIB_MAXIMUM_THREADS",
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from augbench.recipes.models import RecipeCatalog
    from augbench.run_config import FamilyRunConfig
    from augbench.run_records import CellKey


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--family-config", type=Path, required=True)
    parser.add_argument("--dataset-archive", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--pillow-allsimd-site", type=Path, required=True)
    parser.add_argument("--implementation", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    request = GuestRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    config = load_family_config(args.family_config)
    recipes = load_recipe_catalog(config.recipes)
    cells = select_pending_cells(
        request=request,
        cells=build_matrix(run_id=request.run.run_id, config=config, recipes=recipes),
    )
    if not cells:
        return 0
    if args.implementation is None:
        materialized = materialize_rgb_dataset(
            archive=args.dataset_archive,
            dataset=config.dataset,
            cache_root=args.dataset_cache,
        )
        if config.execution.prewarm_dataset:
            prewarm_files(materialized.files)
        dataset_files = json.dumps([str(path) for path in materialized.files])
        pending_implementations = {cell.implementation for cell in cells}
        for implementation in config.implementations:
            if implementation in pending_implementations:
                _launch_worker(argv, implementation, dataset_files, args.pillow_allsimd_site)
        return 0

    _run_implementation(args.implementation, config, recipes, cells, request.gcs_base_uri)
    return 0


def _run_implementation(
    implementation: str,
    config: FamilyRunConfig,
    recipes: RecipeCatalog,
    cells: tuple[CellKey, ...],
    gcs_base_uri: str,
) -> None:
    cells = tuple(cell for cell in cells if cell.implementation == implementation)
    if not cells:
        raise ValueError(f"no pending cells for {implementation!r}")
    if implementation == "pillow_allsimd_cpu":
        fast, kernels = pillow_allsimd_runtime()
        if not fast or kernels == "scalar":
            raise RuntimeError("Pillow-AllSIMD requires fast mode and an active SIMD kernel")
    if implementation == "albumentationsx_cpu":
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    executor = RGBCellExecutor(
        config=config,
        recipes=recipes,
        dataset_files=tuple(Path(path) for path in json.load(sys.stdin)),
    )
    preflight_implementations(executor, cells)
    if implementation == "albumentationsx_cpu":
        _verify_ax_thread_limits()
    publisher = ResultPublisher(GcloudObjectStore(base_uri=gcs_base_uri))
    for cell in cells:
        publisher.publish(executor.execute(cell))


def _launch_worker(argv: list[str], implementation: str, dataset_files: str, allsimd_site: Path) -> None:
    env = os.environ.copy()
    if implementation == "albumentationsx_cpu":
        env.update(dict.fromkeys(_AX_THREAD_ENV, "1"))
    else:
        for name in _AX_THREAD_ENV:
            env.pop(name, None)
    if implementation == "pillow_allsimd_cpu":
        paths = [str(allsimd_site)]
        if pythonpath := env.get("PYTHONPATH"):
            paths.append(pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["PILLOW_ALLSIMD_FAST"] = "1"
    print(f"Starting {implementation}", flush=True)  # noqa: T201 - bootstrap progress in the VM log.
    subprocess.run(  # noqa: S603 - fixed module and validated implementation from the frozen run.
        [sys.executable, "-m", "augbench.guest_worker", *argv, "--implementation", implementation],
        env=env,
        input=dataset_files,
        text=True,
        check=True,
    )


def _verify_ax_thread_limits() -> None:
    import torch
    from threadpoolctl import threadpool_info

    counts = {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()}
    counts.update(torch_intraop=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads())
    if cv2 := sys.modules.get("cv2"):
        counts["opencv"] = cv2.getNumThreads()
    if any(count != 1 for count in counts.values()):
        raise RuntimeError(f"AlbumentationsX worker requires one thread per CPU kernel: {counts}")
    print(f"CPU thread counts: {json.dumps(counts, sort_keys=True)}", flush=True)  # noqa: T201 - bootstrap log.


def select_pending_cells(*, request: GuestRequest, cells: Iterable[CellKey]) -> tuple[CellKey, ...]:
    """Validate the entire request and retain its cells in matrix order."""
    indexed = {cell.cell_id: cell for cell in cells}
    requested = set(request.pending_cell_ids)
    unknown = requested - indexed.keys()
    if unknown:
        raise ValueError(f"guest request references unknown cells: {sorted(unknown)[:3]}")
    return tuple(cell for cell in indexed.values() if cell.cell_id in requested)


def preflight_implementations(executor: RGBCellExecutor, cells: Iterable[CellKey]) -> None:
    """Compile every implementation and execute one real batch per implementation-recipe pair."""
    compiled: set[str] = set()
    preflighted: set[tuple[str, str]] = set()
    for cell in cells:
        if cell.implementation not in compiled:
            compiled.add(cell.implementation)
            executor.compile_implementation(cell.implementation)
        pair = (cell.implementation, cell.recipe_id)
        if pair in preflighted:
            continue
        preflighted.add(pair)
        try:
            executor.preflight(cell)
        except Exception as error:
            raise RuntimeError(f"RGB preflight failed for {cell.implementation}: {cell.recipe_id}") from error


if __name__ == "__main__":
    raise SystemExit(main())
