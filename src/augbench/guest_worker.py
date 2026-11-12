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
from augbench.rgb_executor import RGBCellExecutor
from augbench.run_config import load_family_config

if TYPE_CHECKING:
    from collections.abc import Iterable

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

    cells = tuple(cell for cell in cells if cell.implementation == args.implementation)
    if not cells:
        raise ValueError(f"no pending cells for {args.implementation!r}")
    if args.implementation == "pillow_allsimd_cpu":
        from PIL import Image

        if not Image.get_fast() or Image.core.simd_kernels == "scalar":
            raise RuntimeError("Pillow-AllSIMD requires fast mode and an active SIMD kernel")
    executor = RGBCellExecutor(
        config=config,
        recipes=recipes,
        dataset_files=tuple(Path(path) for path in json.load(sys.stdin)),
    )
    preflight_implementations(executor, cells)
    publisher = ResultPublisher(GcloudObjectStore(base_uri=request.gcs_base_uri))
    for cell in cells:
        publisher.publish(executor.execute(cell))
    return 0


def _launch_worker(argv: list[str], implementation: str, dataset_files: str, allsimd_site: Path) -> None:
    env = os.environ.copy()
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
