"""Run the missing RGB cells on one staged GCE VM."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--family-config", type=Path, required=True)
    parser.add_argument("--dataset-archive", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    args = parser.parse_args(argv)

    request = GuestRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    config = load_family_config(args.family_config)
    recipes = load_recipe_catalog(config.recipes)
    cells = select_pending_cells(
        request=request,
        cells=build_matrix(run_id=request.run.run_id, config=config, recipes=recipes),
    )
    materialized = materialize_rgb_dataset(
        archive=args.dataset_archive,
        dataset=config.dataset,
        cache_root=args.dataset_cache,
    )
    if config.execution.prewarm_dataset:
        prewarm_files(materialized.files)

    executor = RGBCellExecutor(
        config=config,
        recipes=recipes,
        dataset_files=materialized.files,
    )
    preflight_implementations(executor, cells)
    publisher = ResultPublisher(GcloudObjectStore(base_uri=request.gcs_base_uri))
    for cell in cells:
        publisher.publish(executor.execute(cell))
    return 0


def select_pending_cells(*, request: GuestRequest, cells: Iterable[CellKey]) -> tuple[CellKey, ...]:
    """Reject a request that does not exactly describe this immutable run."""
    indexed = {cell.cell_id: cell for cell in cells}
    requested = set(request.pending_cell_ids)
    unknown = requested - indexed.keys()
    if unknown:
        raise ValueError(f"guest request references unknown cells: {sorted(unknown)[:3]}")
    return tuple(cell for cell in indexed.values() if cell.cell_id in requested)


def preflight_implementations(executor: RGBCellExecutor, cells: Iterable[CellKey]) -> None:
    """Compile every recipe and execute one representative batch per implementation."""
    seen: set[str] = set()
    for cell in cells:
        if cell.implementation in seen:
            continue
        seen.add(cell.implementation)
        executor.compile_implementation(cell.implementation)
        executor.preflight(cell)


if __name__ == "__main__":
    raise SystemExit(main())
