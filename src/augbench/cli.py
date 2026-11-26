"""Small command line interface for the active production benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from augbench.datasets.prepare_grayscale import prepare_grayscale_archive
from augbench.launch_rgb import grayscale_config, launch_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="augbench", description="Disk-to-GPU augmentation benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("launch-rgb", help="resume or launch the frozen RGB L4 fleet")
    launch.add_argument("--repository-root", type=Path, default=Path.cwd())
    grayscale = commands.add_parser("launch-grayscale", help="resume or launch grayscale Colorize on L4")
    grayscale.add_argument("--repository-root", type=Path, default=Path.cwd())
    grayscale.add_argument("--archive-uri", required=True)
    grayscale.add_argument("--archive-sha256", required=True)
    prepare = commands.add_parser("prepare-grayscale", help="make grayscale ImageNet validation JPEGs")
    prepare.add_argument("--source-archive", type=Path, required=True)
    prepare.add_argument("--output-archive", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-grayscale":
        print(json.dumps(prepare_grayscale_archive(args.source_archive, args.output_archive), sort_keys=True))
        return 0
    root = args.repository_root.resolve()
    if args.command == "launch-grayscale":
        config = grayscale_config(
            repository_root=root, archive_uri=args.archive_uri, archive_sha256=args.archive_sha256
        )
        frozen, launch = launch_rgb(repository_root=root, config=config)
    else:
        frozen, launch = launch_rgb(repository_root=root)
    print(
        json.dumps(
            {
                "run_id": frozen.run.run_id,
                "status": launch.status,
                "pending_cells": len(launch.pending_cell_ids),
                "instances": [instance.model_dump() for instance in launch.instances],
                "shortfall": launch.shortfall,
            },
            sort_keys=True,
        ),
    )
    return 0
