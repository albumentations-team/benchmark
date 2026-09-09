"""Small command line interface for the active production benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from augbench.launch_rgb import launch_rgb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="augbench", description="Disk-to-GPU augmentation benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("launch-rgb", help="resume or launch the frozen RGB L4 matrix")
    launch.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "launch-rgb":
        raise AssertionError(f"unhandled command {args.command!r}")
    frozen, launch = launch_rgb(repository_root=args.repository_root.resolve())
    print(
        json.dumps(
            {
                "run_id": frozen.run.run_id,
                "status": launch.status,
                "pending_cells": len(launch.pending_cell_ids),
                "instance_name": launch.instance_name,
                "zone": launch.zone,
            },
            sort_keys=True,
        ),
    )
    return 0
