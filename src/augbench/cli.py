"""Small command line interface for the active production benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from augbench.datasets.prepare_grayscale import prepare_grayscale_archive
from augbench.launch import launch
from augbench.run_config import FamilyRunConfig, load_family_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="augbench", description="Disk-to-GPU augmentation benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    launch_command = commands.add_parser("launch", help="resume or launch the frozen run of one family config")
    launch_command.add_argument("--repository-root", type=Path, default=Path.cwd())
    launch_command.add_argument(
        "--config", type=Path, required=True, help="family config, for example configs/families/rgb.yaml"
    )
    rgb = commands.add_parser("launch-rgb", help="alias for launch --config configs/families/rgb.yaml")
    rgb.add_argument("--repository-root", type=Path, default=Path.cwd())
    grayscale = commands.add_parser("launch-grayscale", help="launch configs/families/grayscale.yaml with an archive")
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
    config = _command_config(args, root)
    frozen, launched = launch(repository_root=root, config=config)
    print(
        json.dumps(
            {
                "run_id": frozen.run.run_id,
                "status": launched.status,
                "pending_cells": len(launched.pending_cell_ids),
                "instances": [instance.model_dump() for instance in launched.instances],
                "shortfall": launched.shortfall,
            },
            sort_keys=True,
        ),
    )
    return 0


def _command_config(args: argparse.Namespace, root: Path) -> FamilyRunConfig:
    if args.command == "launch":
        return load_family_config(args.config if args.config.is_absolute() else root / args.config)
    if args.command == "launch-grayscale":
        config = load_family_config(root / "configs/families/grayscale.yaml")
        dataset = {
            **config.dataset.model_dump(),
            "archive_uri": args.archive_uri,
            "archive_sha256": args.archive_sha256,
        }
        return FamilyRunConfig.model_validate({**config.model_dump(), "dataset": dataset})
    return load_family_config(root / "configs/families/rgb.yaml")
