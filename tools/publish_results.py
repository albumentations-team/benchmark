"""Publish curated benchmark result snapshots for docs/website inputs.

Copies only summary ``*_results.json`` files from an ignored run directory into a
tracked snapshot directory and writes a manifest. Raw ``*.pyperf.json`` files are
intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _git_value(args: list[str]) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        return subprocess.check_output(  # noqa: S603 - fixed git metadata probes, no user-controlled executable
            [git, *args],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_snapshot() -> dict[str, Any]:
    dirty = _git_value(["status", "--porcelain"])
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["branch", "--show-current"]),
        "dirty": bool(dirty),
    }


def _summary_result_files(source_dir: Path) -> list[Path]:
    files = sorted(path for path in source_dir.glob("*_results.json") if not path.name.endswith(".pyperf.json"))
    if not files:
        msg = f"No summary *_results.json files found in {source_dir}"
        raise ValueError(msg)
    return files


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path} does not contain a JSON object"
        raise TypeError(msg)
    if "metadata" not in payload or "results" not in payload:
        msg = f"{path} is not a benchmark summary result JSON"
        raise ValueError(msg)
    return payload


def _manifest(
    *,
    source_dir: Path,
    destination_dir: Path,
    files: list[Path],
    purpose: str,
    machine: str,
    notes: str | None,
) -> dict[str, Any]:
    root = _repo_root()
    entries = []
    library_versions: dict[str, Any] = {}
    for path in files:
        payload = _load_json(path)
        metadata = payload["metadata"]
        versions = metadata.get("library_versions", {})
        library = (
            path.name.removesuffix("_micro_results.json")
            .removesuffix("_pipeline_results.json")
            .removesuffix(
                "_results.json",
            )
        )
        if isinstance(versions, dict) and library in versions:
            library_versions[library] = versions[library]
        entries.append(
            {
                "file": path.name,
                "library": library,
                "num_results": len(payload["results"]) if isinstance(payload["results"], dict) else None,
                "benchmark_params": metadata.get("benchmark_params", {}),
                "system_info": metadata.get("system_info", {}),
                "library_versions": versions,
            },
        )

    return {
        "schema_version": 1,
        "published_at_utc": datetime.now(UTC).isoformat(),
        "purpose": purpose,
        "machine": machine,
        "source_dir": _rel(source_dir, root),
        "destination_dir": _rel(destination_dir, root),
        "files": [path.name for path in files],
        "libraries": sorted(library_versions),
        "library_versions": library_versions,
        "git": _git_snapshot(),
        "notes": notes,
        "entries": entries,
    }


def publish_results(
    *,
    source_dir: Path,
    destination_dir: Path,
    purpose: str,
    machine: str,
    notes: str | None = None,
    force: bool = False,
) -> Path:
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    if not source_dir.is_dir():
        msg = f"Source directory does not exist: {source_dir}"
        raise ValueError(msg)
    if destination_dir.exists() and any(destination_dir.iterdir()) and not force:
        msg = f"Destination already exists and is not empty: {destination_dir}. Use --force to replace it."
        raise ValueError(msg)

    files = _summary_result_files(source_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for stale in destination_dir.glob("*"):
        if force:
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()

    for path in files:
        shutil.copy2(path, destination_dir / path.name)

    manifest = _manifest(
        source_dir=source_dir,
        destination_dir=destination_dir,
        files=files,
        purpose=purpose,
        machine=machine,
        notes=notes,
    )
    manifest_path = destination_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish curated benchmark summary JSONs")
    parser.add_argument("--from", dest="source_dir", required=True, type=Path, help="Ignored run result directory")
    parser.add_argument("--to", dest="destination_dir", required=True, type=Path, help="Tracked snapshot directory")
    parser.add_argument("--purpose", required=True, help="Snapshot purpose, e.g. website-full or paper-core")
    parser.add_argument("--machine", required=True, help="Machine label, e.g. macos-m4max")
    parser.add_argument("--notes", help="Optional free-form notes for manifest.json")
    parser.add_argument("--force", action="store_true", help="Replace an existing destination snapshot")
    return parser


def main() -> None:
    args = _parser().parse_args()
    destination = publish_results(
        source_dir=args.source_dir,
        destination_dir=args.destination_dir,
        purpose=args.purpose,
        machine=args.machine,
        notes=args.notes,
        force=args.force,
    )
    print(f"Published summary results to {destination}")


if __name__ == "__main__":
    main()
