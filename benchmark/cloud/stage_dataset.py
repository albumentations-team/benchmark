from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from benchmark.config import BenchmarkRunConfig

ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")
MediaName = Literal["image", "video"]
MEDIA_SUFFIXES = {
    "image": (".jpeg", ".jpg", ".png"),
    "video": (".mp4", ".avi", ".mov"),
}
MACOS_JUNK_NAMES = {".ds_store", "__macosx"}


@dataclass(frozen=True)
class DatasetStagePlan:
    media: MediaName
    mode: str
    limit: int
    is_archive: bool


def value_after_flag(args: list[str], flag: str) -> str:
    try:
        return str(args[args.index(flag) + 1])
    except (ValueError, IndexError):
        return ""


def infer_media(args: list[str]) -> MediaName:
    media = value_after_flag(args, "--media")
    if media in MEDIA_SUFFIXES:
        return cast("MediaName", media)

    scenario = value_after_flag(args, "--scenario")
    if scenario.startswith("video"):
        return "video"
    return "image"


def default_micro_limit(media: MediaName) -> int:
    return 50 if media == "video" else 1000


def _stage_fields_from_partial_run_config(run_config: dict[str, Any]) -> tuple[MediaName, str, str]:
    selection = run_config.get("selection", {})
    data = run_config.get("data", {})
    media = "video" if str(selection.get("scenario", "")).startswith("video") else str(selection.get("media", "image"))
    scenario = str(selection.get("scenario", ""))
    mode = str(selection.get("mode") or ("decode" if scenario.startswith("video-decode") else "micro"))
    num_items = str(data.get("num_items") or "")
    return cast("MediaName", media if media in MEDIA_SUFFIXES else "image"), mode, num_items


def _stage_fields_from_run_config(run_config: dict[str, Any]) -> tuple[MediaName, str, str]:
    try:
        config = BenchmarkRunConfig.model_validate(run_config)
    except ValidationError:
        return _stage_fields_from_partial_run_config(run_config)
    num_items = str(config.data.num_items or "")
    return config.resolved_media(), config.resolved_mode(), num_items


def build_stage_plan(job: dict[str, Any]) -> DatasetStagePlan:
    run_config = job.get("run_config")
    gcs_data_uri = str(job["gcs_data_uri"])
    if isinstance(run_config, dict):
        media, mode, num_items = _stage_fields_from_run_config(run_config)
    else:
        args = [str(arg) for arg in job["benchmark_cli_args"]]
        media = infer_media(args)
        mode = value_after_flag(args, "--mode")
        num_items = value_after_flag(args, "--num-items")
    is_archive = gcs_data_uri.lower().endswith(ARCHIVE_SUFFIXES)

    if mode == "micro" and not is_archive:
        msg = (
            "For --mode micro on GCP, --gcp-gcs-data-uri must point to a tarball "
            "(for example gs://.../imagenet/val.tar or gs://.../ucf101/ucf101.tar). "
            f"Got: {gcs_data_uri!r}"
        )
        raise SystemExit(msg)

    limit = int(num_items) if mode == "micro" and num_items else default_micro_limit(media) if mode == "micro" else 0
    return DatasetStagePlan(media=media, mode=mode, limit=limit, is_archive=is_archive)


def is_macos_junk(name: str) -> bool:
    parts = [part.lower() for part in Path(name).parts]
    return any(part in MACOS_JUNK_NAMES or part.startswith("._") for part in parts)


def is_media_member(member: tarfile.TarInfo, media: MediaName) -> bool:
    if not member.isfile() or is_macos_junk(member.name):
        return False
    return member.name.lower().endswith(MEDIA_SUFFIXES[media])


def extract_dataset_tar(tar_path: Path, data_dir: Path, *, media: MediaName, limit: int) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r:*") as tf:
        members = [member for member in tf.getmembers() if is_media_member(member, media)]
        members.sort(key=lambda member: member.name)
        selected_members = members[:limit] if limit else members
        if not selected_members:
            msg = f"No {media} files found in dataset tarball: {tar_path}"
            raise SystemExit(msg)
        tf.extractall(path=data_dir, members=selected_members, filter="data")
    return len(selected_members)


def load_job(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage a benchmark dataset tarball on a cloud VM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_p = subparsers.add_parser("validate-source")
    validate_p.add_argument("--job-json", type=Path, required=True)

    extract_p = subparsers.add_parser("extract")
    extract_p.add_argument("--job-json", type=Path, required=True)
    extract_p.add_argument("--tar-path", type=Path, required=True)
    extract_p.add_argument("--data-dir", type=Path, required=True)

    args = parser.parse_args()
    plan = build_stage_plan(load_job(args.job_json))
    if args.command == "extract":
        count = extract_dataset_tar(args.tar_path, args.data_dir, media=plan.media, limit=plan.limit)
        sys.stdout.write(f"Extracted {count} {plan.media} files from {args.tar_path} to {args.data_dir}\n")


if __name__ == "__main__":
    main()
