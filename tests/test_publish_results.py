from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tools.publish_results import publish_results

if TYPE_CHECKING:
    from pathlib import Path


def _write_result(path: Path, library: str) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "system_info": {
                        "platform": "test",
                        "python_executable": "/Users/example/workspace/benchmark/.venv/bin/python",
                    },
                    "library_versions": {library: "1.2.3"},
                    "environment": {
                        "pid": 123,
                        "cwd": "/Users/example/workspace/benchmark",
                        "argv": ["/Users/example/workspace/benchmark/benchmark/runner.py", "--ok"],
                    },
                    "data_dir": "/Users/example/data/imagenet/val",
                    "benchmark_params": {"num_images": 10, "num_runs": 1},
                },
                "results": {"Resize": {"supported": True, "median_throughput": 123.0}},
            },
        ),
        encoding="utf-8",
    )


def test_publish_results_copies_summary_jsons_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "output" / "image-rgb" / "micro"
    source.mkdir(parents=True)
    _write_result(source / "albumentationsx_micro_results.json", "albumentationsx")
    _write_result(source / "pillow_micro_results.json", "pillow")
    (source / "albumentationsx_micro_results.pyperf.json").write_text("{}", encoding="utf-8")

    destination = tmp_path / "results" / "published" / "website-rgb"

    publish_results(
        source_dir=source,
        destination_dir=destination,
        purpose="website-full",
        machine="test-machine",
    )

    assert (destination / "albumentationsx_micro_results.json").exists()
    assert (destination / "pillow_micro_results.json").exists()
    assert not (destination / "albumentationsx_micro_results.pyperf.json").exists()
    published = json.loads((destination / "albumentationsx_micro_results.json").read_text(encoding="utf-8"))
    assert published["metadata"]["environment"]["argv"][0] == "<external-path>"
    assert published["metadata"]["data_dir"] == "<external-path>"
    assert "pid" not in published["metadata"]["environment"]

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["purpose"] == "website-full"
    assert manifest["machine"] == "test-machine"
    assert manifest["library_versions"] == {"albumentationsx": "1.2.3", "pillow": "1.2.3"}
    assert manifest["files"] == ["albumentationsx_micro_results.json", "pillow_micro_results.json"]
    assert manifest["entries"][0]["system_info"]["python_executable"] == "<external-path>"


def test_publish_results_rejects_file_destination(tmp_path: Path) -> None:
    source = tmp_path / "output"
    source.mkdir()
    _write_result(source / "albumentationsx_micro_results.json", "albumentationsx")
    destination = tmp_path / "published.json"
    destination.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        publish_results(
            source_dir=source,
            destination_dir=destination,
            purpose="website-full",
            machine="test-machine",
            force=True,
        )


def test_publish_results_requires_force_for_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "output"
    source.mkdir()
    _write_result(source / "albumentationsx_micro_results.json", "albumentationsx")
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="Use --force"):
        publish_results(
            source_dir=source,
            destination_dir=destination,
            purpose="website-full",
            machine="test-machine",
        )


def test_publish_results_rejects_missing_summary_jsons(tmp_path: Path) -> None:
    source = tmp_path / "output"
    source.mkdir()
    (source / "albumentationsx_micro_results.pyperf.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="No summary"):
        publish_results(
            source_dir=source,
            destination_dir=tmp_path / "published",
            purpose="website-full",
            machine="test-machine",
        )
