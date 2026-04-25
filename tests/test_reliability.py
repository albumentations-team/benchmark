from __future__ import annotations

import json

from benchmark.reliability import audit_results, dataset_fingerprint, doctor_report
from benchmark.results import build_metadata, summarize_runs, unsupported_result


def test_summarize_runs_reports_stability_metrics() -> None:
    result = summarize_runs([100.0, 110.0, 90.0], [0.1, 0.09, 0.11])

    assert result["supported"] is True
    assert result["status"] == "ok"
    assert result["num_successful_runs"] == 3
    assert result["p95_throughput"] > 0
    assert result["throughput_ci95"] > 0
    assert "cv_throughput" in result


def test_unsupported_result_has_reliability_fields() -> None:
    result = unsupported_result("not available")

    assert result["status"] == "unsupported"
    assert result["num_successful_runs"] == 0
    assert result["unstable"] is False


def test_build_metadata_includes_timing_and_dataset(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")

    metadata = build_metadata(
        scenario="image-rgb",
        mode="micro",
        library="albumentationsx",
        data_dir=tmp_path,
        media="image",
        timing_backend="simple",
        measurement_scope="augmentation_only",
        data_source="memory",
    )

    assert metadata["timing"]["timing_backend"] == "simple"
    assert metadata["dataset"]["file_count"] == 1
    assert metadata["environment"]["git"]["branch"] is not None


def test_dataset_fingerprint_counts_files(tmp_path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "b.txt").write_bytes(b"b")

    fingerprint = dataset_fingerprint(tmp_path, media="image")

    assert fingerprint["file_count"] == 1
    assert fingerprint["extensions"] == {".jpg": 1}


def test_audit_results_detects_valid_payload(tmp_path) -> None:
    payload = {
        "metadata": build_metadata(
            scenario="image-rgb",
            mode="micro",
            library="albumentationsx",
            timing_backend="simple",
            measurement_scope="augmentation_only",
            data_source="memory",
        ),
        "results": {"Resize": summarize_runs([1.0, 1.1, 0.9], [1.0, 0.9, 1.1])},
    }
    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_results(result_file)

    assert report.ok is True
    assert report.files_checked == 1


def test_audit_results_rejects_missing_timing(tmp_path) -> None:
    result_file = tmp_path / "results.json"
    result_file.write_text(
        json.dumps({"metadata": {"benchmark_params": {}}, "results": {"Resize": {"supported": True}}}),
        encoding="utf-8",
    )

    report = audit_results(result_file)

    assert report.ok is False
    assert any("metadata missing" in issue for issue in report.issues)


def test_doctor_report_is_structured() -> None:
    report = doctor_report()

    assert "warnings" in report
    assert "environment" in report
    assert "gpu" in report
