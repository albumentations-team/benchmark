from __future__ import annotations

import pytest

from augbench.run_records import (
    CellKey,
    GpuMemory,
    OutputObservation,
    ResultRecord,
    RunInputs,
    Throughput,
    TrainingObservation,
    build_run_record,
)
from paper.generate_training import AX, TrainingRun, markdown, summarize

RECIPE = "RandomResizedCrop+HorizontalFlip"
SEEDS = (137, 138, 139, 140, 141)


def _run() -> TrainingRun:
    run = build_run_record(
        family_config={"family": "imagenet_train", "execution": {"seeds": list(SEEDS), "measured_batches": 150}},
        inputs=RunInputs(
            git_commit="a" * 40,
            code_archive_sha256="b" * 64,
            dataset_archive_sha256="c" * 64,
            recipe_catalog_sha256="d" * 64,
            environment_lock_sha256={"rgb": "e" * 64},
        ),
        hardware={"machine_type": "g2-standard-16"},
    )

    def records(path: str, rate: float, ceiling: float, wait: float) -> list[ResultRecord]:
        return [
            ResultRecord(
                run_id=run.run_id,
                cell=CellKey(
                    run_id=run.run_id, family="imagenet_train", implementation=path, recipe_id=RECIPE, seed=seed
                ),
                throughput=Throughput(completed_items=38_400, duration_seconds=38_400 / (rate + index)),
                gpu_memory=GpuMemory(peak_mib=5_000, baseline_mib=200, poll_interval_ms=50, valid_samples=9),
                output=OutputObservation(shape=(256, 3, 224, 224)),
                training=TrainingObservation(
                    model="resnet18",
                    ceiling_images_per_second=ceiling,
                    loader_wait_seconds=wait * 38_400 / (rate + index),
                    steady_peak_mib=4_900,
                    loss_finite=True,
                ),
            )
            for index, seed in enumerate(SEEDS)
        ]

    groups = {
        (AX, RECIPE): records(AX, 1_700, 1_720, 0.0),
        ("kornia_cpu", RECIPE): records("kornia_cpu", 850, 1_720, 0.5),
        ("pillow_cpu", RECIPE): records("pillow_cpu", 1_702, 1_720, 0.01),
    }
    return TrainingRun(
        run=run,
        model="resnet18",
        batch_size=256,
        recipes=(RECIPE,),
        paths=(AX, "kornia_cpu", "pillow_cpu"),
        groups=groups,
    )


def test_each_path_reports_its_share_of_the_ceiling_and_a_verdict_against_ax() -> None:
    rows = {row.implementation: row for row in summarize(_run())}

    assert rows[AX].ratio_to_ax is None
    assert rows["kornia_cpu"].of_ceiling_median == pytest.approx(852 / 1_720)
    assert rows["kornia_cpu"].ratio_to_ax_verdict == "slower"
    assert rows["pillow_cpu"].ratio_to_ax_verdict == "indistinguishable"
    assert rows["kornia_cpu"].loader_wait_fraction_median == pytest.approx(0.5)
    assert rows[AX].interval_confidence == 0.9375


def test_the_summary_orders_paths_by_speed_and_names_the_model_and_verdicts() -> None:
    run = _run()
    text = markdown([(run, summarize(run))])

    assert "## ResNet-18, batch 256" in text
    assert "150 batches" in text
    assert text.index("| Pillow |") < text.index("| Kornia CPU |")
    assert "slower" in text
    assert "tied" in text
