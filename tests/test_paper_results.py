import csv
from pathlib import Path

import pytest

from augbench.recipes.load import load_recipe_catalog
from augbench.result_store import encode_result
from augbench.run_config import load_family_config
from augbench.run_records import (
    CellKey,
    GpuMemory,
    OutputObservation,
    ResultRecord,
    RunInputs,
    RunRecord,
    Throughput,
    build_run_record,
)
from paper import generate_readme as readme
from paper import generate_results as paper


@pytest.fixture
def selected_run() -> RunRecord:
    config = load_family_config(Path(__file__).parents[1] / "configs/families/rgb.yaml")
    return build_run_record(
        family_config=config.model_dump(mode="json"),
        inputs=RunInputs(
            git_commit="a" * 40,
            code_archive_sha256="b" * 64,
            dataset_archive_sha256=config.dataset.archive_sha256,
            recipe_catalog_sha256="c" * 64,
            environment_lock_sha256={"rgb": "d" * 64},
        ),
        hardware={"machine_type": "g2-standard-16", "accelerator": "nvidia-l4"},
    )


@pytest.fixture
def paper_data() -> paper._PaperData:
    recipes = load_recipe_catalog(Path(__file__).parents[1] / "catalog/recipes/rgb.yaml").recipes[:3]
    observations = {
        paper.AX: ((100, 100, 100), (200, 200, 200), (1000, 1000, 1000)),
        "kornia_cpu": ((90, 100, 110), (95, 100, 105), (300, 300, 300)),
        "kornia_gpu": ((190, 200, 210), (1, 99, 99999), (300, 300, 300)),
    }
    memories = {
        paper.AX: (1000, 2000, 3000),
        "kornia_cpu": (100, 5000, 2900),
        "kornia_gpu": (1200, 100, 100),
    }
    records = []
    for implementation, _ in paper.IMPLEMENTATIONS:
        for index, recipe in enumerate(recipes):
            speeds = observations.get(implementation, ((100, 100, 100),) * 3)[index]
            memory = memories.get(implementation, (100, 100, 100))[index]
            for seed, speed in zip((137, 138, 139), speeds, strict=True):
                cell = CellKey(
                    run_id=paper.RUN_ID,
                    family="rgb",
                    implementation=implementation,
                    recipe_id=recipe.recipe_id,
                    seed=seed,
                )
                records.append(
                    ResultRecord(
                        run_id=paper.RUN_ID,
                        cell=cell,
                        status="ok",
                        throughput=Throughput(completed_items=8192, duration_seconds=8192 / speed),
                        gpu_memory=GpuMemory(peak_mib=memory, poll_interval_ms=50, valid_samples=10),
                        output=OutputObservation(shape=(256, 3, 224, 224)),
                        runtime={"library_version": "test"},
                    )
                )
    return paper._prepare(list(reversed(records)), recipes)


def test_pairwise_uses_recipe_medians_and_memory_from_the_selected_path(paper_data: paper._PaperData) -> None:
    rows = {row.recipe: row for row in paper_data.pairwise["Kornia"]}
    first, second, third = (rows[recipe.recipe_id] for recipe in paper_data.recipes)

    assert (first.implementation, first.competitor_memory) == ("kornia_gpu", 1200)
    assert (second.implementation, second.competitor_speed, second.competitor_memory) == ("kornia_cpu", 100, 5000)
    assert (third.implementation, third.competitor_memory) == ("kornia_cpu", 2900)


def test_summary_aggregates_paired_ratios_and_differences(tmp_path: Path, paper_data: paper._PaperData) -> None:
    output = tmp_path / "metrics.tex"
    paper._write_metrics(output, paper_data)

    metrics = output.read_text()
    assert r"\newcommand{\PairKorniaRatio}{0.50}" in metrics
    assert r"\newcommand{\PairKorniaMemoryDelta}{+200}" in metrics


def test_common_mean_averages_recipe_ratios_with_equal_weight(paper_data: paper._PaperData) -> None:
    means = paper._mean_ratios(paper_data, [implementation for implementation, _ in paper.IMPLEMENTATIONS])

    assert means[paper.AX] == 1
    assert means["kornia_cpu"] == pytest.approx(0.60)
    assert means["kornia_gpu"] == pytest.approx(0.9316666667)


def test_mean_uses_the_same_supported_recipes_for_cpu_and_gpu(paper_data: paper._PaperData) -> None:
    paper_data.speed.pop(("kornia_cpu", paper_data.recipes[2].recipe_id))

    means = paper._mean_ratios(paper_data, [paper.AX, "kornia_cpu", "kornia_gpu"])

    assert means == pytest.approx({paper.AX: 1, "kornia_cpu": 0.75, "kornia_gpu": 1.2475})


def test_csv_retains_seed_observations_and_marks_only_the_selected_path(
    tmp_path: Path,
    paper_data: paper._PaperData,
) -> None:
    output = tmp_path / "results.csv"
    paper._write_csv(output, paper_data)
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    second = next(row for row in rows if row["implementation"] == "kornia_gpu" and row["recipe_key"] == "R02")
    selected = next(row for row in rows if row["implementation"] == "kornia_cpu" and row["recipe_key"] == "R02")

    assert len(rows) == 21
    assert second["selected_pairwise_path"] == "False"
    assert selected["selected_pairwise_path"] == "True"
    assert [float(second[f"throughput_images_s_{seed}"]) for seed in (137, 138, 139)] == pytest.approx([1, 99, 99999])
    assert float(second["throughput_median_images_s"]) == pytest.approx(99)
    assert float(selected["gpu_memory_median_mib"]) == 5000


def test_readme_updates_results_and_selection_without_replacing_manual_text(
    paper_data: paper._PaperData,
    selected_run: RunRecord,
) -> None:
    template = "Cite this work.\n<!-- BEGIN GENERATED RESULTS -->\nstale\n<!-- END GENERATED RESULTS -->\nInstructions."
    before = readme.replace_results(template, readme.readme_results(paper_data, selected_run))
    records = [
        record.model_copy(update={"throughput": Throughput(completed_items=8192, duration_seconds=8192 / 200)})
        if record.cell.implementation == "pillow_cpu"
        else record
        for group in paper_data.groups.values()
        for record in group
    ]
    changed = paper._prepare(records, paper_data.recipes)
    after = readme.replace_results(before, readme.readme_results(changed, selected_run))

    assert "| Pillow CPU | test | 0.53\N{MULTIPLICATION SIGN}" in before
    assert "| Pillow CPU | test | 1.07\N{MULTIPLICATION SIGN}" in after
    assert selected_run.run_id[:12] in after
    assert "differs from the published run" in after
    assert "![AX vs Kornia on 3 shared recipes](docs/generated/pairwise-kornia.png)" in after
    assert after.startswith("Cite this work.\n")
    assert after.endswith("\nInstructions.")
    assert readme.replace_results(after, readme.readme_results(changed, selected_run)) == after


def test_selected_run_rejects_missing_foreign_or_tampered_results(
    tmp_path: Path,
    paper_data: paper._PaperData,
    selected_run: RunRecord,
) -> None:
    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog/recipes/rgb.yaml")
    catalog = catalog.model_copy(update={"recipes": paper_data.recipes})
    records = [
        record.model_copy(
            update={
                "run_id": selected_run.run_id,
                "cell": record.cell.model_copy(update={"run_id": selected_run.run_id}),
            },
        )
        for group in paper_data.groups.values()
        for record in group
    ]
    for record in records:
        (tmp_path / f"{record.cell_id}.json").write_bytes(encode_result(record))
    loaded = paper.load_run_results(tmp_path, selected_run, catalog)
    assert {record.cell_id: record for record in loaded} == {record.cell_id: record for record in records}

    missing = tmp_path / f"{records[0].cell_id}.json"
    missing.unlink()
    with pytest.raises(ValueError, match="complete selected run matrix"):
        paper.load_run_results(tmp_path, selected_run, catalog)

    missing.write_bytes(encode_result(records[0]))
    foreign = next(iter(paper_data.groups.values()))[0]
    (tmp_path / f"{foreign.cell_id}.json").write_bytes(encode_result(foreign))
    with pytest.raises(ValueError, match="complete selected run matrix"):
        paper.load_run_results(tmp_path, selected_run, catalog)

    tampered = selected_run.model_copy(update={"hardware": {"machine_type": "different"}})
    with pytest.raises(ValueError, match="immutable identity"):
        paper.load_run_results(tmp_path, tampered, catalog)
