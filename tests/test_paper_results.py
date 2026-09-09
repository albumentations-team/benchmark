import csv
from pathlib import Path

import pytest

from augbench.recipes.load import load_recipe_catalog
from augbench.run_records import CellKey, GpuMemory, OutputObservation, ResultRecord, Throughput
from paper import generate_results as paper


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
