import csv
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest
import yaml
from PIL import Image

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
    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog/recipes/rgb.yaml")
    recipes = (*catalog.recipes[:2], next(recipe for recipe in catalog.recipes if recipe.augmentation_id == "Shear"))
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
    for implementation, _ in (*paper.IMPLEMENTATIONS, paper.PILLOW_ALLSIMD):
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
    assert r"\newcommand{\CropAXBatchRate}{0.8}" in metrics
    assert r"\newcommand{\CropDALIBatchRate}{0.4}" in metrics


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

    assert len(rows) == 24
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
    assert "[per-seed results and manifests](docs/generated/results.json)" in after
    assert "![AX vs Kornia on 3 shared recipes](docs/generated/pairwise-kornia.png)" in after
    assert after.startswith("Cite this work.\n")
    assert after.endswith("\nInstructions.")
    assert readme.replace_results(after, readme.readme_results(changed, selected_run)) == after


def test_ax_rerun_replaces_only_ax_observations(
    tmp_path: Path, paper_data: paper._PaperData, selected_run: RunRecord
) -> None:
    catalog = load_recipe_catalog(Path(__file__).parents[1] / "catalog/recipes/rgb.yaml")
    catalog_path = tmp_path / "rgb.yaml"
    catalog_path.write_text(
        yaml.safe_dump(catalog.model_copy(update={"recipes": paper_data.recipes}).model_dump(mode="json"))
    )
    inputs = selected_run.inputs.model_copy(
        update={"recipe_catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest()}
    )
    run = build_run_record(family_config=selected_run.family_config, inputs=inputs, hardware=selected_run.hardware)
    ax_run = build_run_record(
        family_config={**run.family_config, "implementations": [paper.AX]},
        inputs=inputs,
        hardware=run.hardware,
    )
    run_manifest = tmp_path / "run.json"
    run_manifest.write_text(run.model_dump_json())
    ax_manifest = tmp_path / "ax.json"
    ax_manifest.write_text(ax_run.model_dump_json())
    full_cells = tmp_path / "full"
    full_cells.mkdir()
    ax_cells = tmp_path / "ax"
    ax_cells.mkdir()
    for (implementation, _), group in paper_data.groups.items():
        for record in group:
            _write_observation(full_cells, run, record)
            if implementation == paper.AX:
                _write_observation(ax_cells, ax_run, record)
    args = Namespace(
        run=run_manifest, cells=full_cells, recipes=catalog_path, ax_run=(ax_manifest, ax_cells, catalog_path)
    )

    report, _, replacement = paper.load_report(args, tmp_path)

    assert replacement == ax_run
    assert {(group[0].run_id, implementation) for (implementation, _), group in report.groups.items()} == {
        (ax_run.run_id if implementation == paper.AX else run.run_id, implementation)
        for implementation, _ in paper_data.implementations
    }
    assert {
        record.cell_id for group in report.groups.values() for record in group if record.run_id == ax_run.run_id
    } == {path.stem for path in ax_cells.glob("*.json")}
    mismatched = build_run_record(
        family_config=ax_run.family_config,
        inputs=inputs.model_copy(update={"environment_lock_sha256": {"rgb": "e" * 64}}),
        hardware=ax_run.hardware,
    )
    ax_manifest.write_text(mismatched.model_dump_json())
    with pytest.raises(ValueError, match="same workload, hardware, and package locks"):
        paper.load_report(args, tmp_path)


def _write_observation(directory: Path, run: RunRecord, record: ResultRecord) -> None:
    cell = record.cell.model_copy(update={"run_id": run.run_id})
    updated = record.model_copy(update={"run_id": run.run_id, "cell": cell})
    (directory / f"{cell.cell_id}.json").write_bytes(encode_result(updated))


def test_full_report_rejects_measurement_selection(tmp_path: Path, selected_run: RunRecord) -> None:
    config = {**selected_run.family_config, "measurements": {"Resize224": [paper.AX]}}
    run = build_run_record(family_config=config, inputs=selected_run.inputs, hardware=selected_run.hardware)
    manifest = tmp_path / "run.json"
    manifest.write_text(run.model_dump_json())

    with pytest.raises(ValueError, match="complete RGB run"):
        paper.load_report(Namespace(run=manifest), tmp_path)


@pytest.mark.parametrize("change_torch", [False, True])
def test_torchvision_gpu_replacement_allows_only_addition_of_torchcodec(
    tmp_path: Path, selected_run: RunRecord, *, change_torch: bool
) -> None:
    old_lock = tmp_path / "rgb.lock"
    new_lock = tmp_path / "decoder.lock"
    old_lock.write_bytes(b"torch==2.15.0\ntorchvision==0.29.2\n")
    torch_version = "2.14.0" if change_torch else "2.15.0"
    new_lock.write_bytes(f"torch=={torch_version}\ntorchcodec==0.17.0\ntorchvision==0.29.2\n".encode())
    runs = [
        selected_run.model_copy(
            update={
                "inputs": selected_run.inputs.model_copy(
                    update={"environment_lock_sha256": {"rgb": hashlib.sha256(lock.read_bytes()).hexdigest()}}
                )
            }
        )
        for lock in (old_lock, new_lock)
    ]
    if change_torch:
        with pytest.raises(ValueError, match="sole change"):
            paper._torchcodec_lock_version(*runs, old_lock, new_lock)
    else:
        assert paper._torchcodec_lock_version(*runs, old_lock, new_lock) == "0.17.0"


def test_published_export_flag_excludes_ax_rerun(paper_data: paper._PaperData, selected_run: RunRecord) -> None:
    published = selected_run.model_copy(update={"run_id": paper.RUN_ID})

    assert readme.export_results(paper_data, published)["is_published_run"] is True
    assert readme.export_results(paper_data, published, ax_run=selected_run)["is_published_run"] is False


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


def test_machine_export_uses_selected_run_and_the_readme_aggregates(
    paper_data: paper._PaperData,
    selected_run: RunRecord,
) -> None:
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
    data = paper._prepare(records, paper_data.recipes)
    result = readme.export_results(data, selected_run)

    assert result["run"]["run_id"] == selected_run.run_id
    assert result["is_published_run"] is False
    assert result["measurement_count"] == 72
    assert result["gpu_memory_poll_ms"] == 50
    second = result["recipes"][1]["results"]["kornia_gpu"]
    assert second["throughput"] == pytest.approx({"median": 99, "min": 1, "max": 99999})
    assert second["gpu_memory"] == {"median": 100, "min": 100, "max": 100}
    assert [observation["seed"] for observation in second["observations"]] == [137, 138, 139]
    assert second["observations"][0]["cell_id"] == data.groups[("kornia_gpu", data.recipes[1].recipe_id)][0].cell_id
    common = result["summaries"][0]
    means = {path["id"]: path["mean_relative_throughput"] for path in common["paths"]}
    assert means == paper._mean_ratios(data, [implementation for implementation, _ in data.implementations])
    assert len(common["recipe_ids"]) == 3
    pillow = next(summary for summary in result["summaries"] if summary["id"] == "pillow")
    assert [path["id"] for path in pillow["paths"]] == [paper.AX, "pillow_cpu", "pillow_allsimd_cpu"]
    assert {library["label"] for library in result["libraries"]} >= {"Pillow CPU", "Pillow-AllSIMD CPU (fast)"}


def test_figure_export_records_exact_generated_image_bytes_and_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "common-summary.png"
    captions = {"common-summary": "Throughput and peak GPU memory"}
    Image.new("RGB", (120, 80), "white").save(path)

    figures = readme.export_figures(tmp_path, captions)

    assert figures == {
        "common-summary": {
            "caption": captions["common-summary"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": 120,
            "height": 80,
        }
    }
    Image.new("RGB", (120, 80), "black").save(path)
    assert readme.export_figures(tmp_path, captions)["common-summary"]["sha256"] != figures["common-summary"]["sha256"]
