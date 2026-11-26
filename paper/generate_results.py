# ruff: noqa: INP001
"""Generate the RGB paper and its audit table from validated production runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import TYPE_CHECKING

import yaml

from augbench.matrix import build_matrix
from augbench.recipes.load import load_recipe_catalog
from augbench.result_store import decode_result
from augbench.run_config import FamilyRunConfig
from augbench.run_records import RunRecord, build_run_record
from paper import figures
from paper.intervals import RatioInterval, median_interval, ratio_interval

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from augbench.recipes.models import RecipeCatalog, RecipeSpec
    from augbench.run_records import ResultRecord

RUN_ID = "3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c"
INVALID_DALI_FLIP_RUN = "75bc062fe770cfec3d857f401e16366deb01f6582c8645e952930e3cd3d788fb"
AX = "albumentationsx_cpu"
TORCHVISION_GPU = "torchvision_gpu"
PILLOW_ALLSIMD = ("pillow_allsimd_cpu", "Pillow-AllSIMD CPU (fast)")
IMPLEMENTATIONS = (
    (AX, "AlbumentationsX"),
    ("pillow_cpu", "Pillow CPU"),
    ("torchvision_cpu", "TorchVision CPU"),
    ("torchvision_gpu", "TorchVision GPU"),
    ("kornia_cpu", "Kornia CPU"),
    ("kornia_gpu", "Kornia GPU"),
    ("dali_gpu", "DALI GPU"),
)
PAIRWISE = (
    ("Kornia", ("kornia_cpu", "kornia_gpu")),
    ("TorchVision", ("torchvision_cpu", "torchvision_gpu")),
    ("Pillow", ("pillow_cpu",)),
    ("DALI", ("dali_gpu",)),
)
_COMMON_RECIPE_RATIO_PATHS = (
    ("dali_gpu", "DALI"),
    ("kornia_cpu", "Kornia CPU"),
    ("kornia_gpu", "Kornia GPU"),
    ("pillow_cpu", "Pillow"),
    (PILLOW_ALLSIMD[0], "Pillow-AllSIMD"),
    ("torchvision_cpu", "TorchVision CPU"),
    ("torchvision_gpu", "TorchVision GPU"),
)
RGB_2D_COVERAGE_DOMAINS = frozenset({"dropout_or_multi_image", "geometry", "pixel"})


@dataclass(frozen=True)
class _PaperData:
    groups: dict[tuple[str, str], list[ResultRecord]]
    recipes: tuple[RecipeSpec, ...]
    speed: dict[tuple[str, str], float]
    memory: dict[tuple[str, str], float]
    implementations: tuple[tuple[str, str], ...]
    comparisons: tuple[tuple[str, tuple[str, ...]], ...]
    common: tuple[str, ...]
    supplemental_runs: tuple[RunRecord, ...] = ()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    data, run, ax_run = load_report(args, root)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_metrics(args.output / "metrics.tex", data)
    write_figures(args.output, data)
    _write_csv(args.output / "recipe-results.csv", data)
    _write_recipe_catalog(args.output / "recipes.tex", data.recipes)
    _write_recipe_results(args.output / "recipe-results.tex", data)
    _write_versions(args.output / "versions.tex", data)
    _write_coverage(args.output, root / "catalog/operations.yaml")
    _write_sources(args.output / "sources.tex", run, ax_run, data.supplemental_runs)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="immutable run.json for the full RGB run")
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, help="full run's catalog; defaults to the repository catalog")
    parser.add_argument(
        "--ax-run", nargs=3, type=Path, metavar=("MANIFEST", "CELLS", "RECIPES"), help="optional AX-only rerun"
    )
    parser.add_argument(
        "--ax-locks",
        nargs=2,
        type=Path,
        metavar=("PRIMARY_LOCK", "AX_LOCK"),
        help="RGB locks of the full run and the AX rerun when they differ; only the AlbumentationsX entry may change",
    )
    parser.add_argument(
        "--supplemental-run",
        nargs=3,
        action="append",
        type=Path,
        metavar=("MANIFEST", "CELLS", "RECIPES"),
        help="another immutable run with complementary implementation cells",
    )
    parser.add_argument(
        "--torchvision-gpu-run",
        nargs=5,
        type=Path,
        metavar=("MANIFEST", "CELLS", "RECIPES", "PRIMARY_LOCK", "DECODER_LOCK"),
        help="replace all TorchVision GPU cells with a TorchCodec run and verify both RGB locks",
    )


def _run_catalog(run: RunRecord, path: Path) -> RecipeCatalog:
    if hashlib.sha256(path.read_bytes()).hexdigest() != run.inputs.recipe_catalog_sha256:
        raise ValueError(f"{path} does not match the recipe catalog checksum for {run.run_id}")
    return load_recipe_catalog(path)


def load_report(args: argparse.Namespace, root: Path) -> tuple[_PaperData, RunRecord, RunRecord | None]:
    run = RunRecord.model_validate_json(args.run.read_bytes())
    config = FamilyRunConfig.model_validate(run.family_config)
    base = {implementation for implementation, _ in IMPLEMENTATIONS}
    if config.measurements is not None or set(config.implementations) not in (base, base | {PILLOW_ALLSIMD[0]}):
        raise ValueError("report requires a complete RGB run")
    if run.run_id == INVALID_DALI_FLIP_RUN:
        raise ValueError("this run contains incorrect DALI VerticalFlip measurements")
    recipes = _run_catalog(run, args.recipes or root / config.recipes)
    supplemental_inputs = getattr(args, "supplemental_run", None) or []
    ax_run_inputs = getattr(args, "ax_run", None)
    if supplemental_inputs and ax_run_inputs:
        raise ValueError("supplemental implementation runs and an AX replacement run cannot be combined")
    records = load_run_results(args.cells, run, recipes, allow_partial=bool(supplemental_inputs))
    supplemental_runs = []
    for manifest, cells, catalog_path in supplemental_inputs:
        supplemental = RunRecord.model_validate_json(manifest.read_bytes())
        _require_compatible_workload(supplemental, run)
        if supplemental.inputs.environment_lock_sha256 != run.inputs.environment_lock_sha256:
            raise ValueError("complementary runs must use the same package locks")
        supplemental_catalog = _run_catalog(supplemental, catalog_path)
        records.extend(load_run_results(cells, supplemental, supplemental_catalog, allow_partial=True))
        supplemental_runs.append(supplemental)
    if supplemental_runs:
        _validate_combined_matrix(records, run, recipes)
    if torchvision_inputs := getattr(args, "torchvision_gpu_run", None):
        decoder_run, decoder_records = _load_torchvision_gpu_run(torchvision_inputs, run, recipes)
        records = [record for record in records if record.cell.implementation != TORCHVISION_GPU] + decoder_records
        _validate_combined_matrix(records, run, recipes)
        supplemental_runs.append(decoder_run)
    ax_run = None
    if ax_run_inputs:
        ax_run, ax_records = _load_ax_run(ax_run_inputs, run, getattr(args, "ax_locks", None))
        records = [record for record in records if record.cell.implementation != AX] + ax_records
    data = _prepare(records, recipes.recipes, tuple(supplemental_runs))
    if not data.common:
        raise ValueError("the selected results have no recipes shared by all paths")
    return data, run, ax_run


def _require_compatible_workload(supplemental: RunRecord, primary: RunRecord) -> None:
    primary_config = {key: value for key, value in primary.family_config.items() if key != "measurements"}
    supplemental_config = {key: value for key, value in supplemental.family_config.items() if key != "measurements"}
    if (
        supplemental_config != primary_config
        or supplemental.hardware != primary.hardware
        or supplemental.inputs.dataset_archive_sha256 != primary.inputs.dataset_archive_sha256
        or supplemental.inputs.recipe_catalog_sha256 != primary.inputs.recipe_catalog_sha256
    ):
        raise ValueError("source runs must use the same workload, hardware, and recipes")


def _load_torchvision_gpu_run(
    inputs: Sequence[Path], primary: RunRecord, primary_catalog: RecipeCatalog
) -> tuple[RunRecord, list[ResultRecord]]:
    manifest, cells, catalog, primary_lock, decoder_lock = inputs
    replacement = RunRecord.model_validate_json(manifest.read_bytes())
    _require_compatible_workload(replacement, primary)
    expected = {
        recipe.recipe_id: [TORCHVISION_GPU]
        for recipe in primary_catalog.recipes
        if TORCHVISION_GPU in recipe.supported_implementations
    }
    if replacement.family_config.get("measurements") != expected:
        raise ValueError("replacement must select exactly all TorchVision GPU recipes")

    version = _torchcodec_lock_version(primary, replacement, primary_lock, decoder_lock)
    records = load_run_results(cells, replacement, _run_catalog(replacement, catalog))
    if any(record.runtime.get("jpeg_decoder_version") != version for record in records):
        raise ValueError("TorchVision GPU cells must record the locked TorchCodec version")
    return replacement, records


def _torchcodec_lock_version(primary: RunRecord, replacement: RunRecord, primary_lock: Path, decoder_lock: Path) -> str:
    base_hashes = primary.inputs.environment_lock_sha256
    decoder_hashes = replacement.inputs.environment_lock_sha256
    original = primary_lock.read_bytes()
    updated = decoder_lock.read_bytes()
    if (
        hashlib.sha256(original).hexdigest() != base_hashes.get("rgb")
        or hashlib.sha256(updated).hexdigest() != decoder_hashes.get("rgb")
        or {key: value for key, value in base_hashes.items() if key != "rgb"}
        != {key: value for key, value in decoder_hashes.items() if key != "rgb"}
    ):
        raise ValueError("supplied RGB locks must match the manifests and other package locks must agree")

    matches = list(re.finditer(rb"(?m)^torchcodec==([^\s]+)[^\n]*\n(?:[ \t]+[^\n]*\n)*", updated))
    if len(matches) != 1:
        raise ValueError("replacement RGB lock must add exactly one TorchCodec entry")
    match = matches[0]
    if updated[: match.start()] + updated[match.end() :] != original:
        raise ValueError("TorchCodec must be the sole change to the RGB lock")
    return match[1].decode()


def _validate_combined_matrix(records: Sequence[ResultRecord], run: RunRecord, recipes: RecipeCatalog) -> None:
    config = FamilyRunConfig.model_validate(run.family_config)
    expected = {
        (cell.implementation, cell.recipe_id, cell.seed)
        for cell in build_matrix(run_id=run.run_id, config=config, recipes=recipes)
    }
    found = [(record.cell.implementation, record.cell.recipe_id, record.cell.seed) for record in records]
    if set(found) != expected or len(found) != len(expected):
        raise ValueError("combined source runs do not form one complete RGB matrix without duplicate cells")


def _load_ax_run(
    inputs: Sequence[Path], run: RunRecord, locks: Sequence[Path] | None = None
) -> tuple[RunRecord, list[ResultRecord]]:
    """Replace the AX cells with an AX-only rerun of the same workload.

    Each implementation runs in its own process, so a newer AX release leaves the other cells valid. The two RGB
    locks may differ only in the AlbumentationsX entry, and the rerun's cells must record that version.
    """
    manifest, cells, catalog = inputs
    ax_run = RunRecord.model_validate_json(manifest.read_bytes())
    if (
        ax_run.family_config != {**run.family_config, "implementations": [AX]}
        or ax_run.hardware != run.hardware
        or ax_run.inputs.dataset_archive_sha256 != run.inputs.dataset_archive_sha256
        or ax_run.inputs.recipe_catalog_sha256 != run.inputs.recipe_catalog_sha256
    ):
        raise ValueError("AX rerun must use the same workload, hardware, dataset, and recipes")
    version = None
    if ax_run.inputs.environment_lock_sha256 != run.inputs.environment_lock_sha256:
        if locks is None:
            raise ValueError("AX rerun uses another lock; pass --ax-locks so the generator can check the difference")
        version = _ax_lock_version(run, ax_run, *locks)
    records = load_run_results(cells, ax_run, _run_catalog(ax_run, catalog))
    if version is not None and any(record.runtime.get("library_version") != version for record in records):
        raise ValueError(f"AX rerun cells must record the locked AlbumentationsX {version}")
    return ax_run, records


def _ax_lock_version(primary: RunRecord, replacement: RunRecord, primary_lock: Path, ax_lock: Path) -> str:
    base_hashes = primary.inputs.environment_lock_sha256
    ax_hashes = replacement.inputs.environment_lock_sha256
    original = primary_lock.read_bytes()
    updated = ax_lock.read_bytes()
    if (
        hashlib.sha256(original).hexdigest() != base_hashes.get("rgb")
        or hashlib.sha256(updated).hexdigest() != ax_hashes.get("rgb")
        or {key: value for key, value in base_hashes.items() if key != "rgb"}
        != {key: value for key, value in ax_hashes.items() if key != "rgb"}
    ):
        raise ValueError("supplied RGB locks must match the manifests and other package locks must agree")
    entry = re.compile(rb"(?m)^albumentationsx==([^\s]+)[^\n]*\n(?:[ \t]+[^\n]*\n)*")
    before, after = list(entry.finditer(original)), list(entry.finditer(updated))
    if len(before) != 1 or len(after) != 1:
        raise ValueError("each RGB lock must pin AlbumentationsX exactly once")
    if entry.sub(b"", original) != entry.sub(b"", updated):
        raise ValueError("AlbumentationsX must be the only change between the RGB locks")
    return after[0][1].decode()


def load_run_results(
    cells_directory: Path, run: RunRecord, recipes: RecipeCatalog, *, allow_partial: bool = False
) -> list[ResultRecord]:
    identity = build_run_record(family_config=run.family_config, inputs=run.inputs, hardware=run.hardware)
    if identity != run:
        raise ValueError("run manifest does not match its immutable identity")
    cells = build_matrix(run_id=run.run_id, config=FamilyRunConfig.model_validate(run.family_config), recipes=recipes)
    records = [
        decode_result(path.read_bytes(), expected_cell_id=path.stem)
        for path in sorted(cells_directory.glob("*.json"))
        if path.name != "run.json"
    ]
    found = {record.cell_id for record in records}
    expected = {cell.cell_id for cell in cells}
    if (not allow_partial and found != expected) or (allow_partial and not found <= expected):
        raise ValueError("results do not match the complete selected run matrix")
    config = run.family_config
    output = config["output"]
    execution = config["execution"]
    expected_shape = (execution["batch_size"], output["channels"], output["height"], output["width"])
    expected_items = execution["batch_size"] * execution["measured_batches"]
    for record in records:
        if record.output.shape != expected_shape or record.throughput.completed_items != expected_items:
            raise ValueError(f"cell {record.cell_id} violates the frozen output or timing window")
        if not math.isfinite(record.throughput.value) or not math.isfinite(record.gpu_memory.peak_mib):
            raise ValueError(f"cell {record.cell_id} has a non-finite metric")
    return records


def _prepare(
    records: Sequence[ResultRecord],
    recipes: tuple[RecipeSpec, ...],
    supplemental_runs: tuple[RunRecord, ...] = (),
) -> _PaperData:
    groups: dict[tuple[str, str], list[ResultRecord]] = defaultdict(list)
    for record in records:
        groups[(record.cell.implementation, record.cell.recipe_id)].append(record)
    for group in groups.values():
        group.sort(key=lambda record: record.cell.seed)
    speed = {key: median(record.throughput.value for record in group) for key, group in groups.items()}
    memory = {key: median(record.gpu_memory.peak_mib for record in group) for key, group in groups.items()}
    support: dict[str, set[str]] = defaultdict(set)
    for implementation, recipe in groups:
        support[implementation].add(recipe)
    implementations: tuple[tuple[str, str], ...] = IMPLEMENTATIONS
    comparisons: tuple[tuple[str, tuple[str, ...]], ...] = PAIRWISE
    if PILLOW_ALLSIMD[0] in support:
        implementations = (*implementations, PILLOW_ALLSIMD)
        comparisons = tuple(
            (family, (*paths, PILLOW_ALLSIMD[0]) if family == "Pillow" else paths) for family, paths in comparisons
        )
    common = set.intersection(*(support[implementation] for implementation, _ in implementations))
    ordered = tuple(recipe.recipe_id for recipe in recipes if recipe.recipe_id in common)
    return _PaperData(
        dict(groups),
        recipes,
        speed,
        memory,
        implementations,
        comparisons,
        ordered,
        supplemental_runs,
    )


def _write_tex(path: Path, lines: Iterable[str]) -> None:
    path.write_text("% Generated by paper/generate_results.py. Do not edit by hand.\n" + "\n".join(lines) + "\n")


def _tex(value: str) -> str:
    replacements = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(character, character) for character in value)


def _name(value: str) -> str:
    parts = re.split(r"(?<=[a-z])(?=[A-Z])|(?<=[._/=(),])", value)
    return r"\allowbreak{}".join(_tex(part) for part in parts)


def _recipe_key(recipe: str, recipes: Sequence[RecipeSpec]) -> str:
    return f"R{next(index for index, item in enumerate(recipes, 1) if item.recipe_id == recipe):02d}"


def _command(implementation: str) -> str:
    return "AllSIMD" if implementation == PILLOW_ALLSIMD[0] else dict(IMPLEMENTATIONS)[implementation].replace(" ", "")


def _write_metrics(path: Path, data: _PaperData) -> None:
    records = [record for group in data.groups.values() for record in group]
    batch_size = _single({record.output.shape[0] for record in records})
    metrics = {
        "ProductionCellCount": f"{len(records):,}",
        "ProductionPairCount": str(len(data.groups)),
        "RecipeCount": str(len(data.recipes)),
        "CommonRecipeCount": str(len(data.common)),
        "SeedCount": str(_single({len(group) for group in data.groups.values()})),
        "BatchSize": str(batch_size),
        "MemoryPollMs": str(_single({record.gpu_memory.poll_interval_ms for record in records})),
        "CommonDALIFastest": str(
            sum(
                all(
                    data.speed["dali_gpu", recipe] > data.speed[other, recipe]
                    for other, _ in data.implementations
                    if other != "dali_gpu"
                )
                for recipe in data.common
            )
        ),
    }
    spreads = sorted(
        (max(speeds) - min(speeds)) / median(speeds) for speeds in (_speeds(data, key) for key in data.groups)
    )
    metrics["SeedSpreadMedian"] = f"{100 * median(spreads):.1f}"
    metrics["SeedSpreadNinety"] = f"{100 * spreads[int(0.9 * (len(spreads) - 1))]:.0f}"
    for implementation, _ in data.implementations:
        command = _command(implementation)
        speed = median(data.speed[implementation, recipe] for recipe in data.common)
        metrics[f"CommonSpeed{command}"] = f"{speed:,.0f}"
        metrics[f"CommonBatchRate{command}"] = f"{speed / batch_size:.1f}"
        metrics[f"CommonMemory{command}"] = (
            f"{median(data.memory[implementation, recipe] for recipe in data.common):,.0f}"
        )
        baselines = [
            record.gpu_memory.baseline_mib
            for recipe in data.common
            for record in data.groups[implementation, recipe]
            if record.gpu_memory.baseline_mib is not None
        ]
        if baselines:
            metrics[f"CommonBaseline{command}"] = f"{median(baselines):,.0f}"
        metrics[f"Recipes{command}"] = str(sum(key[0] == implementation for key in data.groups))
        if implementation == AX:
            continue
        shared = _shared_recipes(data, [AX, implementation])
        verdicts = [ratio_to_ax(data, implementation, recipe).verdict for recipe in shared]
        metrics[f"Shared{command}"] = str(len(shared))
        metrics[f"Faster{command}"] = str(verdicts.count("faster"))
        metrics[f"Tied{command}"] = str(verdicts.count("indistinguishable"))
        metrics[f"Slower{command}"] = str(verdicts.count("slower"))
        metrics[f"MedianRatio{command}"] = (
            f"{median(data.speed[implementation, r] / data.speed[AX, r] for r in shared):.2f}"
        )
    if PILLOW_ALLSIMD in data.implementations:
        recipes, ratios = pillow_allsimd_ratios(data)
        metrics["AllSIMDShared"] = str(len(recipes))
        metrics["AllSIMDMedianRatio"] = f"{median(ratios):.2f}"
        blur = next((recipe.recipe_id for recipe in data.recipes if recipe.augmentation_id == "MedianBlur"), None)
        if blur is not None and blur in recipes:
            metrics["AllSIMDMedianBlurRatio"] = f"{ratios[recipes.index(blur)]:.1f}"
    _write_tex(path, (rf"\newcommand{{\{command}}}{{{value}}}" for command, value in metrics.items()))


def write_figures(output: Path, data: _PaperData) -> None:
    """Write the PGFPlots sources: the overview, every path against AX, and one AX comparison per library family."""
    (output / "fig-rgb-overview.tex").write_text(figures.rgb_overview(data))
    (output / "fig-ax-pairs.tex").write_text(figures.ax_pairs(data))
    for family, paths in data.comparisons:
        (output / f"fig-pairwise-{family.lower()}.tex").write_text(figures.ax_pairs(data, paths))


def _shared_recipes(data: _PaperData, implementations: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        recipe.recipe_id
        for recipe in data.recipes
        if all((implementation, recipe.recipe_id) in data.speed for implementation in implementations)
    )


def _speeds(data: _PaperData, key: tuple[str, str]) -> list[float]:
    return [record.throughput.value for record in data.groups[key]]


def ratio_to_ax(data: _PaperData, implementation: str, recipe: str) -> RatioInterval:
    """Throughput ratio of one path to AX on one recipe, with its exact interval over seeds."""
    return ratio_interval(_speeds(data, (implementation, recipe)), _speeds(data, (AX, recipe)))


def pillow_allsimd_ratios(data: _PaperData) -> tuple[tuple[str, ...], list[float]]:
    recipes = _shared_recipes(data, ["pillow_cpu", PILLOW_ALLSIMD[0]])
    return recipes, [data.speed[(PILLOW_ALLSIMD[0], recipe)] / data.speed[("pillow_cpu", recipe)] for recipe in recipes]


def _mean_ratios(data: _PaperData, implementations: Sequence[str]) -> dict[str, float]:
    recipes = _shared_recipes(data, implementations)
    return {
        implementation: fmean(data.speed[(implementation, recipe)] / data.speed[(AX, recipe)] for recipe in recipes)
        for implementation in implementations
    }


def _common_memories(data: _PaperData) -> dict[str, float]:
    return {
        implementation: median(data.memory[(implementation, recipe)] for recipe in data.common)
        for implementation, _ in data.implementations
    }


def _single[T](values: set[T]) -> T:
    if len(values) != 1:
        raise ValueError(f"expected one frozen value, got {values}")
    return next(iter(values))


def _write_versions(path: Path, data: _PaperData) -> None:
    lines = []
    for implementation, label in data.implementations:
        versions = {
            r.runtime["library_version"]
            for (impl, _), group in data.groups.items()
            if impl == implementation
            for r in group
        }
        lines.append(rf"{label} & \texttt{{{_tex(_single(versions))}}} \\")
    _write_tex(path, lines)


def _write_sources(
    path: Path, run: RunRecord, ax_run: RunRecord | None, supplemental_runs: Sequence[RunRecord] = ()
) -> None:
    lines: list[str] = []
    for label, source in (("Full", run), ("AX", ax_run or run)):
        lines.extend(
            (
                rf"\newcommand{{\{label}RunID}}{{{source.run_id}}}",
                rf"\newcommand{{\{label}SourceRevision}}{{{source.inputs.git_commit}}}",
            )
        )
    details = [
        rf"\item[Correction {index}] \texttt{{\expandafter\seqsplit\expandafter{{{source.run_id}}}}}\\"
        rf"Source \texttt{{\expandafter\seqsplit\expandafter{{{source.inputs.git_commit}}}}}."
        for index, source in enumerate(supplemental_runs, start=1)
    ] or [r"\item[Correction runs] none"]
    details_tex = "\n".join(details)
    lines.append(rf"\newcommand{{\SupplementalRunDetails}}{{{details_tex}}}")
    _write_tex(path, lines)


def _interval_columns(data: _PaperData, implementation: str, recipe: str) -> dict[str, object]:
    interval = median_interval(_speeds(data, (implementation, recipe)))
    columns: dict[str, object] = {
        "throughput_interval_low_images_s": interval.low,
        "throughput_interval_high_images_s": interval.high,
        "throughput_interval_confidence": interval.confidence,
    }
    versus = None if implementation == AX else ratio_to_ax(data, implementation, recipe)
    columns.update(
        ratio_to_ax=versus.ratio if versus else "",
        ratio_to_ax_low=versus.low if versus else "",
        ratio_to_ax_high=versus.high if versus else "",
        ratio_to_ax_confidence=versus.confidence if versus else "",
        ratio_to_ax_verdict=versus.verdict if versus else "",
    )
    return columns


def _csv_row(group: list[ResultRecord], recipe: RecipeSpec, data: _PaperData) -> dict[str, object]:
    implementation = group[0].cell.implementation
    key = (implementation, recipe.recipe_id)
    run_ids = {record.run_id for record in group}
    row: dict[str, object] = {
        "run_id": _single(run_ids) if len(run_ids) == 1 else "",
        "implementation": implementation,
        "library_version": group[0].runtime["library_version"],
        "recipe_key": _recipe_key(recipe.recipe_id, data.recipes),
        "recipe_id": recipe.recipe_id,
        "common_intersection": recipe.recipe_id in data.common,
        "throughput_median_images_s": data.speed[key],
        **_interval_columns(data, implementation, recipe.recipe_id),
        "gpu_memory_median_mib": data.memory[key],
        "recipe_stages_json": json.dumps(
            [stage.model_dump(mode="json") for stage in recipe.stages], separators=(",", ":")
        ),
    }
    for record in group:
        seed = record.cell.seed
        row.update(
            {
                f"run_id_{seed}": record.run_id,
                f"cell_id_{seed}": record.cell_id,
                f"completed_images_{seed}": record.throughput.completed_items,
                f"duration_seconds_{seed}": record.throughput.duration_seconds,
                f"throughput_images_s_{seed}": record.throughput.value,
                f"gpu_memory_peak_mib_{seed}": record.gpu_memory.peak_mib,
                f"gpu_memory_baseline_mib_{seed}": record.gpu_memory.baseline_mib,
                f"gpu_memory_poll_ms_{seed}": record.gpu_memory.poll_interval_ms,
            }
        )
    return row


def _write_csv(path: Path, data: _PaperData) -> None:
    rows = [
        _csv_row(data.groups[(implementation, recipe.recipe_id)], recipe, data)
        for implementation, _ in data.implementations
        for recipe in data.recipes
        if (implementation, recipe.recipe_id) in data.groups
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parameters(parameters: Mapping[str, object]) -> str:
    return (
        "; ".join(rf"\texttt{{{_tex(key)}}}={_tex(json.dumps(value))}" for key, value in parameters.items())
        or "no parameters"
    )


def _write_recipe_catalog(path: Path, recipes: Sequence[RecipeSpec]) -> None:
    lines = []
    for recipe in recipes:
        key = _recipe_key(recipe.recipe_id, recipes)
        stages = recipe.stages[:-2]
        sequence = r" $\to$ ".join(_name(stage.operation_id) for stage in stages)
        details = [
            rf"\textbf{{{_name(stage.operation_id)}}}: {_parameters(stage.parameters)}. "
            rf"Sampling: {_tex(stage.randomness_scope)}."
            for stage in stages
        ]
        lines.append(rf"\hypertarget{{{key}}}{{\textbf{{{key}}}}} {sequence} & " + r"\newline ".join(details) + r" \\")
    _write_tex(path, lines)


def _write_recipe_results(path: Path, data: _PaperData) -> None:
    lines: list[str] = []
    for implementation, label in data.implementations:
        lines.extend(
            (
                rf"\subsection{{{label}}}",
                r"\begin{longtable}{lrrrrrr}",
                (
                    r"\textbf{Recipe} & \multicolumn{3}{c}{\textbf{Throughput (images/s)}} & "
                    r"\multicolumn{3}{c}{\textbf{GPU memory (MiB)}} \\"
                ),
                r" & Median & Min & Max & Median & Min & Max \\ \midrule \endhead",
            )
        )
        for recipe in data.recipes:
            group = data.groups.get((implementation, recipe.recipe_id))
            if group is None:
                continue
            speed = [record.throughput.value for record in group]
            memory = [record.gpu_memory.peak_mib for record in group]
            key = _recipe_key(recipe.recipe_id, data.recipes)
            values = [median(speed), min(speed), max(speed), median(memory), min(memory), max(memory)]
            lines.append(rf"\hyperlink{{{key}}}{{{key}}} & " + " & ".join(f"{v:,.0f}" for v in values) + r" \\")
        lines.append(r"\bottomrule\end{longtable}")
    _write_tex(path, lines)


def _write_coverage(output: Path, path: Path) -> None:
    catalog = yaml.safe_load(path.read_text())
    operations = [item for item in catalog["operations"] if item["domain"] in RGB_2D_COVERAGE_DOMAINS]
    names = {
        "albumentationsx": "AlbumentationsX",
        "kornia": "Kornia",
        "torchvision": "TorchVision",
        "pillow": "Pillow",
        "dali": "DALI",
    }
    counts = dict.fromkeys(names, 0)
    counts["albumentationsx"] = len(operations)
    mappings = []
    for operation in operations:
        cells = [_name(operation["operation_id"])]
        for library in ("kornia", "torchvision", "pillow", "dali"):
            match = operation["support"].get(library, {"api": None, "status": "none"})
            status = match["status"]
            if status not in {"direct", "composed", "approximate", "none"}:
                raise ValueError(f"unexpected {library} coverage status: {status}")
            counts[library] += status in {"direct", "composed"}
            suffix = {"composed": " [composed]", "approximate": " [approx.]"}.get(status, "")
            cells.append(_name(match["api"] or "---") + suffix)
        mappings.append(" & ".join(cells) + r" \\")
    _write_tex(
        output / "coverage.tex",
        (rf"{names[lib]} & {count} & {catalog['versions'][lib]} \\" for lib, count in counts.items()),
    )
    _write_tex(output / "coverage-mapping.tex", mappings)
    _write_tex(
        output / "coverage-metrics.tex",
        [
            *(rf"\newcommand{{\Coverage{names[lib]}}}{{{count}}}" for lib, count in counts.items()),
            rf"\newcommand{{\CoverageCatalogSHA}}{{{hashlib.sha256(path.read_bytes()).hexdigest()}}}",
        ],
    )


if __name__ == "__main__":
    main()
