---
name: paper-benchmark-execution
description: Executes the paper benchmark plan for RGB, multichannel, DataLoader, and video benchmarks. Use when the user mentions the paper benchmark, deadline plan, machine matrix, RGB micro, multichannel, DataLoader, video GPU, c4/c4d/g2 machines, or what to run next.
---

# Paper Benchmark Execution

Use `_internal/plans/paper_benchmark_execution_plan.md` as the source of truth.

## Rules

- Use `benchmark/matrix.py` as the source of truth for built-in paper scenario/library/mode support, spec paths,
  requirement groups, paper transform-set files, device support, pipeline scopes, and backend selection.
- Use `benchmark/policy.py` as the source of truth for slow-transform thresholds and media defaults. Do not patch
  separate image/video defaults in micro or DataLoader runners.
- Paper run command construction should flow through `benchmark/jobs.py`, and backend-specific execution should flow
  through `benchmark/orchestrator.py`. Do not add paper-only command branches in `benchmark/cli.py`.
- Do not run every benchmark on every CPU.
- CPU rows run on CPU-only machines, usually `c4-standard-16`.
- GPU rows run only for GPU libraries/paths, usually `g2-standard-16` with L4.
- Do not run CPU-only rows on GPU VMs for hardware symmetry; label hardware per row instead.
- Respect the 32-vCPU quota by running at most two 16-vCPU CPU machines at once.
- Treat RGB micro as a profiler, not the main user-facing training throughput table.
- Keep micro specs native: no `Normalize`, `ToTensor`, axis conversion, or DataLoader collation work in micro rows.
- DataLoader pipeline rows use recipe specs with `Normalize+ToTensor`; the conversion belongs in `*_pipeline_impl.py`,
  not in `pipeline_runner.py`.
- Keep slow-transform preflight enabled for micro and DataLoader runs. Image transforms below the practical floor (`>=0.05 sec/image`, `<=20 img/s`) should early-stop instead of consuming full paper sweep time; these transforms are not usable in practical DataLoader training pipelines.
- DataLoader paper sweeps should default to epoch-based timing (`--min-time 0`) and rely on `--num-runs`, full dataset size, and slow-preflight guards rather than a fixed 30-second minimum per recipe.
- Before cloud runs, reduced local production-path runs should show visible tqdm progress for library loops, media loading, micro transforms, and pipeline transforms. Missing or anonymous progress bars are a benchmark UX bug because long paper sweeps must be diagnosable while running.
- Do not run every transform from `benchmark/transforms/specs.py` for the paper. Use only transforms that exist in at least two selected libraries. The paper transform sets live in `docs/paper_transform_sets/rgb.md`, `docs/paper_transform_sets/9ch.md`, and `docs/paper_transform_sets/video.md`.
- Use `--transform-set paper` for paper micro/pipeline runs unless explicitly testing a smaller transform subset with `--transforms`.
- If paper scenario support changes, update `docs/benchmark_architecture.md`, `docs/benchmark_scope.md`,
  `.cursor/skills/benchmark-runner/SKILL.md`, and matrix/job tests in the same patch.

## Core Matrix

Already done:

- MacBook M4 RGB micro.
- `n2-standard-16` RGB micro.
- `n2d-standard-16` RGB micro.

Core remaining:

- `c4-standard-16`: CPU-only paper tables.
- `c4d-standard-16`: RGB micro AMD sanity check only.
- `g2-standard-16`: torchvision/Kornia/DALI GPU video rows only.

## Required Paper Runs

Main CPU suite on `c4-standard-16` or equivalent modern Intel CPU:

- RGB micro: `--scenario image-rgb --mode micro --libraries albumentationsx torchvision kornia pillow --transform-set paper`.
- 9ch micro: `--scenario image-9ch --mode micro --libraries albumentationsx torchvision kornia --transform-set paper`.
- RGB DataLoader memory: `--scenario image-rgb --mode pipeline --pipeline-scope memory_dataloader_augment`.
- RGB DataLoader decode: `--scenario image-rgb --mode pipeline --pipeline-scope decode_dataloader_augment`.
- 9ch DataLoader memory: `--scenario image-9ch --mode pipeline --pipeline-scope memory_dataloader_augment`.
- 9ch DataLoader decode: `--scenario image-9ch --mode pipeline --pipeline-scope decode_dataloader_augment`.
- Video rows: transforms from `docs/paper_transform_sets/video.md`; run CPU/GPU subsets according to the machine plan.

Recommended final DataLoader flags:

```bash
--batch-size 256 \
--workers 8 \
--num-runs 3 \
--min-time 0 \
--thread-policy pipeline-single-worker
```

AMD sanity on `c4d-standard-16` or equivalent:

- RGB micro only, full selected transform set.
- Optional reduced RGB DataLoader sanity with `--num-items 1000`; do not run the full CPU matrix on AMD unless studying CPU-vendor effects.

GPU/video suite on `g2-standard-16` with L4 or equivalent:

- GPU video micro for `torchvision` and `kornia`.
- GPU video pipeline/DataLoader for GPU-capable paths.
- DALI video pipeline when DALI is available.

Do not rerun CPU-only image rows on GPU machines for hardware symmetry. Label hardware per row instead.

## Execution Order

1. Inventory existing results and avoid rerunning completed `n2`/`n2d` baselines.
2. Run each scenario through the production path with tiny `--num-items`, `--num-runs 1`, and short or zero `--min-time`.
3. Run RGB micro on `c4-standard-16` and `c4d-standard-16`.
4. Run CPU suite on `c4-standard-16`: 9ch micro, RGB DataLoader, 9ch DataLoader, Albumentations video CPU micro.
5. Run GPU suite on `g2-standard-16`: torchvision/Kornia video GPU micro and GPU video DataLoader.
6. Pull and validate artifacts before generating plots/tables.

## Validation

After pulling results, run:

```bash
python -m tools.check_paper_coverage gcp_runs output
```

Use `--require-optional-libraries` only when DALI must be present.
