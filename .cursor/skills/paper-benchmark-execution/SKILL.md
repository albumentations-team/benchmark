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
- Use checked-in YAML configs under `configs/paper/` for paper and GCP runs. Run `python -m benchmark.cli plan --config ...`
  before launch, then use `python -m benchmark.cli run --config ...`; use small overrides such as `--num-items`,
  `--num-runs`, `--device`, `--workers`, `--batch-size`, and `--output` instead of rebuilding long flag commands.
- `BenchmarkRunConfig` in `benchmark/config/models.py` is the typed source of truth for run shape. `resolved_config.yaml`
  and result metadata must contain the resolved config, including expanded paper transform names.
- `benchmark/config/resolve.py` owns YAML loading and CLI overrides, while `benchmark/config/plan.py` owns dry-run job and
  expected-output expansion.
- Keep result filename changes in `benchmark/output_naming.py`, and detached GCP path changes in `benchmark/cloud/paths.py`.
- Paper run command construction should flow through `benchmark/jobs.py`, and backend-specific execution should flow
  through `benchmark/orchestrator.py`. Do not add paper-only command branches in `benchmark/cli.py`.
- Do not run every benchmark on every CPU.
- CPU rows run on CPU-only machines, usually `c4-standard-16`.
- GPU rows run only for GPU libraries/paths, usually `g2-standard-16` with L4.
- Do not run CPU-only rows on GPU VMs for hardware symmetry; label hardware per row instead.
- Respect the current 64-vCPU quota by running at most four 16-vCPU CPU machines at once. The current GPU quota is one
  GPU, so run at most one `g2-standard-16` GPU benchmark VM at a time and remember it also consumes 16 vCPUs.
- Treat RGB micro as a profiler, not the main user-facing training throughput table.
- Keep micro specs native: no `Normalize`, `ToTensor`, axis conversion, or DataLoader collation work in micro rows.
- DataLoader pipeline rows use recipe specs with `Normalize+ToTensor`; the conversion belongs in `*_pipeline_impl.py`,
  not in `pipeline_runner.py`.
- Video DataLoader rows also use dedicated `*_video_pipeline_impl.py` recipe specs. Do not run video DataLoader through
  the transform-only `*_video_impl.py` micro specs.
- Keep slow-transform preflight enabled for micro and DataLoader runs. Image transforms below the practical floor (`>=0.05 sec/image`, `<=20 img/s`) should early-stop instead of consuming full paper sweep time; these transforms are not usable in practical DataLoader training pipelines.
- DataLoader paper sweeps should default to epoch-based timing (`--min-time 0`) and rely on `--num-runs`, full dataset size, and slow-preflight guards rather than a fixed 30-second minimum per recipe.
- Before cloud runs, reduced local production-path runs should show visible tqdm progress for library loops, media loading, micro transforms, and pipeline transforms. Missing or anonymous progress bars are a benchmark UX bug because long paper sweeps must be diagnosable while running.
- Do not run every transform from `benchmark/transforms/specs.py` for the paper. Use only transforms that exist in at least two selected libraries. The paper transform sets live in `docs/paper_transform_sets/rgb.md`, `docs/paper_transform_sets/9ch.md`, and `docs/paper_transform_sets/video.md`.
- Use `--transform-set paper` for paper micro/pipeline runs unless explicitly testing a smaller transform subset with `--transforms`.
- Prefer the checked-in examples over raw commands for current smoke runs:
  - `configs/paper/gcp_c4_rgb_micro_cpu.yaml`
  - `configs/paper/gcp_g2_rgb_gpu_smoke.yaml`
  - `configs/paper/gcp_g2_9ch_gpu_smoke.yaml`
  - `configs/paper/gcp_g2_video_smoke.yaml`
- Use `gs://imagenet_validation/ucf101/ucf101.tar` for paper video cloud runs; uploaded object size is `14136559616` bytes.
- Cloud paper runs should use one dataset tarball per dataset (`val.tar`, `ucf101.tar`) rather than GCS directories full of individual media files. Create tarballs on macOS with `COPYFILE_DISABLE=1`, `tar --no-xattrs`, and excludes for `.DS_Store`, AppleDouble `._*`, and `__MACOSX`; detached GCP staging filters those entries again while extracting.
- If paper scenario support changes, update `docs/benchmark_architecture.md`, `docs/benchmark_scope.md`,
  `.cursor/skills/benchmark-runner/SKILL.md`, config examples, and matrix/config/job tests in the same patch.

## Core Matrix

Already done:

- MacBook M4 RGB micro.
- `n2-standard-16` RGB micro.
- `n2d-standard-16` RGB micro.
- Reduced `g2-standard-16` video smoke:
  - `821ae79852204f5cb4d5bea42fab99b1`: video micro, `torchvision kornia`, `DONE`, `exit_code=0`.
  - `861bd4a840a84ec28ff711f3f68c81a8`: video pipeline CUDA batch-copy smoke,
    `albumentationsx torchvision kornia`, `DONE`, `exit_code=0`.
  - `b7e8cdf6fd154357be68a0b38d134136`: repeat video pipeline CUDA batch-copy smoke,
    `albumentationsx torchvision kornia`, `DONE`, `exit_code=0`.

Core remaining:

- `c4-standard-16`: CPU-only paper tables.
- `c4d-standard-16`: RGB micro AMD sanity check only.
- `g2-standard-16`: final torchvision/Kornia/DALI GPU video rows only.

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

- Video micro on the G2 machine for `albumentationsx`, `torchvision`, and `kornia`, labeled by execution device:
  host CPU for AlbumentationsX, L4 GPU for torchvision/Kornia.
- GPU video pipeline/DataLoader for GPU-capable paths.
- DALI video pipeline when DALI is available.

Do not rerun CPU-only image rows on GPU machines for hardware symmetry. Label hardware per row instead.

## Execution Order

1. Inventory existing results and avoid rerunning completed `n2`/`n2d` baselines.
2. Run each scenario through the production path with tiny `--num-items`, `--num-runs 1`, and short or zero `--min-time`.
3. Run RGB micro on `c4-standard-16` and `c4d-standard-16`.
4. Run CPU suite on `c4-standard-16`: 9ch micro, RGB DataLoader, 9ch DataLoader, Albumentations video CPU micro.
5. Run GPU suite on `g2-standard-16`: AlbumentationsX/torchvision/Kornia video micro and GPU video DataLoader.
6. Pull and validate artifacts before generating plots/tables.

## Validation

After pulling results, run:

```bash
python -m tools.check_paper_coverage gcp_runs output
```

Use `--require-optional-libraries` only when DALI must be present.
