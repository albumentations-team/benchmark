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
- Respect the current 128-vCPU all-regions quota, 96-vCPU C4-family quota in `us-central1`, and 1-GPU quota. In
  practice, run up to six `c4-standard-16` CPU jobs if no GPU job is active, or up to five C4 jobs plus one
  `g2-standard-16` GPU job. Keep only one G2 job active because L4 quota remains one GPU.
- Respect the current 500 GB Hyperdisk Balanced quota in `us-central1`. Production C4 configs use 100 GB boot disks; a
  200 GB C4 boot disk can block the third parallel C4 VM with `HDB_TOTAL_GB` quota errors.
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
- Prefer the checked-in production configs over raw commands for current paper runs:
  - `configs/paper/prod_c4_rgb_micro_cpu.yaml`
  - `configs/paper/prod_c4_rgb_dataloader_cpu.yaml`
  - `configs/paper/prod_c4_9ch_micro_cpu.yaml`
  - `configs/paper/prod_c4_9ch_dataloader_cpu.yaml`
  - `configs/paper/prod_g2_rgb_micro_gpu.yaml`
  - `configs/paper/prod_g2_rgb_dataloader_gpu.yaml`
  - `configs/paper/prod_g2_9ch_micro_gpu.yaml`
  - `configs/paper/prod_g2_9ch_dataloader_gpu.yaml`
  Keep `gcp_*_smoke.yaml` configs for fast path checks and reruns only.
  - `configs/paper/gcp_g2_video_smoke.yaml`
- Use `gs://imagenet_validation/ucf101/ucf101.tar` for paper video cloud runs; uploaded object size is `14136559616` bytes.
- Cloud paper runs should use one dataset tarball per dataset (`val.tar`, `ucf101.tar`) rather than GCS directories full of individual media files. Create tarballs on macOS with `COPYFILE_DISABLE=1`, `tar --no-xattrs`, and excludes for `.DS_Store`, AppleDouble `._*`, and `__MACOSX`; detached GCP staging filters those entries again while extracting.
- `benchmark/cloud/stage_dataset.py` runs in the VM bootstrap before the control venv exists. Keep it stdlib-only: no
  Pydantic, no `benchmark.config`, and no imports that require benchmark dependencies.
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

- RGB micro: start from `configs/paper/prod_c4_rgb_micro_cpu.yaml`.
- 9ch micro: start from `configs/paper/prod_c4_9ch_micro_cpu.yaml`.
- RGB DataLoader memory: start from `configs/paper/prod_c4_rgb_dataloader_cpu.yaml`.
- 9ch DataLoader memory: start from `configs/paper/prod_c4_9ch_dataloader_cpu.yaml`.
- Video rows: transforms from `docs/paper_transform_sets/video.md`; run CPU/GPU subsets according to the machine plan.

Deadline-first DataLoader config fields:

```yaml
execution:
  batch_size: 256  # RGB; use 128 for 9-channel
  workers: 8
  num_runs: 1
  min_time: 0
  thread_policy: pipeline-default
```

Use `batch_size: 128` for all 9-channel DataLoader libraries. The first 9-channel CPU DataLoader attempt at
`batch_size: 256` OOM-killed Kornia, so partial `b256` 9-channel rows are exploratory and should not be mixed into the
main table.

After the full one-run matrix is covered and validated, add two more runs for important rows and aggregate them into
3-run statistics. Use 5 total runs only for high-variance or close-call conclusions.

AMD sanity on `c4d-standard-16` or equivalent:

- RGB micro only, full selected transform set.
- Optional reduced RGB DataLoader sanity with `--num-items 1000`; do not run the full CPU matrix on AMD unless studying CPU-vendor effects.

GPU/video suite on `g2-standard-16` with L4 or equivalent:

- GPU image micro and DataLoader production rows for `torchvision` and `kornia` on RGB and 9-channel images, starting
  from the `prod_g2_*` configs.
- Video micro on the G2 machine for `albumentationsx`, `torchvision`, and `kornia`, labeled by execution device:
  host CPU for AlbumentationsX, L4 GPU for torchvision/Kornia.
- GPU video pipeline/DataLoader for GPU-capable paths.
- DALI video pipeline when DALI is available.

Kornia image GPU rows intentionally exclude `Shear` in micro and DataLoader modes because Kornia's current CUDA shear
parameter generator can fail with mixed CPU/CUDA tensors when moved to GPU. Keep `Shear` in the global RGB/9-channel paper
transform sets for AlbumentationsX, Pillow, torchvision where supported, and Kornia CPU rows; mention this as a benchmark
methodology limitation.

TorchVision image GPU DataLoader rows use a per-sample GPU loop for the measured transform, then batch normalization,
because TorchVision v2 does not expose a `same_on_batch=False` equivalent for per-image random parameters in batched image
transforms. Label this explicitly in paper tables.

TorchVision `JpegCompression` uses `torchvision.transforms.v2.JPEG`, which requires `uint8` CPU input. Exclude it from
TorchVision GPU image rows; keep it in CPU TorchVision rows and other libraries that support it. Mention this
JPEG-compression device constraint when interpreting GPU tables.

CUDA DataLoader rows record per-transform peak GPU memory during timed runs. Use `gpu_memory.peak_allocated_bytes` and
`gpu_memory.peak_reserved_bytes` as paper-facing cost columns for GPU augmentation. Pyperf micro rows do not report peak
memory because pyperf executes timed loops in worker processes.

Do not rerun CPU-only image rows on GPU machines for hardware symmetry. Label hardware per row instead.

## Execution Order

1. Inventory existing results and avoid rerunning completed `n2`/`n2d` baselines.
2. Run production Wave 1 RGB: CPU micro plus GPU micro, then CPU DataLoader plus GPU DataLoader.
3. Run production Wave 2 9-channel: CPU micro plus GPU micro, then CPU DataLoader plus GPU DataLoader.
4. Pull and validate artifacts after each run before starting top-up repeats.
5. Run video production sizing only after image tables are secured, unless video becomes central to the paper claim.
6. Pull and validate artifacts before generating plots/tables.

## Validation

After pulling results, run:

```bash
python -m tools.check_paper_coverage gcp_runs output
```

Use `--require-optional-libraries` only when DALI must be present.
