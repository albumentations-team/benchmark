# Paper Benchmark Execution Plan

## Strategy

Do not run every benchmark on every CPU. The main paper needs complete, well-labeled tables across the seven benchmark
sections, not a giant hardware survey.

Use this split:

- **Profiler tables**: micro modes, useful for algorithmic implementation quality.
- **User-facing tables**: DataLoader/pipeline and video GPU runs. These get priority.
- **Hardware sensitivity**: only for RGB micro, because MacBook M4, `n2-standard-16`, and `n2d-standard-16` are already done.

Relevant code paths:

- `[benchmark/scenarios.py](../../benchmark/scenarios.py)`: defines `image-rgb`, `image-9ch`, `video-16f`, and `video-decode-16f`.
- `[benchmark/cli.py](../../benchmark/cli.py)`: exposes `--scenario`, `--mode micro`, `--mode pipeline`, `--gcp-machine-type`, and `--gcp-gpu-type`.
- `[benchmark/pipeline_runner.py](../../benchmark/pipeline_runner.py)`: current DataLoader/pipeline runner.
- `[benchmark/transforms/torchvision_video_impl.py](../../benchmark/transforms/torchvision_video_impl.py)` and `[benchmark/transforms/kornia_video_impl.py](../../benchmark/transforms/kornia_video_impl.py)`: GPU video micro path when CUDA is available.

## Hardware Matrix

Already done:

- MacBook M4: RGB micro local baseline.
- `n2-standard-16`: older Intel RGB micro baseline.
- `n2d-standard-16`: older AMD RGB micro baseline.

Core remaining machines:

- `c4-standard-16` in `us-central1-b`: canonical modern CPU for CPU-only paper tables.
- `c4d-standard-16` in `us-central1-b`: modern AMD sanity check for RGB micro only.
- `g2-standard-16` in `us-central1-b` with L4: canonical GPU machine for video torchvision/Kornia GPU micro and GPU video DataLoader only.

Cost rule:

- Do not run CPU-library rows on GPU VMs. If a table compares CPU and GPU libraries, run the CPU library on
`c4-standard-16` and the GPU libraries on `g2-standard-16`, then label the hardware per row.
- `c4-standard-16` is the CPU-only counterpart for main CPU rows. It is close enough to a modern training input CPU
without paying GPU VM prices.

Optional after core is done:

- `a2-highgpu-1g` in `us-central1-b`: A100 stress test for GPU video paths only.
- `c4a-standard-16` in `us-central1-b`: cloud Arm, only if non-Apple Arm belongs in the appendix.

With 32 total CPUs:

- Wave 1: `c4-standard-16` + `c4d-standard-16` for RGB micro only.
- Wave 2: `c4-standard-16` CPU suite, if Wave 1 did not already cover needed CPU runs.
- Wave 3: `g2-standard-16` GPU video suite, GPU libraries only.
- Wave 4 optional: `a2-highgpu-1g` A100 rerun of GPU video only.

## Seven Paper Sections

1. **Micro RGB**
  Use completed MacBook M4, `n2-standard-16`, and `n2d-standard-16`; add `c4-standard-16` and `c4d-standard-16`.
   This is the only section where hardware breadth is worth keeping.
2. **Micro Multichannel**
  Run only on `c4-standard-16` for the main paper. This is a profiler table, not a hardware study.
3. **DataLoader RGB**
  Run on `c4-standard-16`. This is more important than extra CPU micro machines. Use full ImageNet validation unless
   runtime is impossible; if using a subset, label it as non-final.
   If time allows, add a small worker sweep: `--workers 0`, `4`, `8`.
4. **DataLoader Multichannel**
  Run on `c4-standard-16`, same policy as RGB pipeline. Use fewer variants than RGB if needed.
5. **Video Albumentations CPU Micro**
  Run on `c4-standard-16` with only `albumentationsx` first. Add `albumentations_mit` only if the paper explicitly
   compares old vs new Albumentations.
6. **Video torchvision/Kornia GPU Micro**
  Run on `g2-standard-16` with L4. This is core for the paper because it tests the GPU augmentation path directly.
7. **Video DataLoader**
  Split by execution device to avoid paying GPU prices for CPU-library rows.
   CPU row on `c4-standard-16`:
   GPU rows on `g2-standard-16` with L4:
   If time allows, add worker sweep `0`, `4`, `8`. Do not add many clip lengths until the `T=16` table is complete.

## Deadline Rules

- First, produce one complete row set for all seven sections.
- Only after all seven sections have a complete result, add sweeps.
- Prefer `T=16`, one batch size, and one worker count over incomplete multi-axis sweeps.
- Do not rerun `n2`/`n2d` unless result files are corrupt; they are appendix baselines now.
- Skip `c4a-standard-16` unless Arm portability is a stated paper claim.
- Skip A100 unless L4 results are complete and clearly show a GPU crossover worth stress-testing.
- Never use GPU machines for CPU-only rows just to keep hardware identical. The paper should label hardware per row
instead of burning budget on GPU VMs for CPU work.

## Execution Order

1. Inventory existing result directories and mark best completed runs for MacBook M4, `n2-standard-16`, and `n2d-standard-16`.
2. Run smoke tests for each scenario on the target machine with tiny `--num-items`, `--num-runs 1`, and short `--min-time`.
3. Run final RGB micro on `c4-standard-16` and `c4d-standard-16`.
4. Run final CPU-only suite on `c4-standard-16`: 9ch micro, RGB DataLoader, 9ch DataLoader, Albumentations video CPU micro.
5. Run final GPU suite on `g2-standard-16`: torchvision/Kornia video GPU micro and GPU-only video DataLoader rows.
6. Pull all results from GCS, validate that each expected library produced both summary and pyperf/raw artifacts where applicable.
7. Generate paper tables/plots only after the complete core matrix exists.

## Result Coverage Target

Core paper is complete when these exist:

- `image-rgb/micro`: MacBook M4, `n2-standard-16`, `n2d-standard-16`, `c4-standard-16`, `c4d-standard-16`.
- `image-9ch/micro`: `c4-standard-16`.
- `image-rgb/pipeline`: `c4-standard-16`.
- `image-9ch/pipeline`: `c4-standard-16`.
- `video-16f/micro`: `c4-standard-16` for AlbumentationsX CPU; `g2-standard-16` for torchvision/Kornia GPU.
- `video-16f/pipeline`: `c4-standard-16` for AlbumentationsX CPU; `g2-standard-16` for torchvision, Kornia, and DALI if available.
