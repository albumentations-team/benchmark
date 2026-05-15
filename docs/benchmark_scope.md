# Benchmark Scope

For the prose methodology used by the website, README, papers, and longer-form writing, see
`docs/benchmark_methodology.md`. This scope document is the shorter operational reference.

## Library Sets

RGB image benchmarks compare four libraries:

- `albumentationsx`
- `torchvision`
- `kornia`
- `pillow`

9-channel image benchmarks compare three libraries:

- `albumentationsx`
- `torchvision`
- `kornia`

Pillow is excluded from 9-channel benchmarks because its direct image API is RGB/PIL-image oriented.

## Transform Selection

The shared transform catalog lives in `benchmark/transforms/specs.py`.

For RGB benchmarks, a transform is eligible when it exists in at least two of the four RGB libraries.
For 9-channel benchmarks, a transform is eligible when it exists in at least two of the three 9-channel libraries.
For video benchmarks, a transform is eligible when it exists in at least two selected video libraries.

The canonical transform sets are fixed in:

- `docs/paper_transform_sets/rgb.md`
- `docs/paper_transform_sets/9ch.md`
- `docs/paper_transform_sets/video.md`

Use them from the CLI with:

```bash
--transform-set paper
```

Each library still reports only transforms it supports directly. We do not recreate missing library features with large benchmark-side helper implementations just to force every library to have every row.

## DataLoader Pipeline Recipes

DataLoader pipeline benchmarks measure training-style recipes, not isolated transforms.
Micro benchmarks are different: they measure only the named transform in the library's native format and must not add
`Normalize`, `ToTensor`, axis conversion, or DataLoader collation work.

The fixed recipe steps are part of the DataLoader measurement. For every DataLoader recipe, the benchmark includes crop
shape preparation, normalization, and tensor conversion in addition to the measured augmentation. These steps are not
subtracted from throughput and are why DataLoader rows should be read as recipe throughput rather than primitive
transform throughput.

For non-crop transforms, the recipe shape is:

```text
RandomCrop224 + <transform> + Normalize + ToTensor
```

For crop transforms, the crop replaces `RandomCrop224`:

```text
<Crop> + Normalize + ToTensor
```

Example recipe names:

```text
RandomCrop224+Affine+Normalize+ToTensor
RandomCrop224+Brightness+Normalize+ToTensor
RandomResizedCrop+Normalize+ToTensor
```

`CenterCrop224` is not in the shared catalog (use `RandomCrop224` / `RandomResizedCrop` for crop coverage).

`Normalize` is not benchmarked as a pipeline augmentation because it is already part of every pipeline recipe.

`ToTensor` is implemented in the pipeline spec, not in the generic runner. AlbumentationsX pipeline recipes use
`Normalize` followed by `ToTensorV2`; Pillow pipeline recipes use `torchvision.transforms.PILToTensor` before normalization;
torchvision and Kornia already operate on tensors. The DataLoader runner should receive fixed-shape recipe outputs that
PyTorch can default-collate into a single batched tensor, and should not guess or repair channel layout. The
`decode_dataloader_augment_batch_copy` scope adds batch tensor materialization and optional CUDA/MPS transfer; default
collation itself is part of every DataLoader scope.

## Slow Transform Guard

Micro and DataLoader pipeline benchmarks run a preflight check before spending the full benchmark budget on a transform.

For image benchmarks, transforms slower than `0.05` seconds per image are early-stopped by default. This corresponds to `20 img/s`. For video benchmarks, transforms slower than `2.0` seconds per video are early-stopped by default.

The result JSON keeps an `early_stopped` entry with the preflight throughput and reason, instead of hanging the benchmark on transforms that are too slow for practical training use.

This guard is especially important for DataLoader benchmarks. A full pipeline run can multiply by:

```text
number of recipes * number of runs * full dataset epochs
```

Very slow transforms can otherwise make the benchmark appear stuck and block the full sweep. The early-stop policy is intentional: transforms below the practical throughput floor are not useful DataLoader candidates.

Use `--disable-slow-skip` only when explicitly measuring slow transforms.

## Architecture Source Of Truth

Benchmark policy is intentionally centralized:

- `benchmark/matrix.py` declares scenario/mode/library support, spec paths, requirement groups, canonical transform-set files,
  device policy, pipeline scopes, and backend names.
- `benchmark/policy.py` declares media defaults and slow-transform preflight thresholds.
- `benchmark/jobs.py` builds immutable benchmark jobs and subprocess commands.
- `benchmark/orchestrator.py` dispatches jobs to pyperf, DataLoader, or DALI pipeline backends.
- `benchmark/envs.py` owns joined virtualenvs, dependency refresh, and dependency cache keys.
- `benchmark/specs/load.py` and `benchmark/media/loaders.py` keep spec validation and media loading out of the CLI.
- Video pipeline-only ecosystem baselines live in the matrix as explicit libraries: DALI for native GPU pipelines,
  `dali_experimental` for the modern experimental DALI video reader path, and PyTorchVideo for a canonical per-clip
  training recipe. Batch-shared TorchVision video is excluded because it can share random parameters across clips in one
  batch and is therefore not the realistic training story this benchmark targets.

Do not add new benchmark matrix constants directly to `benchmark/cli.py`. Add them to `benchmark/matrix.py`, then extend
tests in `tests/test_matrix.py` and `tests/test_jobs_orchestrator.py`.

For a full module map, see `docs/benchmark_architecture.md`. For the rationale behind the benchmark scopes and timing
decisions, see `docs/benchmark_methodology.md`.

## Visual Progress

Long benchmark runs must show tqdm progress with descriptive labels. Progress bars should make it clear which dimension is moving:

- Library loops: `<scenario>/<mode>`.
- Media loading: `Load images (<library>, <channels>ch)` or `Load videos (<library>, <clip-length>f)`.
- Micro transforms: `Micro transforms (<library>, <media>)`.
- Pyperf micro transforms: `Pyperf micro transforms (<library>, <media>)`.
- Pipeline transforms: `Pipeline transforms (<library>, <scope>, w=<workers>, b=<batch_size>)`.

Do not add anonymous tqdm bars. Every tqdm must have a useful `desc` and a unit such as `lib`, `img`, `video`, or `transform`.

## Plot Policy

Plot choice must follow the public claim and benchmark regime. Use `docs/good_plots.md` as the source of truth for
claim-to-plot mapping, figure captions, and visualization anti-patterns.

In particular:

- Never use one merged leaderboard to support claims across micro, CPU DataLoader, GPU DataLoader, and DALI regimes.
- Pair throughput plots with coverage or unsupported-row reporting when library support differs.
- Use paired ratios for paired claims, for example `GPU pipeline / AlbumentationsX CPU pipeline` for the same transform.
- Keep unsupported and early-stopped rows visible in either the main figure, a coverage figure, or the generated supplement.
- Put memory-versus-throughput plots in the appendix unless the manuscript makes GPU memory an explicit claim.

## Production Run Plan

The website and README do not need the full benchmark matrix on every CPU vendor. Run the complete CPU suite once on a
modern Intel VM, run a small AMD sanity check, and run video GPU benchmarks separately.

**GCP quota:** Current project quota is **128 vCPUs** (`CPUS_ALL_REGIONS`), **96 C4-family vCPUs** in `us-central1`
(`CPUS_PER_VM_FAMILY`, `vm_family=C4`), and **1 GPU** (`GPUS_ALL_REGIONS`). That allows up to six concurrent
`c4-standard-16` CPU benchmark VMs from the C4-family quota, or five C4 jobs plus one `g2-standard-16` GPU job from the
all-CPU quota. Keep only one G2 job active because L4 quota remains one GPU.
The regional Hyperdisk Balanced quota is currently **500 GB** (`HDB_TOTAL_GB`), so production C4 configs use **100 GB**
boot disks. A 200 GB disk limits parallel C4 launch capacity to two active VMs before the third creation can fail on disk
quota.

### Main CPU Suite

Machine: `c4-standard-16` or equivalent modern Intel CPU.

Run these as the main public tables:

- RGB micro benchmark: `image-rgb`, `micro`, libraries `albumentationsx torchvision kornia pillow`, transforms from `docs/paper_transform_sets/rgb.md`.
- RGB DataLoader memory pipeline: `image-rgb`, `pipeline`, `memory_dataloader_augment`.
- RGB DataLoader disk/decode pipeline: `image-rgb`, `pipeline`, `decode_dataloader_augment`.
- RGB GPU image benchmarks: `image-rgb`, modes `micro` and `pipeline`, libraries `torchvision kornia`, `--device cuda`
  on `g2-standard-16`. Micro rows are device-resident transform-only measurements; DataLoader rows include CPU
  load/decode, library-native CPU crop/pad shape preparation, batch collation, host-to-device copy, GPU augmentation plus
  normalization, and synchronization. Kornia uses batched GPU augmentation with `same_on_batch=False`; TorchVision uses a
  per-sample GPU loop to preserve per-image random parameters.

Defer these from the headline RGB table:

- 9-channel micro/DataLoader benchmarks. They target multichannel imaging audiences and should be a separate result page or
  appendix after RGB is complete.
- Video micro/DataLoader benchmarks. Video has distinct decode, clip sampling, temporal consistency, and GPU pipeline
  questions; keep current smoke results as path validation only.

Recommended DataLoader settings for final production runs:

```text
--num-items 10000
--batch-size 256  # RGB
--workers 8
--num-runs 1
--min-time 0
--thread-policy pipeline-default
```

Use `10,000` ImageNet validation images for the deadline-first DataLoader table. Keep the same production path and
reduce only explicit sizing flags for cheaper iteration, for example
`--num-items 1000 --batch-size 64 --workers 8 --num-runs 1 --min-time 0`. After one complete coverage pass, add repeat
runs for important rows and aggregate them; do not block first coverage on 3- or 5-run sweeps.

For 16-frame video DataLoader runs, size by frame budget rather than item budget: `625` clips equals `10,000` frames. The
GPU video DataLoader batch is `16` clips because `16 clips * 16 frames = 256` frames, matching the intended image-scale GPU
batch. Cached DataLoader data lives in CPU RAM for RGB, 9-channel, and video. GPU DataLoader rows include the CPU batch to
GPU transfer and then run augmentation on the GPU; DALI remains separate because it owns its own input/decode pipeline.
For CPU DataLoader rows across RGB, 9-channel, and video, augmentation runs inside `DataLoader` workers before default
collation. Collation only stacks fixed-shape, already-augmented samples into a batch. For TorchVision and Kornia GPU
DataLoader rows, workers prepare CPU samples, collation builds the CPU batch, and the main process performs the
host-to-device copy plus GPU augmentation.

### AMD Sanity Check

Machine: `c4d-standard-16` or equivalent modern AMD CPU.

Run these only to confirm trends do not invert on another modern CPU:

- RGB micro benchmark: `image-rgb`, `micro`, libraries `albumentationsx torchvision kornia pillow`.
- Optional reduced RGB DataLoader sanity run: `image-rgb`, `pipeline`, one of `memory_dataloader_augment` or `decode_dataloader_augment`, with `--num-items 1000` or another small subset.

Do not rerun the full RGB + 9ch + DataLoader matrix on AMD unless the paper explicitly studies CPU-vendor effects.

### GPU / Video Suite

Machine: `g2-standard-16` with an L4 GPU, or equivalent.

Run these for video/GPU tables:

- GPU image micro and DataLoader sanity checks for `torchvision` and `kornia` on RGB and 9-channel images. DataLoader
  workers use the same library on CPU for crop/pad shape preparation, then the fixed-shape batch is copied to GPU.
  RGB GPU micro uses 2,000 device-resident images. 9-channel GPU micro uses 1,000 device-resident images because the
  2,000-sample Kornia preload OOMs on an L4; label this row as memory-limited and do not compare it as a same-`n` row.
  Kornia applies the measured augmentation with `same_on_batch=False` plus normalization. TorchVision applies the measured
  augmentation in a per-sample GPU loop, then normalizes the batch, because TorchVision v2 does not expose a
  `same_on_batch=False` equivalent for batched image transforms.
  For video rows, Kornia uses `VideoSequential(data_format="BTCHW", same_on_frame=True)` and TorchVision applies v2
  transforms per clip after the host-to-device copy, so both keep frame-consistent randomness within a clip without sharing
  random parameters across the whole batch.
  TorchVision `JpegCompression` is attempted in TorchVision GPU rows; `torchvision.transforms.v2.JPEG` may report
  unsupported at runtime because it requires `uint8` CPU input. Keep it in CPU TorchVision rows and in other libraries
  that support it.
  CUDA DataLoader rows also record per-transform peak GPU memory during timed runs. Use these fields when discussing the
  accelerator-memory cost of GPU augmentations; pyperf micro rows remain transform-time measurements and do not report
  peak memory because they execute inside pyperf worker processes.
  Production GPU image configs are `configs/paper/prod_g2_rgb_micro_gpu.yaml`,
  `configs/paper/prod_g2_9ch_micro_gpu.yaml`, `configs/paper/prod_g2_rgb_dataloader_gpu.yaml`, and
  `configs/paper/prod_g2_9ch_dataloader_gpu.yaml`. The corresponding `gcp_*_smoke.yaml` configs remain for fast path
  checks and reruns.
- Kornia/TorchVision GPU rows should not be silently removed for fixable dtype, device, layout, or CPU-only adapter
  issues. Attempt the row and record an unsupported result with the exact runtime reason unless the transform is proven
  to crash the worker process or poison the CUDA context. These are library/device limitations, not global benchmark
  transform-set removals: the transforms remain in CPU rows and in other libraries that support them.
- Kornia RGB GPU DataLoader can fail `GaussianIllumination` with a mixed CPU/CUDA tensor error in the current L4 run.
  Treat this as an unsupported Kornia GPU recipe result and keep it as methodology evidence for GPU augmentation
  benchmarking complexity.
- GPU video micro benchmarks for GPU-capable libraries, especially `torchvision` and `kornia`. Micro video preload uses
  fixed-length clips from `--clip-length` (16 frames for `video-16f`), not full source videos.
- Kornia CPU video micro excludes `Rotate` and `Elastic`. The May 11, 2026 C4 run reached Kornia after completing
  AlbumentationsX and TorchVision, then crashed reproducibly in the pyperf child process for Kornia `Rotate` with
  `SIGSEGV: 11`. The May 12, 2026 Kornia-only rerun excluded `Rotate` and then crashed the same way on `Elastic`.
  These look like native-code crashes in Kornia's CPU video augmentation path, not VM memory pressure
  (`SIGKILL`/OOM). Keep these as library/scenario exclusions rather than removing the transforms from the global video
  transform set.
- GPU video DataLoader/pipeline benchmarks for GPU-capable paths. These use dedicated video pipeline specs rather than
  micro specs, so AlbumentationsX, torchvision, and Kornia all run recipe-style clips through DataLoader collation.
- Reduced G2 smoke has already succeeded on UCF101 for `torchvision kornia` video micro and for
  `albumentationsx torchvision kornia` video pipeline with `decode_dataloader_augment_batch_copy`, `--device cuda`,
  `--num-items 10`, `--batch-size 2`, and `--workers 2`.
- Kornia video DataLoader/pipeline rows use `benchmark/transforms/kornia_unstable.py` only for confirmed crash-only
  exclusions. Rows with ordinary Python exceptions, dtype/device mismatches, or unsupported CPU-only operators should
  remain visible as unsupported results rather than disappearing before execution.
- Earlier Kornia video DataLoader runs recorded `RandomCrop224+Snow+Normalize+ToTensor` with
  `NotImplementedError: "check_uniform_bounds" not implemented for 'Long'` and GPU affine with a mixed float/half
  grid-sampler error. These are adapter-correctness targets before any final rerun: Kornia video wrappers should use
  float bounds, float32 affine parameters, and contiguous tensors before classifying the remaining rows as true
  library/device limitations.
- DALI pipeline benchmarks when DALI is available on the target image. Current DALI coverage is video pipeline plus RGB
  image GPU DataLoader-style pipeline; DALI image rows use the DALI-supported subset and report unsupported recipes
  explicitly. For video, `dali` means stable public `fn.readers.video`; `dali_experimental` means
  `fn.experimental.readers.video` and is diagnostic until the smoke/prod recipe set proves stable.

CPU-only image rows should not be rerun on GPU machines for hardware symmetry. GPU image rows are a separate
TorchVision/Kornia/DALI sanity section and must be labeled with device, machine class, whether transfer is included, and
whether TorchVision used the per-sample GPU loop or DALI used its own graph executor.

### Website Data Status

Use this checklist for website-facing benchmark data and README figures. Keep PIL out of 9-channel/video tables because it
is RGB-only in this benchmark. Keep DALI and PyTorchVideo explicitly labeled as ecosystem pipeline baselines rather than
folding them into primitive transform microbenchmarks.

Current status as of 2026-05-15:

| Area | Config | Status | Next action |
| --- | --- | --- | --- |
| 9-channel GPU micro | `configs/paper/prod_g2_9ch_micro_gpu.yaml` | Complete and fetched locally from GCS run `27218e63a7cf435bb72245f0048e9b50`. | Include in supplement aggregation. Kornia has 34 ok rows and 4 unsupported CUDA rows; TorchVision has 21 ok rows. |
| 9-channel CPU micro | `configs/paper/prod_c4_9ch_micro_cpu.yaml` | GCS run `bd0b4a811a6d4b59b8af0ad0a811859c` completed, wrote `DONE`, and was fetched locally. AlbumentationsX has 41 ok rows and TorchVision has 22 ok rows. Kornia is not usable as a final comparable row: 35 of 40 rows are `SIGKILL`/exit `-9` failures, with only 5 ok rows. | Keep the fetched AlbumentationsX/TorchVision rows. Rerun only Kornia CPU micro, preferably on a higher-memory CPU VM or with a clearly labeled reduced-memory protocol if highmem quota blocks the run. |
| 9-channel CPU DataLoader | `configs/paper/prod_c4_9ch_dataloader_cpu.yaml` | GCS run `b6548a5b153349b7bd47e9dc9defccc0` failed during Kornia with `SIGKILL`; AlbumentationsX and TorchVision outputs were fetched locally. | Keep the salvaged AlbumentationsX/TorchVision rows. Do not rerun the 3-library standard-C4 config. |
| 9-channel CPU DataLoader Kornia | `configs/paper/prod_c4_highmem_9ch_dataloader_cpu_kornia.yaml` | Added after the standard C4 run killed Kornia while preloading 10k float32 9-channel tensors. | Run on `c4-highmem-16` to preserve the same `memory_dataloader_augment`, 10k-item protocol. If highmem quota is unavailable, run a reduced-n Kornia row and label it memory-limited/not same-n. |
| 9-channel GPU DataLoader | `configs/paper/prod_g2_9ch_dataloader_gpu.yaml` | Not confirmed complete locally. | Run or locate the G2 job after the current 1-GPU queue is clear. |
| Video CPU micro | `configs/paper/prod_c4_video_micro_cpu.yaml` | GCS run `ca332e6f5dd949fd85b8435c5b56346d` failed during Kornia `Rotate` with `SIGSEGV: 11`; GCS run `132f29d9503b49fead76c45ba25a6851` excluded `Rotate` and then failed during Kornia `Elastic` with `SIGSEGV: 11`. AlbumentationsX and TorchVision outputs were fetched locally from the first run. | Keep AlbumentationsX/TorchVision. Rerun Kornia CPU micro with `Rotate` and `Elastic` excluded by the scenario filter. |
| Video CPU DataLoader | `configs/paper/prod_c4_video_dataloader_cpu.yaml` | GCS run `561ceed0c7a44b42b13fa9d9cde58a1e` wrote `DONE`, but all result JSONs had empty `results` because video pipeline `paper` transforms resolved to micro names instead of recipe names. After the resolver fix, GCS run `d444860ec02b4c8da189dc5df893eee2` completed with `DONE`, exit code `0`, and uploaded `vm.log`. It produced 43 AlbumentationsX ok rows, 25 TorchVision ok rows, and 37 Kornia ok rows plus one Kornia unsupported row: `RandomCrop224+Snow+Normalize+ToTensor` failed with `NotImplementedError: "check_uniform_bounds" not implemented for 'Long'`. | Keep the May 13 run as the CPU video DataLoader result set, with Kornia `Snow` documented as unsupported in this recipe path. |
| Video GPU micro | `configs/paper/prod_g2_video_micro_gpu.yaml` | Published from GCS run `d0b48b7a9b8c4b7eaf01b90edf4603ea`. Kornia has 40 ok rows and 9 unsupported rows; TorchVision has 25 ok rows and one unsupported row. | Keep as the video GPU micro result set. |
| Video GPU DataLoader | `configs/paper/prod_g2_video_dataloader_gpu.yaml` | GCS run `5a50bff8adf64efeb1f682684eea4c3b` wrote `DONE`, but all result JSONs had empty `results` for the video pipeline transform-name resolver bug. After the resolver fix, GCS run `e3e897517238436abb1359112d15b18f` completed with `DONE`, exit code `0`, and uploaded `vm.log`. It produced 24 TorchVision ok rows plus one TorchVision unsupported row (`JpegCompression` CPU-only), and 36 Kornia ok rows plus two Kornia unsupported rows (`Affine` mixed CUDA float/half grid sampler, `Snow` Long bounds error). | Keep the May 13 run as the GPU video DataLoader result set, with unsupported rows documented as library/device recipe limitations. |
| Video DALI GPU DataLoader | `configs/paper/prod_g2_video_dataloader_dali.yaml` | Published from GCS run `a22bac0b4a9946e3aaf25278fa206966`. DALI native video pipeline has 21 ok rows and 29 unsupported rows. | Keep as a separately labeled native DALI pipeline baseline. Do not compare it as a micro-transform benchmark. |
| Video PyTorchVideo GPU DataLoader | `configs/paper/prod_g2_video_dataloader_pytorchvideo.yaml` | Published from GCS run `8e2206a2028941f8988f11d273c28bf9`. PyTorchVideo has one ok canonical training-pipeline row. | Keep as a canonical pipeline baseline, not as a 50-recipe transform coverage matrix. |

Fetch completed detached runs with the `fetch_results_hint` in each `gcp_last_run.json`, for example:

```bash
gcloud storage cp -r 'gs://imagenet_validation/augmentation-results/<run-id>/results/*' gcp_runs/<local-run-dir>/
```

For GPU configs, do not assume the zone in the YAML currently has capacity. We do not have usable G2/L4 capacity in every
GCP zone, and stocked-out zones are common. Launch production GPU jobs through the zone-search helper so the config stays
unchanged while the script tries known GPU-capable zones:

```bash
scripts/run_gcp_first_available_gpu_zone.sh configs/paper/prod_g2_video_dataloader_gpu.yaml
```

Extra `benchmark.cli run` overrides can follow the config path, for example `--libraries kornia` or
`--gcp-timeout-hours 8`. Use a direct `python -m benchmark.cli run --config ... --gcp-zone ...` launch only when you have
already confirmed that the chosen zone has the required GPU capacity.

Detached GCP jobs now have a bootstrap hard timeout, defaulting to 6 hours for GPU VMs and 8 hours for CPU VMs. Override
with `--gcp-timeout-hours` when a run legitimately needs longer. Self-delete still runs on normal `DONE` or `FAILED`, but
it is not sufficient if the VM wedges before the delete command can execute. Use `scripts/gcp_cleanup_stale_benchmarks.sh`
to list expired benchmark VMs, then rerun it with `--delete` after confirming the stale instances.

After fetching each run, summarize result status before plotting:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("gcp_runs").glob("**/*_results.json")):
    data = json.loads(path.read_text())
    statuses = {}
    for row in data.get("results", {}).values():
        status = row.get("status", "ok") if isinstance(row, dict) else "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    print(path, statuses)
PY
```

### Validation

After pulling artifacts, validate coverage before producing tables:

```bash
python -m tools.check_paper_coverage gcp_runs output
```

Use optional-library coverage checks only when DALI is required for the current table.
