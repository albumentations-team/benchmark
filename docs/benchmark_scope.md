# Benchmark Scope

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

The paper transform sets are fixed in:

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

Very slow transforms can otherwise make the benchmark appear stuck and block the paper sweep. The early-stop policy is intentional: transforms below the practical throughput floor are not useful DataLoader candidates.

Use `--disable-slow-skip` only when explicitly measuring slow transforms.

## Architecture Source Of Truth

Benchmark policy is intentionally centralized:

- `benchmark/matrix.py` declares scenario/mode/library support, spec paths, requirement groups, paper transform-set files,
  device policy, pipeline scopes, and backend names.
- `benchmark/policy.py` declares media defaults and slow-transform preflight thresholds.
- `benchmark/jobs.py` builds immutable benchmark jobs and subprocess commands.
- `benchmark/orchestrator.py` dispatches jobs to pyperf, DataLoader, or DALI pipeline backends.
- `benchmark/envs.py` owns joined virtualenvs, dependency refresh, and dependency cache keys.
- `benchmark/specs/load.py` and `benchmark/media/loaders.py` keep spec validation and media loading out of the CLI.

Do not add new benchmark matrix constants directly to `benchmark/cli.py`. Add them to `benchmark/matrix.py`, then extend
tests in `tests/test_matrix.py` and `tests/test_jobs_orchestrator.py`.

For a full module map, see `docs/benchmark_architecture.md`.

## Visual Progress

Long benchmark runs must show tqdm progress with descriptive labels. Progress bars should make it clear which dimension is moving:

- Library loops: `<scenario>/<mode>`.
- Media loading: `Load images (<library>, <channels>ch)` or `Load videos (<library>, <clip-length>f)`.
- Micro transforms: `Micro transforms (<library>, <media>)`.
- Pyperf micro transforms: `Pyperf micro transforms (<library>, <media>)`.
- Pipeline transforms: `Pipeline transforms (<library>, <scope>, w=<workers>, b=<batch_size>)`.

Do not add anonymous tqdm bars. Every tqdm must have a useful `desc` and a unit such as `lib`, `img`, `video`, or `transform`.

## Paper Run Plan

The paper does not need the full benchmark matrix on every CPU vendor. Run the complete CPU suite once on a modern Intel VM, run a small AMD sanity check, and run video GPU benchmarks separately.

**GCP quota:** Current project quota is **128 vCPUs** (`CPUS_ALL_REGIONS`), **96 C4-family vCPUs** in `us-central1`
(`CPUS_PER_VM_FAMILY`, `vm_family=C4`), and **1 GPU** (`GPUS_ALL_REGIONS`). That allows up to six concurrent
`c4-standard-16` CPU benchmark VMs from the C4-family quota, or five C4 jobs plus one `g2-standard-16` GPU job from the
all-CPU quota. Keep only one G2 job active because L4 quota remains one GPU.
The regional Hyperdisk Balanced quota is currently **500 GB** (`HDB_TOTAL_GB`), so production C4 configs use **100 GB**
boot disks. A 200 GB disk limits parallel C4 launch capacity to two active VMs before the third creation can fail on disk
quota.

### Main CPU Suite

Machine: `c4-standard-16` or equivalent modern Intel CPU.

Run these as the main paper tables:

- RGB micro benchmark: `image-rgb`, `micro`, libraries `albumentationsx torchvision kornia pillow`, transforms from `docs/paper_transform_sets/rgb.md`.
- 9-channel micro benchmark: `image-9ch`, `micro`, libraries `albumentationsx torchvision kornia`, transforms from `docs/paper_transform_sets/9ch.md`.
- RGB DataLoader memory pipeline: `image-rgb`, `pipeline`, `memory_dataloader_augment`.
- RGB DataLoader disk/decode pipeline: `image-rgb`, `pipeline`, `decode_dataloader_augment`.
- 9-channel DataLoader memory pipeline: `image-9ch`, `pipeline`, `memory_dataloader_augment`.
- 9-channel DataLoader disk/decode pipeline: `image-9ch`, `pipeline`, `decode_dataloader_augment`.
- Video micro benchmarks use transforms from `docs/paper_transform_sets/video.md`. Video DataLoader benchmarks use
  dedicated recipe specs with `crop + transform + Normalize + ToTensor` semantics, matching RGB pipeline structure.
- GPU image sanity benchmarks: `image-rgb` and `image-9ch`, modes `micro` and `pipeline`, libraries
  `torchvision kornia`, `--device cuda` on `g2-standard-16`. Micro rows are device-resident transform-only measurements;
  DataLoader rows include CPU load/decode, library-native CPU crop/pad shape preparation, batch collation, host-to-device
  copy, GPU augmentation plus normalization, and synchronization. Kornia uses batched GPU augmentation with
  `same_on_batch=False`; TorchVision uses a per-sample GPU loop to preserve per-image random parameters.

Recommended DataLoader settings for final paper runs:

```text
--num-items 10000
--batch-size 256  # RGB
--batch-size 128  # 9-channel
--workers 8
--num-runs 1
--min-time 0
--thread-policy pipeline-default
```

Use `10,000` ImageNet validation images for the deadline-first DataLoader table. Keep the same production path and
reduce only explicit sizing flags for cheaper iteration, for example
`--num-items 1000 --batch-size 64 --workers 8 --num-runs 1 --min-time 0`. After one complete coverage pass, add repeat
runs for important rows and aggregate them; do not block first coverage on 3- or 5-run sweeps.
The main 9-channel DataLoader table uses `batch_size=128` for all libraries because the first `batch_size=256` CPU run
OOM-killed Kornia. Keep 9-channel batch size uniform across libraries; do not mix the partial `b256` rows into the main
table.

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
  Kornia applies the measured augmentation with `same_on_batch=False` plus normalization. TorchVision applies the measured
  augmentation in a per-sample GPU loop, then normalizes the batch, because TorchVision v2 does not expose a
  `same_on_batch=False` equivalent for batched image transforms.
  TorchVision `JpegCompression` is excluded from TorchVision GPU image rows because `torchvision.transforms.v2.JPEG`
  requires `uint8` CPU input. Keep it in CPU TorchVision rows and in other libraries that support it.
  CUDA DataLoader rows also record per-transform peak GPU memory during timed runs. Use these fields when discussing the
  accelerator-memory cost of GPU augmentations; pyperf micro rows remain transform-time measurements and do not report
  peak memory because they execute inside pyperf worker processes.
  Production GPU image configs are `configs/paper/prod_g2_rgb_micro_gpu.yaml`,
  `configs/paper/prod_g2_9ch_micro_gpu.yaml`, `configs/paper/prod_g2_rgb_dataloader_gpu.yaml`, and
  `configs/paper/prod_g2_9ch_dataloader_gpu.yaml`. The corresponding `gcp_*_smoke.yaml` configs remain for fast path
  checks and reruns.
- Kornia image GPU rows exclude `Shear` in both micro and DataLoader modes because the current Kornia CUDA shear path can
  fail while moving the transform's parameter generator to GPU. TorchVision image GPU rows exclude `JpegCompression`
  because TorchVision's JPEG op is CPU-only. These are library/device limitations, not global paper transform-set
  removals: the transforms remain in CPU rows and in other libraries that support them.
- GPU video micro benchmarks for GPU-capable libraries, especially `torchvision` and `kornia`. Micro video preload uses
  fixed-length clips from `--clip-length` (16 frames for `video-16f`), not full source videos.
- GPU video DataLoader/pipeline benchmarks for GPU-capable paths. These use dedicated video pipeline specs rather than
  micro specs, so AlbumentationsX, torchvision, and Kornia all run recipe-style clips through DataLoader collation.
- Reduced G2 smoke has already succeeded on UCF101 for `torchvision kornia` video micro and for
  `albumentationsx torchvision kornia` video pipeline with `decode_dataloader_augment_batch_copy`, `--device cuda`,
  `--num-items 10`, `--batch-size 2`, and `--workers 2`.
- Kornia video DataLoader/pipeline rows exclude transforms in `benchmark/transforms/kornia_unstable.py` due to CUDA
  stability issues in that recipe path only. Kornia image GPU rows additionally exclude only `Shear`; Kornia image CPU
  rows, 9-channel CPU rows, and video micro keep the global paper transform sets.
- DALI video pipeline benchmarks when DALI is available on the target image.

CPU-only image rows should not be rerun on GPU machines for hardware symmetry. GPU image rows are a separate
TorchVision/Kornia sanity section and must be labeled with device, machine class, whether transfer is included, and
whether TorchVision used the per-sample GPU loop.

### Validation

After pulling artifacts, validate coverage before producing tables:

```bash
python -m tools.check_paper_coverage gcp_runs output
```

Use optional-library coverage checks only when DALI is required for the current table.
