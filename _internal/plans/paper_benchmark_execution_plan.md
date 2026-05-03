# Paper Benchmark Execution Plan

Goal: run the same production benchmark paths first with small inputs locally,
then repeat them on GCP. These runs are for end-to-end coverage before spending
compute on tighter paper error bars.

## Constants

- Local RGB data: set `RGB_DATA_DIR=/path/to/imagenet/val`
- 9-channel runs use the same RGB data source; the loader stacks 3 RGB copies into 9 channels in memory.
- GCP project: `albumentations`
- GCP zone: `us-central1-b`
- GCP CPU machine: `c4-standard-16`
- Current GCP quota: `CPUS_ALL_REGIONS = 64`, `GPUS_ALL_REGIONS = 1`
- GCP ImageNet tarball: `gs://imagenet_validation/imagenet/val.tar`
- Local UCF101 data: `/Users/vladimiriglovikov/data/ucf101`
- GCP UCF101 tarball: `gs://imagenet_validation/ucf101/ucf101.tar` (uploaded; size `14136559616` bytes)
- GCP results prefix: `gs://imagenet_validation/augmentation-results`
- Micro sizing: `data.num_items: 1000`, `execution.num_runs: 1`
- DataLoader sizing: `data.num_items: 10000`, `execution.num_runs: 1`
- DataLoader scope for paper CPU checks: `memory_dataloader_augment`
- GPU image smoke sizing: `data.num_items: 100`, `execution.num_runs: 1`

Create and upload cloud datasets as tarballs, not GCS directories of individual files:

```bash
COPYFILE_DISABLE=1 tar --no-xattrs \
  --exclude="__MACOSX" \
  --exclude="*/__MACOSX/*" \
  --exclude=".DS_Store" \
  --exclude="*/.DS_Store" \
  --exclude="._*" \
  --exclude="*/._*" \
  -cf /tmp/ucf101.tar \
  -C /Users/vladimiriglovikov/data ucf101

gcloud storage cp /tmp/ucf101.tar gs://imagenet_validation/ucf101/ucf101.tar
gcloud storage objects describe gs://imagenet_validation/ucf101/ucf101.tar --format="yaml(size,crc32c,md5Hash,updated)"
tar -tf /tmp/ucf101.tar | rg '(^__MACOSX/|/\.DS_Store$|^\.DS_Store$|/\._|^\._)'
```

The uploaded paper video dataset is `gs://imagenet_validation/ucf101/ucf101.tar`. The final `tar -tf ... | rg ...`
command should print nothing. Detached GCP staging filters media files by scenario while extracting, so video micro and
pipeline runs can use the same UCF101 tarball.

## TODO: Local Reduced Production-Path Runs

- [x] RGB micro, all RGB paper libraries, all RGB paper transforms.

```bash
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml \
  --data-dir "$RGB_DATA_DIR" \
  --output output/paper-local-rgb-micro-small \
  --num-items 1000
```

- [x] 9-channel micro, all 9-channel paper libraries, all 9-channel paper transforms. Initial local pass used 500 images.

```bash
python -m benchmark.cli run --config configs/examples/local_9ch_micro_cpu.yaml \
  --data-dir "$RGB_DATA_DIR" \
  --output output/paper-local-9ch-micro-small \
  --num-items 1000
```

- [x] RGB RAM DataLoader, all RGB paper libraries, all RGB paper transforms. Initial local pass used 2,000 images; only Tensor Elastic worker warmup failed.

```bash
python -m benchmark.cli run --config configs/examples/local_rgb_dataloader_cpu.yaml \
  --data-dir "$RGB_DATA_DIR" \
  --output output/paper-local-rgb-dataloader-memory-small \
  --num-items 10000
```

- [x] 9-channel RAM DataLoader, all 9-channel paper libraries, all 9-channel paper transforms. Initial local pass used 2,000 images; Tensor Elastic worker warmup failed, Kornia MedianBlur slow-skipped.

```bash
python -m benchmark.cli run --config configs/examples/local_9ch_dataloader_cpu.yaml \
  --data-dir "$RGB_DATA_DIR" \
  --output output/paper-local-9ch-dataloader-memory-small \
  --num-items 10000
```

## TODO: GCP Reduced Production-Path Runs

- [x] RGB micro on `c4-standard-16`.

```bash
python -m benchmark.cli run --config configs/paper/gcp_c4_rgb_micro_cpu.yaml
```

- [x] RGB RAM DataLoader on `c4-standard-16`. Reduced GCP pass used 2,000 images, 4 workers, split across tensor libs (`albumentationsx`, `torchvision`, `kornia`) and Pillow; all completed, with expected Kornia slow-skips for `Elastic` and `MedianBlur`.

```bash
python -m benchmark.cli run --config configs/paper/gcp_c4_rgb_dataloader_cpu.yaml
```

- [x] 9-channel micro on `c4-standard-16`. Reduced GCP pass requested 1,000 images and used 982 valid RGB images stacked to 9 channels; all libraries completed, with expected slow-skips for very slow tensor transforms.

```bash
python -m benchmark.cli run --config configs/paper/gcp_c4_9ch_micro_cpu.yaml
```

- [x] 9-channel RAM DataLoader on `c4-standard-16`. Reduced GCP pass used 2,000 images, 4 workers; all three libraries completed, with expected Kornia slow-skips for `Elastic` and `MedianBlur`.

```bash
python -m benchmark.cli run --config configs/paper/gcp_c4_9ch_dataloader_cpu.yaml
```

The following GPU image smoke runs are still being completed. Run them before interpreting GPU image rows.
If `us-central1-b` is out of L4 capacity, retry the same config with `--gcp-zone us-central1-a` or
`--gcp-zone us-central1-c`, matching the zones suggested by GCP.
Kornia image GPU jobs intentionally exclude `Shear` in both micro and DataLoader modes because Kornia's current CUDA
shear parameter generator can fail with mixed CPU/CUDA tensors when the transform is moved to GPU. Keep `Shear` in the
overall RGB/9-channel paper transform sets for AlbumentationsX, Pillow, torchvision where supported, and Kornia CPU rows;
call out this Kornia GPU limitation in the paper methodology.
GPU image DataLoader smoke configs include TorchVision and Kornia. Both use library-native CPU crop/pad shape preparation
before collation. Kornia then applies GPU augmentation batched with `same_on_batch=False`; TorchVision applies the measured
augmentation in a per-sample GPU loop, then normalizes the batch, because TorchVision v2 does not expose a
`same_on_batch=False` equivalent for batched image transforms.
TorchVision `JpegCompression` is excluded from TorchVision GPU image rows because `torchvision.transforms.v2.JPEG`
requires `uint8` CPU input. Keep it in CPU TorchVision rows and in other libraries that support it; call out this
JPEG-compression augmentation constraint in the paper methodology.
CUDA DataLoader rows record per-transform peak GPU memory during timed runs (`gpu_memory.peak_allocated_bytes` and
`gpu_memory.peak_reserved_bytes`). Use this as a paper-facing cost column for GPU augmentation; pyperf micro rows do not
report peak memory because pyperf executes timed loops in worker processes.

- [x] GPU RGB image micro smoke on `g2-standard-16` for tensor-native libraries. TorchVision completed in the earlier
  mixed run; Kornia completed in the follow-up Kornia-only rerun after filtering Kornia GPU `Shear`.
  Kornia rerun prefix: `gs://imagenet_validation/augmentation-results/e9b939dc478d411d9dc2fa1b914dc699`.

```bash
python -m benchmark.cli run --config configs/paper/gcp_g2_rgb_micro_gpu_smoke.yaml
python -m benchmark.cli run --config configs/paper/gcp_g2_rgb_micro_gpu_smoke.yaml --libraries kornia --gcp-zone us-central1-a
```

- [x] GPU 9-channel image micro smoke on `g2-standard-16` for tensor-native libraries. Completed for TorchVision and
  Kornia with CUDA results uploaded.
  Run prefix: `gs://imagenet_validation/augmentation-results/00471889465b4e3087c960ba6adde8e4`.

```bash
python -m benchmark.cli run --config configs/paper/gcp_g2_9ch_micro_gpu_smoke.yaml
```

- [x] GPU RGB image DataLoader smoke on `g2-standard-16` for tensor-native libraries. Fresh fetched validation passed
  after excluding TorchVision `JpegCompression` on GPU: TorchVision has 25/25 supported rows, Kornia has 48/50 supported
  rows plus 2 expected library/device failures, and all non-preflight measured rows include CUDA memory fields.
  Run prefix: `gs://imagenet_validation/augmentation-results/8d015c32ddc8482f8b8187bc2f29825c`.

```bash
python -m benchmark.cli run --config configs/paper/gcp_g2_rgb_dataloader_gpu_smoke.yaml
```

- [x] GPU 9-channel image DataLoader smoke on `g2-standard-16` for tensor-native libraries. Fresh fetched validation
  passed after excluding TorchVision `JpegCompression` on GPU: TorchVision has 21/21 supported rows and CUDA memory fields;
  Kornia has 37/39 supported rows plus 2 expected library/device failures, with CUDA memory fields for measured rows.
  Kornia prefix: `gs://imagenet_validation/augmentation-results/c2668476216a441cb6dc747146129d31`.
  TorchVision prefix: `gs://imagenet_validation/augmentation-results/8b54138ef610416cb7c21f2ffcad4262`.

```bash
python -m benchmark.cli run --config configs/paper/gcp_g2_9ch_dataloader_gpu_smoke.yaml
```

- [x] GPU video smoke on `g2-standard-16` with the UCF101 tarball.

Successful smoke prefixes:

- `gs://imagenet_validation/augmentation-results/821ae79852204f5cb4d5bea42fab99b1`: video micro,
  `torchvision kornia`, `DONE`, `exit_code=0`.
- `gs://imagenet_validation/augmentation-results/861bd4a840a84ec28ff711f3f68c81a8`: video pipeline,
  `albumentationsx torchvision kornia`, `decode_dataloader_augment_batch_copy`, `--device cuda`,
  `DONE`, `exit_code=0`.
- `gs://imagenet_validation/augmentation-results/b7e8cdf6fd154357be68a0b38d134136`: repeat video pipeline
  smoke with the same CUDA batch-copy settings, `DONE`, `exit_code=0`.

Known failed smoke prefix:

- `gs://imagenet_validation/augmentation-results/cebb07f24c5444b1aea96e80a471af84`: video micro with
  `albumentationsx torchvision kornia`, `FAILED`, `exit_code=1`; only the AlbumentationsX partial result was uploaded.

Check current CPU/GPU quota before launching more detached jobs:

```bash
python - <<'PY'
import json
import subprocess

raw = subprocess.check_output(
    ["gcloud", "compute", "project-info", "describe", "--project", "albumentations", "--format=json"],
    text=True,
)
for quota in json.loads(raw).get("quotas", []):
    metric = quota.get("metric", "")
    if "CPU" in metric or "GPU" in metric or "NVIDIA" in metric:
        print(f"{metric}: limit={quota.get('limit')} usage={quota.get('usage')}")
PY
```

```bash
python -m benchmark.cli run --config configs/paper/gcp_g2_video_smoke.yaml
```

## After Each Run

- [ ] Confirm the local output directory contains summary JSON files.
- [ ] For GCP, check the run prefix from `gcp_last_run.json`.
- [ ] For GCP failures, inspect `<run_prefix>/vm.log`.
- [ ] After the full pass, run coverage validation:

```bash
python -m tools.check_paper_coverage --profile ram-reduced gcp_runs output
```

## Final Paper Reruns Later

After the reduced production-path pass is clean, rerun the rows that feed paper
claims with larger/repeated measurements, usually `--num-runs 3` or `5`, and
only broaden further where variance or close comparisons require it.

Paper note: Kornia video pipeline excludes transforms that are unstable in the current CUDA recipe path. Smoke runs showed
device mismatches, integer-bound failures, and CUDA device-side assertions for several Kornia video recipes. Treat those as
library/path behavior and report them as unsupported/unstable rather than forcing them into the main throughput table.
