# Production benchmark contract

This repository measures the path from a prepared file on local disk to a batch ready for a
model on the GPU. Only RGB runs today. Nine-channel images, video, and volumes will use the same
runner and result schema once each family has an agreed dataset, recipe catalog, output
contract, and batch size.

The [methodology FAQ](benchmark_faq.md) explains the choices behind this contract.

## What one cell measures

A cell is identified by `(family, implementation, recipe, seed)`. A successful RGB cell records
all of the following in one pass:

- Throughput from JPEG on disk to a model-ready batch.
- Peak process GPU memory from pipeline construction through the final CUDA synchronization.
- GPU memory used by the same process before pipeline construction (`gpu_memory.baseline_mib`).
- Validated output: CUDA `float16`, BCHW, `B×3×224×224`.
- Library and GPU versions for diagnostics.

A separate memory run is prohibited. CPU memory, microbenchmarks, isolated H2D transfers,
decoder-only benchmarks, training loops, and maximum-batch sweeps answer other questions and
fall outside the production benchmark.

## Fixed RGB workload

| Field | Value |
| --- | --- |
| Data | The first 10,000 `val/*.JPEG` files in lexicographic order from the SHA-256-verified ImageNet validation archive |
| Recipes | 73 manually matched RGB recipes, each supported by at least two libraries |
| Matrix | 441 supported implementation–recipe pairs × 3 seeds = 1,323 cells |
| Seeds | 137, 138, 139 |
| Batch size | 256 images |
| Warm-up | 1 full batch outside the timer |
| Measurement | 32 full batches, 8,192 images |
| Workers | 15, `persistent_workers=True`, `prefetch_factor=2`, `pin_memory=True` |
| Hardware | One Standard `g2-standard-16` VM with one NVIDIA L4 in any eligible GCP zone |

All PyTorch DataLoader paths use `prefetch_factor=2`. Each worker can prepare up to two
batches. The setting is fixed across libraries.

The AlbumentationsX process limits BLAS, OpenMP, OpenCV, NumExpr, and PyTorch to one
thread per CPU operation. Its DataLoader workers inherit the environment, and the
runner logs the effective settings after preflight. Other implementations use their
libraries' default thread settings, including PyTorch's built-in DataLoader behavior.
DALI uses its configured pipeline worker count.

## Timing boundary

```text
JPEG on local disk
  → library reader and decoder
  → recipe
  → collate
  → pinned H2D transfer
  → remaining GPU recipe
  → GPU normalization
  → output validation
  → CUDA synchronization
```

Throughput excludes `Compose` initialization, imports, and preflight. NVML samples GPU memory
for the implementation's PID before pipeline construction and measures the peak in the same pass
as throughput. The peak includes the entire process's VRAM use, including CUDA state retained
after preflight. Subtracting the baseline from the peak gives the increase during the cell.

Normalization always runs on the GPU. Every path produces normalized CUDA `float16` output.
The table describes new runs. Each completed run retains the implementation in its frozen code archive.

| Implementation | Before GPU transfer | On the GPU |
| --- | --- | --- |
| AlbumentationsX CPU | SimpleJPEG, `uint8` recipe, pinned H2D transfer in `uint8` | BCHW view, then `albumentations.batch.InputNormalizer` with FP32 arithmetic and FP16 output |
| Pillow CPU | Pillow decode, `uint8` recipe, pinned H2D transfer in `uint8` | Convert to `float16` and normalize |
| Pillow-AllSIMD CPU (fast) | Same code with the Pillow-AllSIMD wheel in a separate process | Convert to `float16` and normalize |
| TorchVision CPU | `torchvision.io`, `uint8` recipe, pinned H2D transfer in `uint8` | Convert the batch to `float16` and apply `torchvision.transforms.v2.Normalize` |
| Kornia CPU | `kornia.io.load_image` with `kornia-rs`, CPU recipe in `float32`, cast the host batch to `float16`, pinned H2D transfer | Apply `kornia.enhance.Normalize` to the `float16` batch |
| TorchVision GPU | DataLoader workers read JPEG bytes and remove EXIF metadata to preserve the stored pixel orientation | TorchCodec decodes the batch with nvJPEG, TorchVision applies transforms per image, then normalizes the FP16 batch |
| Kornia GPU | DataLoader workers decode with `kornia.io.load_image` and `kornia-rs` and apply the shape-forming stages in FP32. The assembled FP16 batch uses pinned H2D transfer. | Apply remaining transforms and `kornia.enhance.Normalize` to the FP16 batch |
| DALI GPU | Native file reader and mixed JPEG decoder | Native DALI graph and `float16` normalization |

Source JPEGs vary in size. TorchVision GPU passes a list of JPEG bytes to TorchCodec,
then applies its shape-forming stages on the GPU per image before stacking. Kornia GPU
forms collatable samples in DataLoader workers.
CPU paths complete the image operations before transfer. The `Resize` recipe has one image operation, `Resize(224)`,
followed by GPU normalization and output preparation. Recipes such as `Pad` and
`LongestMaxSize` list the subsequent crop explicitly.

For a given operation, parameters are matched by hand for practical effect. APIs, interpolation,
random number generation, and rounding can differ, so pixel equality is outside the comparison
contract.

New recipes use one RGB JPEG per sample. Compare the same operation with fixed parameters or
matching sampling ranges and the same output shape. An adapter may compose native operations,
such as Pillow channel splitting and merging or a byte lookup table. Mosaic, reference-image
histogram matching, and box-aware crops require a different input protocol. ModeFilter is outside
the recipe selection. DALI's grid mask drops square patches, while AX GridMask drops stripes.

DALI uses ROI JPEG decoding when a recipe starts with a compatible crop.
`RandomCrop224` samples a separate window at native resolution for every image and adds
centered zero padding where a dimension is too short. `RandomResizedCrop` samples separate windows
inside the decoder and then resizes them. All readers use the pixel orientation stored in the file.
`Blur` uses a 5×5 box in every measured path. AX, Kornia, and DALI reflect the boundary; the
Pillow box filter repeats edge pixels. ChannelDropout zeros one uniformly sampled RGB channel.

`Resize` (R01) and `RandomResizedCrop` (R03) use bilinear interpolation without antialiasing. AX
uses OpenCV's `INTER_LINEAR`. DALI and TorchVision use `antialias=False`. Pillow and
Pillow-AllSIMD use `Image.transform(AFFINE, resample=BILINEAR)` because Pillow's `resize`
automatically smooths when downsampling. Kornia uses `align_corners=False` for both operations.
Its `Resize` explicitly sets `antialias=False`, and its `RandomResizedCrop` slices the crop and
resizes it without antialiasing. Crop selection and rounding may differ by library.

For R10 (Elastic), the AX, TorchVision, and Kornia parameters are
[matched by their displacement fields](rgb_elastic_calibration.md) at roughly 0.74 px RMS and a
spatial scale of about 10 px. DALI and Pillow omit this recipe.

R08 applies scale 2, rotation 25°, X/Y shears 10°/15°, and translation
20 px on each axis as one centered bilinear warp. R09 uses a centered
projective map with independent sign and a uniform magnitude in 0.05–0.10,
divided by crop width, for each axis of each image.

R24 samples an integer JPEG quality uniformly from 30 through 90 for each
image. TorchVision GPU encodes images in groups with the same sampled quality
and restores sample order after decoding. Kornia and DALI apply JPEG artifact
simulators. Nominal quality is shared across implementations and does not
calibrate distortion strength.

ColorJitter recipes sample factors and operation order per image. DALI uses
its native hue adjustment, an RGB-space approximation to HSV hue rotation.

R46 (Shear) applies a fixed 10° X shear about the image center on all eight paths.
In pixel-center coordinates its forward map is
`x' = x + tan(10°) * (y - (height - 1) / 2)`, `y' = y`.
The output keeps the input size and uses bilinear sampling with zero-valued samples
outside the frame. Pillow and Pillow-AllSIMD pad with one zero pixel before warping.
TorchVision and Kornia use FP32 sampling in their GPU Shear stage and convert back to
FP16 before normalization. Each implementation retains its own reader and crop path.

## Data access

After extraction, the VM reads all 10,000 JPEGs sequentially once outside the timed cells. All
implementations then use the same deterministic PCG64 permutation without replacement for each
seed. PyTorch DataLoader adds no shuffle. DALI receives the same preordered file list.

The selected files fit in the VM's 64 GiB host memory. Prewarming therefore lets the Linux page
cache serve the timed random reads in the usual case. The timer still includes file lookup,
cache access, decoding, and augmentation. It measures the warm cache-resident workload and does
not establish cold random disk I/O or physical disk bandwidth.

## Runs, VMs, and resume

`augbench launch-rgb` requires a clean Git worktree. It archives the current commit and creates
an immutable `run.json` from the family config, including the archive SHA-256 and selection
rule, plus the package lock, recipe catalog, and hardware config. It then:

1. Checks GCS for `runs/<run_id>/cells/*.json`.
2. Validates each existing cell's contents and identity.
3. Reuses an active VM with the same `augbench-run` label.
4. Otherwise, looks for zones with `g2-standard-16` and L4 availability and creates one Standard
   VM.
5. On the VM, downloads the code, raw dataset, and cached RGB environment once, verifies the
   Pillow-AllSIMD lock and wheel, extracts the first 10,000 sorted `val/*.JPEG` files, and
   prewarms them. If the run measures TorchVision GPU, bootstrap makes the installed NVIDIA
   driver's `libnvcuvid.so.1` available to that worker. Each
   implementation runs in its own process. It checks one batch through CUDA
   output per recipe, then runs the missing cells with the configured 15 workers. For preflight,
   paths other than DALI use one temporary DataLoader worker. DALI uses its native graph.
6. Writes an immutable GCS object immediately after each cell.
7. Saves bootstrap stdout and stderr in `runs/<run_id>/logs/` before the VM shuts down, whatever
   the exit status.
8. Before provisioning, removes only terminated or suspended `augbench` VMs so their boot disks
   release quota. Active VMs and GCS artifacts are preserved.

On the VM, each implementation job runs through all its recipes and seeds in sequence. Jobs are
never split by recipe, seed, duration, or arbitrary shard.

## Aggregation

Aggregate only after validating the entire matrix. Compare throughput and GPU memory row by row
with the same family, recipe, seed policy, batch size, dataset, hardware, and timing boundary.
Pairwise and all-library summaries use the exact intersection of supported recipes. The coverage
census is separate. Unsupported operations never count as speed losses.

A report may compare paths from separately validated runs with the same workload and hardware
settings. Each observation keeps its source run and cell IDs. For an invalid result or a
corrected recipe, freeze a correction run covering the affected implementation–recipe pairs
and all three seeds. Validate the replacement cells before regenerating the report. Keep the
original cells for reproduction.

The report's TorchVision GPU replacement uses a lock that adds TorchCodec to
the primary RGB environment. The generator verifies both lock checksums,
requires all other entries to match, and replaces the complete TorchVision
GPU selection. [Reproduction instructions](../paper/README.md#regenerate-the-results)

For a targeted run, `measurements` lists the implementation IDs to measure for each
recipe. The catalog still records full support. The paper generators take a complete
RGB run and can replace its AX observations with a complete AX-only run under the
same workload, hardware, and package locks.

## Adding a benchmark family

Before the first production matrix for nine-channel images, video, or volumes, add:

1. A prepared dataset archive and a compact, deterministic selection rule with one standardized
   sample format.
2. A family config with the output shape, dtype, batch size, seeds, and throughput units.
3. A recipe catalog and support matrix.
4. Implementations with GPU-only normalization and output validation.
5. A one-batch L4 preflight for each implementation–recipe pair. Paths other than DALI use one
   temporary DataLoader worker. DALI uses its native graph.
6. A new run and a complete matrix.

Each new family needs its own batch size, data format, recipes, and results.
