# Production benchmark contract

This repository measures the path from a prepared file on local disk to a batch ready for a
model on the GPU. The image runner supports RGB, a separate grayscale Colorize workload, and an
ImageNet training-step workload. Nine-channel images, video, and volumes need their own dataset,
recipe catalog, output contract, and batch size.

The [methodology FAQ](benchmark_faq.md) explains the choices behind this contract.
The [TorchVision and Kornia adapter inventory](torchvision_kornia_adapters.md) lists
the work each adapter adds around the native operators.

## RGB workload and reference implementations

Each sample starts with one 8-bit RGB JPEG, as in ImageNet image classification. The augmentation
call receives only the image. Bounding boxes, masks, keypoints, `additional_targets`, and other
annotations are outside this workload. Each library can use its native working dtype after decode.

AlbumentationsX, TorchVision, and Kornia are the augmentation libraries in this comparison. Their
higher-level transforms provide ready-made behavior for richer workloads, including annotation
handling where supported. That API can add bookkeeping and dispatch overhead in image-only use.
Pillow, Pillow-AllSIMD, and DALI provide the preprocessing primitives in this comparison. Using
lower-level primitives can reduce overhead, and richer workloads require more application code to
implement, test, and maintain.

Each AX stage is one stock transform with its public parameters. The top-level `A.Compose` runs the
stages listed in the catalog. Declare any required input preparation as its own stage.
Keep `A.Lambda`, `A.OneOf`, and `A.Sequential` out of the AX adapter. The one exception is the
`RandAugment` stage of the training workload, which AX assembles as one `A.SomeOf` of stock transforms.
AX owns image processing and parameter sampling for its path. Choose native transforms
or functional operators in TorchVision and Kornia. Tune their parameters against the AX
effect and retain their algorithms and kernel costs. Sample parameters in the adapter
when the native operator accepts values and provides no sampler for them. Leave a path
unsupported when it would require implementing the target effect in the benchmark. Assemble Pillow,
Pillow-AllSIMD, and DALI stages from native primitives as needed.

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
| Recipes | 72 RGB recipes, each supported by at least two libraries |
| Matrix | 421 supported implementation–recipe pairs × 5 seeds = 2,105 cells |
| Seeds | 137, 138, 139, 140, 141 |
| Batch size | 256 images |
| Warm-up | 1 full batch outside the timer |
| Measurement | 32 full batches, 8,192 images |
| Workers | 15, `persistent_workers=True`, `prefetch_factor=2`, `pin_memory=True` |
| Hardware | Standard `g2-standard-16` VMs, each with one NVIDIA L4, in any eligible GCP zone. A run uses up to 20 VMs at once |

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
as throughput. Before sampling the baseline, the runner releases unused PyTorch CUDA cache
blocks and, for DALI, unused blocks in DALI's shared memory pools. This also clears cached
blocks left by preflight. The peak includes persistent CUDA state. Subtracting the baseline
from the peak gives the increase during the cell.

Mean/std normalization runs on the GPU on every path. The catalog entries `Normalize` and
`ToTensor` specify batch statistics, layout, and dtype. The `recipe_id` names the image
operations, for example `Resize224+RGBToBGR`. The per-image CPU recipe contains only
image operations. The AX adapter ends its `A.Compose` with `ToTensorV2`, which turns each HWC
`uint8` image into a CHW tensor. Workers stack the tensors into a BCHW batch and transfer it as
`uint8`. On CUDA, the native `InputNormalizer` returns normalized FP16 with no layout change.
Each completed run retains the implementation in its frozen code archive.

| Implementation | Before GPU transfer | On the GPU |
| --- | --- | --- |
| AlbumentationsX CPU | SimpleJPEG, `uint8` recipe, `ToTensorV2`, pinned H2D transfer in `uint8` | `albumentations.batch.InputNormalizer` with FP32 arithmetic and FP16 output |
| Pillow CPU | Pillow decode, `uint8` recipe, pinned H2D transfer in `uint8` | Convert to `float16` and normalize |
| Pillow-AllSIMD CPU (fast) | Same code with the Pillow-AllSIMD wheel in a separate process | Convert to `float16` and normalize |
| TorchVision CPU | `torchvision.io`, `uint8` recipe, pinned H2D transfer in `uint8` | Convert the batch to `float16` and apply `torchvision.transforms.v2.Normalize` |
| Kornia CPU | `kornia.io.load_image` with `kornia-rs`, CPU recipe in `float32`, cast the host batch to `float16`, pinned H2D transfer | Apply `kornia.enhance.Normalize` to the `float16` batch |
| TorchVision GPU | DataLoader workers read the prepared JPEG bytes | TorchCodec decodes the batch with nvJPEG, TorchVision applies transforms per image, then normalizes the FP16 batch |
| Kornia GPU | DataLoader workers decode with `kornia.io.load_image` and `kornia-rs` and apply the shape-forming stages in FP32. The assembled FP16 batch uses pinned H2D transfer. | Apply remaining transforms and `kornia.enhance.Normalize` to the FP16 batch |
| DALI GPU | Native file reader and mixed JPEG decoder | Native DALI graph and `float16` normalization |

Source JPEGs vary in size. TorchVision GPU passes a list of JPEG bytes to TorchCodec,
then applies its shape-forming stages on the GPU per image before stacking. Kornia GPU
forms collatable samples in DataLoader workers.
CPU paths complete the image operations before transfer. The `Resize` recipe has one image operation, `Resize(224)`,
followed by GPU normalization and output preparation. Recipes such as `Pad` and
`LongestMaxSize` list the subsequent crop explicitly.

Set the available geometry, interpolation, boundary fill, and sampling parameters against AX.
When the native operators differ, compare an observable property such as displacement or the
strength of the pixel change. Record any material difference the public parameters leave.

Compare the same operation with fixed parameters or matching sampling ranges and the same output
shape. A preprocessing adapter may compose native operations such as Pillow channel splitting
and merging or a byte lookup table. Mosaic, reference-image
histogram matching, and box-aware crops require a different input protocol. ModeFilter is outside
the recipe selection. DALI's grid mask drops square patches, while AX GridMask drops stripes.

Each stochastic stage samples independently for every image during recipe execution. A
batched implementation may draw all its parameters in one call. When a native kernel takes
one parameter for the whole call, the adapter may group images after drawing their parameters
and restore the original image order. Sampling and any grouping run inside the input pipeline.

DALI uses ROI JPEG decoding when a recipe starts with a compatible crop.
`RandomCrop224` samples a separate window at native resolution for every image and adds
centered zero padding where a dimension is too short. `RandomResizedCrop` samples separate windows
inside the decoder and then resizes them. All readers use the pixel orientation stored in the file.
`Blur` samples a 3×3 or 5×5 box with equal probability for each image. AX, Kornia, and DALI
reflect the boundary. The Pillow paths reflect-pad before their box filter. ChannelDropout zeros one
uniformly sampled RGB channel.

R29 (CLAHE) uses an 8×8 grid. AX processes Lab lightness with a clip limit drawn from 1.0 to 4.0.
Kornia processes RGB channels independently with a clip limit drawn from 1.0 to 3.5.
DALI processes luma and groups draws that yield the same integer histogram threshold.
R63 (`RGB <-> BGR`) uses the fixed order `[2, 1, 0]`. R12 (ChannelShuffle) uniformly
chooses from all six orders for each image.

`Resize` (R01) and `RandomResizedCrop` (R03) use bilinear interpolation without antialiasing. AX
uses OpenCV's `INTER_LINEAR`. DALI and TorchVision use `antialias=False`. Pillow and
Pillow-AllSIMD use `Image.transform(AFFINE, resample=BILINEAR)` because Pillow's `resize`
automatically smooths when downsampling. Kornia uses `align_corners=False` for both operations.
Its `Resize` explicitly sets `antialias=False`, and its `RandomResizedCrop` slices the crop and
resizes it without antialiasing. Crop selection and rounding may differ by library.

For R10 (Elastic), AX, TorchVision, and Kornia draw a new field and a uniform amplitude
multiplier from 0.5 to 1.0 for each image. The upper endpoint uses the
[calibrated displacement fields](rgb_elastic_calibration.md) at roughly 0.74 px RMS and a
spatial scale of about 10 px. DALI and Pillow omit this recipe.

R07 (Rotate) samples an angle from −45° to +45° and performs one bilinear warp at
the original crop size, with zero fill and no separate antialiasing pass.

R08 (Affine) samples one isotropic scale from 0.75–1.25, rotation from ±15°, two shears from ±8°,
and two shifts from ±12 px. ShiftScaleRotate uses scale 0.875–1.125, rotation ±12°, shifts ±14
and ±7 px, and zero shear. TorchVision and Kornia use native `RandomAffine` and express the
shift limits as fractions of the 224-pixel crop. The libraries use their native samplers and
matrix conventions.

R09 uses native AlbumentationsX `Perspective` as the reference. Each image draws a
standard deviation from 0.05–0.10 and eight normal offsets, which move the four corners
inward. Pillow and DALI use those corner and output-scale rules with their native warps.
TorchVision samples and rounds the inner corners, then passes them to its native
`perspective`. Kornia samples its corners with native `RandomPerspective` at distortion
scale 0.24. Both pass the inner quadrilateral as the source and adjust the destination
extent to keep the original canvas size. Kornia retains its uniform corner sampler.

R24 draws JPEG quality independently for each image, uniformly from the integers
30 through 90 inclusive. TorchVision GPU encodes images in groups with the same
sampled quality and restores sample order after decoding. Kornia maps the sampled
quality to its differentiable codec using the calibration points 30→24, 70→63,
and 90→89. At the same nominal quality, differences in chroma subsampling,
quantization, and rounding can change the resulting artifacts.

DALI applies its native `jpeg_compression_distortion` GPU operator to the decoded
224×224 RGB crop. The kernel subsamples chroma, applies block DCT and quantization,
and reconstructs RGB directly without entropy coding or a JPEG bitstream. For
a fixed crop size, `quality` changes the quantization tables. The kernel processes
the same number of blocks and performs the same operations at every quality.
Reading and decoding the source JPEG still count toward the measured throughput.
[DALI operator](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.jpeg_compression_distortion.html) ·
[DALI 2.4.0 GPU kernel](https://github.com/NVIDIA/DALI/blob/v2.4.0/dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_impl.cuh)

ColorJitter recipes sample factors and operation order per image. DALI uses
its native hue adjustment, an RGB-space approximation to HSV hue rotation.

R46 (Shear) samples an X shear angle uniformly from −10° to +10° per image. TorchVision and
Kornia use native `RandomAffine` with rotation, shift, scale, and Y shear fixed at identity.
The output keeps its size and uses bilinear sampling and a zero border. Pillow and
Pillow-AllSIMD add one zero pixel before warping.

R71 (PixelSpread) uses two per-pixel integer draws in `[-2, 2]` for spatial offsets, shared
by all three channels. DALI uses nearest-neighbor remap with zero outside the source.

R52 (Transpose), R61 (LetterBox), and R70 (Emboss) use AX, Pillow, Pillow-AllSIMD,
and DALI. Kornia uses native `PadTo` for R62 (PadIfNeeded), which pads the right and bottom
edges. R06 (Pad) and R60 (CropAndPad) require independent control of each border and
omit Kornia.

## Grayscale Colorize workload

Prepare one-channel `uint8` JPEGs offline from ImageNet validation. The converter preserves
the source dimensions and file names, re-encodes at JPEG quality 95, and reports the archive
SHA-256. The grayscale recipe begins with a native grayscale decode, crops to 224×224, and
then colorizes to RGB. The output uses the same CUDA batch contract as RGB. The six eligible
paths are AX, Pillow, Pillow-AllSIMD, Kornia CPU, Kornia GPU, and DALI. Results use independent
run and cell identities and a separate catalog.

```bash
uv run augbench prepare-grayscale --source-archive val.tar --output-archive val-grayscale.tar
uv run augbench launch-grayscale --archive-uri "$GRAYSCALE_ARCHIVE_URI" --archive-sha256 "$GRAYSCALE_ARCHIVE_SHA256"
```

The launcher takes the uploaded grayscale archive URI and the SHA-256 printed by preparation.
It uses the RGB selection policy and execution settings. A production launch verifies the archive
and preflights each path on CUDA before collecting throughput and peak VRAM.

## ImageNet training workload

This workload answers one question: does the loader ranking survive when a model trains on the same
GPU and the same CPU cores? It reuses the eight input paths, the timing boundary, and the RGB output
contract. Each ready batch feeds one training step, and the cell reports the rate of those steps.

| Field | Value |
| --- | --- |
| Family | `imagenet_train`; one run per model (`configs/families/train-resnet18.yaml`, `train-resnet50.yaml`) |
| Data | All 50,000 ImageNet validation JPEGs; one pass per seed, no rollover |
| Models | ResNet-18 (batch 256) and ResNet-50 (batch 128), `torchvision.models`, untrained |
| Step | `torch.compile`, bfloat16 autocast without `GradScaler`, `channels_last`, fused SGD, cross-entropy on fixed random labels |
| Window | 10 warm-up and 150 measured batches; 15 workers, prefetch 2, persistent workers, pinned memory |
| Recipes | `RandomResizedCrop+HorizontalFlip`; `+ColorJitter`; `+ColorJitter+Erasing`; `RandomResizedCrop+HorizontalFlip+RandAugment`. Stage definitions come from the RGB catalog. Pillow paths lack ColorJitter and Erasing |
| Cells | 5 seeds × (8 + 6 + 6 + 8) pairs = 140 per run |

The step runs on the delivered batch with no extra copy: the loader's normalized `float16` BCHW batch
goes into autocast. The labels are random and fixed per seed because the cell measures speed. It makes no
claim about accuracy or convergence.

A cell records, in one pass:

- throughput of training steps over the measured window (images/s);
- the model-only ceiling: the same step on one resident synthetic batch, measured in the same process before
  the loader opens, which also compiles the model. Ceiling and cell share the machine, so machine
  differences cancel in their ratio;
- loader wait: the seconds the training loop blocked on the next batch during the window. Work that a path runs
  on the GPU after the batch arrives (Kornia GPU, TorchVision GPU) belongs to the step, not to the wait, and DALI's
  wait includes its GPU work for that batch;
- two GPU-memory peaks: the contract peak from pipeline construction to the end, and the peak after
  the ceiling phase, which leaves out the compilation spike;
- a finite loss at the end of the window. A non-finite loss fails the cell.

### RandAugment

Every path applies two operations per image at magnitude 9 of 31 bins, with bilinear warps and zero fill.
TorchVision (`v2.RandAugment`), Kornia (`kornia.augmentation.auto.RandAugment`), and DALI
(`nvidia.dali.auto_aug.rand_augment`) run their native policy. AX and the two Pillow paths assemble it
from stock operations: AX as one `A.SomeOf(n=2, replace=True)` in which each signed operation is an
`A.OneOf` of its two signs, Pillow from `ImageOps`, `ImageEnhance`, and `Image.transform`.

TorchVision, DALI, and the assembled paths share one pool of 14 operations (Identity, ShearX, ShearY,
TranslateX, TranslateY, Rotate, Brightness, Color, Contrast, Sharpness, Posterize, Solarize, AutoContrast,
Equalize), draw each operation with replacement, and pick the sign of a signed operation at random for each
application. At this magnitude they give shear 0.09, translation 30 px, rotation 9°, enhancement factors
1 ± 0.27, 7 posterize bits, and a solarize threshold at 70% of the maximum. The assembled paths take these
values from TorchVision and a test checks each operation against TorchVision's output on a 224×224 image
(mean absolute difference at most 1.5 of 255; the largest observed is 0.9).

Differences that remain:

- Kornia draws its operations without replacement from its own pool at one global magnitude (`m=9` of 30)
  and one sequence per call, so the adapter calls it once per image.
- AX shears about the image center and TorchVision about the top-left corner, so AX recenters the shear.
  AX Sharpness is `UnsharpMask` with a signed alpha (supported since AX 2.6.9); its 3×3 Gaussian differs
  slightly from Pillow's smoothing kernel.
- TorchVision truncates the translation to whole pixels (30), DALI uses 100/224 of the side, and Equalize
  differs between Pillow-style, OpenCV-style, and Kornia's own.

The worker keeps one compiled model per process. Each cell restores the initial weights in place and starts
a new optimizer, so compilation happens once per job. The VM keeps the Inductor cache for later jobs and
refuses to start when it has no C compiler.

## Data access

The VM removes EXIF APP1 headers from the selected JPEGs during extraction. This copies the
compressed image data into one shared cache while preserving the orientation of the stored pixels.
The preparation version is part of the cache key. Every implementation reads the same prepared
JPEGs, including TorchCodec, which normally applies EXIF orientation.
A CMYK or YCCK JPEG is re-encoded as RGB at JPEG quality 95 during preparation, because some decoders reject
four-component files and the others convert them differently. The first 10,000 validation files hold none, so
the RGB workload is unaffected; the full 50,000-file training workload holds at least one. Other files are copied unchanged.

After preparation, the VM reads all 10,000 JPEGs sequentially once outside the timed cells. All
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
3. Counts the active VMs with the same `augbench-run` label and releases claims that belong to VMs that no longer exist.
4. Looks for zones with `g2-standard-16` and L4 availability and creates Standard VMs until the run has
   `min(max_vms, pending jobs)` of them. A job is one implementation and seed. Creation rotates the zone
   order per VM, and a quota or capacity error moves on to the next zone.
5. On the VM, downloads the code, raw dataset, and cached RGB environment once, verifies the
   Pillow-AllSIMD lock and wheel, extracts the first 10,000 sorted `val/*.JPEG` files, and
   prewarms them. If the run measures TorchVision GPU, bootstrap makes the installed NVIDIA
   driver's `libnvcuvid.so.1` available to that worker. Each
   VM then takes jobs from the shared queue: it claims the next unclaimed job with a conditional
   `runs/<run_id>/claims/<implementation>-<seed>.json` create, runs it in its own process, and
   claims the next job when it finishes. A job checks one batch through CUDA output per recipe,
   then runs its missing cells with the configured 15 workers. For preflight,
   paths other than DALI use one temporary DataLoader worker. DALI uses its native graph.
6. Writes an immutable GCS object immediately after each cell.
7. Saves bootstrap stdout and stderr in `runs/<run_id>/logs/` before the VM shuts down, whatever
   the exit status.
8. Before provisioning, removes only terminated or suspended `augbench` VMs so their boot disks
   release quota. Active VMs and GCS artifacts are preserved.

A job runs one implementation and one seed through all its recipes in sequence on one VM. Jobs are
never split by recipe, duration, or arbitrary shard. A VM that ends while holding a claim leaves the
claim in place. The next launch releases it and reruns only the job's missing cells.

## Aggregation

Aggregate only after validating the entire matrix. Compare throughput and GPU memory row by row
with the same family, recipe, seed policy, batch size, dataset, hardware, and timing boundary.
Pairwise and all-library summaries use the exact intersection of supported recipes. The coverage
census is separate. Unsupported operations never count as speed losses.

A report may compare paths from separately validated runs with the same workload and hardware
settings. Each observation keeps its source run and cell IDs. For an invalid result or a
corrected recipe, freeze a correction run covering the affected implementation–recipe pairs
and all seeds. Validate the replacement cells before regenerating the report. Keep the
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
