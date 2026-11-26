# Benchmark methodology FAQ

This FAQ explains the choices behind the RGB benchmark. The [execution contract](benchmark_execution_contract.md) specifies the current setup. The [result manifests](generated/results.json) identify the exact code and settings behind each published measurement, including older runs.

## Why measure from a JPEG on disk to a batch ready for the model?

A training loop needs ready batches. If the next batch arrives late, the model waits even when the augmentation kernel itself is fast. Libraries differ in where they decode, how they assemble batches, and when they transfer data to the GPU. The benchmark times those choices together, including file access, decoding, augmentation, batching, transfer, and normalization. For example, a pipeline that crops during JPEG decode can deliver a batch sooner even if its later transforms run at the same speed.

The output is a materialized CUDA FP16 batch in BCHW layout with three channels and 224×224 pixels. The runner synchronizes CUDA at the end of the timer so work already queued on the device counts. Training step time also depends on the model, its GPU use, and how training overlaps with input preparation. [Timing boundary](benchmark_execution_contract.md#timing-boundary)

## Why start with 8-bit RGB JPEGs from ImageNet validation?

JPEG input for RGB classification is a common workload with a large set of comparable operations across the libraries. ImageNet validation provides real files with varying dimensions and compression characteristics. A fixed selection from a checksum-verified archive gives each path the same bytes, and each seed gives all paths the same file order. We use the first 10,000 matching files in lexical order for reproducibility. This is one fixed ImageNet workload.

The 8-bit format describes the decoded source. A library can convert it to the dtype its operators expect, and every path delivers the same FP16 output. A workload that starts with FP32 or FP16 arrays changes which readers can participate, how many bytes move, and which kernels run. Nine-channel images, video, volumes, and sequences of volumes also need their own sample format, recipes, and batch size. Those comparisons have yet to be run. [RGB workload configuration](../configs/families/rgb.yaml) · [Requirements for a new workload](benchmark_execution_contract.md#adding-a-benchmark-family)

## Why compare augmentation libraries and preprocessing libraries?

ImageNet-style classification passes one 8-bit RGB image to the augmentation pipeline. This benchmark excludes bounding boxes, masks, keypoints, and `additional_targets`. A pipeline that transforms annotations or multiple related inputs has additional work and needs its own comparison.

AlbumentationsX, TorchVision, and Kornia offer higher-level augmentation APIs. For richer workloads, their ready-made transforms can keep images and supported annotations in sync, which reduces application code and the risk of mistakes. General-purpose dispatch and bookkeeping can also add overhead. We use stock AX transforms as the reference and adapt the TorchVision and Kornia paths to match their effect.

Pillow, Pillow-AllSIMD, and DALI serve as preprocessing libraries in this benchmark. Their primitives let us build a narrower pipeline with fewer layers of dispatch. For an image-only workload, that can improve speed. A richer workload can require application code for sampling, annotation geometry, and coordinated transforms. That code takes time to write, test, and maintain. The timing rows measure the input pipeline. The API census records direct, composed, and approximate matches. [Workload and reference implementations](benchmark_execution_contract.md#rgb-workload-and-reference-implementations) · [API census](../catalog/operations.yaml)

## What does one recipe test?

Each recipe has one target transformation. In `RandomCrop224 → Rotate`, the crop brings each source image to 224×224, with zero padding on short axes when needed. Rotate is the target. Throughput includes crop, padding, and the rest of the JPEG-to-CUDA pipeline. The same pass records peak process GPU memory. A recipe can also finish size preparation after its target, as in `Pad → RandomCrop224`.

In the Resize recipe, `Resize(224)` is itself the target and forms the required shape. The catalog records all preparation stages explicitly. [RGB recipe catalog](../catalog/recipes/rgb.yaml) · [Timing boundary](benchmark_execution_contract.md#timing-boundary)

## Why measure Colorize on grayscale JPEGs?

Colorize maps one intensity channel to RGB. The grayscale workload starts with one-channel JPEGs prepared offline from the same ImageNet validation images. The measured pipeline reads and decodes each grayscale JPEG, crops to 224×224 with padding when needed, runs Colorize, and normalizes the RGB batch on CUDA. Grayscale conversion takes place when the archive is built. The grayscale archive has its own checksum and run identity. [Grayscale recipe](../catalog/recipes/grayscale.yaml) · [Grayscale workload](benchmark_execution_contract.md#grayscale-colorize-workload)

## JPEG decoders

### How do we choose a decoder?

We start with the reader provided by the library or its I/O stack. This lets a path use features such as batch decoding on CUDA or cropping during decode. For a library that takes decoded images, we choose a loader that supplies the dtype and layout its transforms consume.

When we test multiple valid options for a path, we keep the one that delivers the highest measured throughput from JPEG on disk to a normalized CUDA batch. We record peak process VRAM in the same pass. The recipe's crop size, interpolation, sampling, padding, and output contract still apply when we change the decoder or move a stage between CPU and GPU. [Timing boundary and output contract](benchmark_execution_contract.md#timing-boundary)

| Benchmark path | Selected JPEG reader and decoder |
| --- | --- |
| AlbumentationsX CPU | SimpleJPEG in the benchmark loader. It supplies RGB `uint8` NumPy arrays to AX. |
| Pillow and Pillow-AllSIMD | Each wheel's own JPEG decoder through the Pillow image API. |
| TorchVision CPU | `torchvision.io.decode_image` on the CPU. |
| TorchVision GPU | TorchCodec batch decoding through nvJPEG on the L4. |
| Kornia CPU and GPU | `kornia.io.load_image` with `kornia-rs` on the CPU. |
| DALI GPU | DALI's file reader and mixed JPEG decoder, with decode-and-crop where the recipe allows it. |

The comparisons below use warm-cache ImageNet JPEGs on a Standard GCP `g2-standard-16` with one NVIDIA L4. Each configuration uses 256 images per batch and reports the median across the run's seeds. The frozen run identifies the exact decoder, code, and settings for each measurement. [Fixed RGB workload](benchmark_execution_contract.md#fixed-rgb-workload) · [Published result manifests](generated/results.json)

### Why keep SimpleJPEG for AlbumentationsX?

With SimpleJPEG, DataLoader workers decode and augment RGB on the CPU. The GPU receives the assembled `uint8` batch after Resize or Crop, about 36.75 MiB for 256 images at 224×224. With TorchCodec and nvJPEG, a batch of JPEGs decodes on the GPU. CPU workers need the full-resolution RGB back before running the same AX transform. That reverse transfer carries about 162 MiB per batch, and the finished host batch still goes to the GPU for normalization. [AX image loader](../src/augbench/datasets/loaders.py) · [AX stage profile](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/d94b2030d357c541c4f07a878fb74fa055d0be0c1b5fe689adbd1461c6781bb0?project=albumentations)

In a direct `Resize(224)` A/B, SimpleJPEG delivered **4,805 images/s** at **404 MiB** peak VRAM. The nvJPEG path delivered **2,883 images/s** at **1,098 MiB**, about 40% lower throughput. Both paths used AX Resize on the CPU and the same GPU normalization. The SimpleJPEG path used 15 DataLoader processes. The nvJPEG path used a batch decoder and 15 CPU threads for reading and augmentation, so the comparison covers the full input pipelines. [Direct AX decoder A/B](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/562ca6f032a2d97cfed4a157a58aa12f6cca1dd3d40e391e8e925d4a233e0709?project=albumentations)

We also packed the decoded images into one GPU buffer and copied that buffer into pinned host memory. The GPU-to-CPU stage runs from decoder return until the frames are ready for CPU transforms, including the time spent packing. Both copy modes used the same JPEG order and AX operations within each recipe. The `RandomCrop224` comparison also paired the crop windows. [Resize copy comparison](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/cf21229c8f9af55023e60930dae6576fd76e309fe142fad9f29b8237aec8ac4b?project=albumentations) · [RandomCrop224 copy comparison](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/a9c0a417c44fb478b12877e9a84dbdd187fdb0898fa844062041356998ade4f3?project=albumentations)

| AX recipe with nvJPEG | Copies per batch | GPU-to-CPU stage, ms/batch | End-to-end images/s | Peak process VRAM, MiB |
| --- | ---: | ---: | ---: | ---: |
| Resize | 256 | 23.6 | 2,960 | 1,098 |
| Resize | 1 | 18.3 | 3,016 | 1,410 |
| RandomCrop224 | 256 | 46.8 | 3,029 | 1,098 |
| RandomCrop224 | 1 | 26.5 | 3,170 | 1,410 |

Packing improved throughput by 1.9% on Resize and 4.7% on `RandomCrop224` and raised peak VRAM by 312 MiB for both recipes. SimpleJPEG remains the selected AX decoder.

### How do we handle EXIF orientation?

Dataset preparation removes EXIF APP1 headers once, as the selected JPEGs are extracted. Their compressed image data stays byte-for-byte identical, so every decoder receives the same stored pixel orientation. This covers Pillow, Pillow-AllSIMD, TorchVision, Kornia, DALI, and the AX decoder. TorchVision CPU and DALI also disable automatic orientation in their decoder calls. The prepared cache is shared across implementations and EXIF removal happens before timing. [Dataset preparation](../src/augbench/dataset_materialize.py) · [Data access contract](benchmark_execution_contract.md#data-access)

### Why do TorchVision GPU and Kornia GPU use different JPEG pipelines?

TorchVision benefits from GPU decode because its transforms can consume the decoded `uint8` CUDA tensors directly. DataLoader workers read the prepared JPEG bytes. The main process passes the JPEGs to TorchCodec in one batch call. TorchVision applies Resize or Crop on the GPU, stacks the equal-size images, and normalizes the FP16 batch. This path keeps decoded RGB on the GPU from decode through normalization. [TorchVision adapter](../src/augbench/implementations/torchvision_pipeline_impl.py) · [TorchCodec JPEG decoding](https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/image_decoding.html)

On `RandomCrop224`, the earlier hybrid path using `torchvision.io.decode_jpeg` delivered **495 images/s**. The targeted run with TorchCodec and hardware nvJPEG delivered **4,198 images/s**. The subsequent 44-recipe TorchVision GPU run measured **4,464 images/s** on `RandomCrop224` and **3,425 images/s** on Resize. We use the TorchCodec path for TorchVision GPU. [Earlier TorchVision run](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/9c2a6d0b54469dbd1f14833b54e7f4d0758a4242a25673ecdd2d13ff32fd64f3?project=albumentations) · [Targeted TorchCodec run](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/df3a0bb0c948f7e418d1a2f93a830d1b71917c70983c16f63e73210b3b4e8bab?project=albumentations) · [Full TorchVision GPU run](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/d1235938fea1d0333e5a503343b502e067644849dc659f5f2b28ef86a29cd067?project=albumentations)

Kornia GPU uses `kornia-rs` to decode on the CPU. Workers crop or resize the FP32 images to make them stackable, and the assembled batch is converted to FP16 before the pinned transfer. The GPU runs any remaining transforms and normalizes the batch. In the TorchCodec trial, we moved crop to CUDA as well. Each decoded full-resolution image was converted to FP32 on the GPU before Kornia cropped it, increasing the amount of data handled on the GPU before the spatial dimensions shrank. [Kornia I/O](https://kornia.readthedocs.io/en/stable/io.html) · [Kornia adapter](../src/augbench/implementations/kornia_pipeline_impl.py)

On `RandomCrop224`, the native Kornia path delivered **2,255 images/s** at **464 MiB** peak VRAM. The TorchCodec and CUDA crop trial delivered **423 images/s** at **1,440 MiB**. The trial measures the combined effect of switching the decoder, moving the crop, and converting full-resolution tensors on the GPU. We kept CPU decode and shape preparation for Kornia GPU. [Kornia native path](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/9c2a6d0b54469dbd1f14833b54e7f4d0758a4242a25673ecdd2d13ff32fd64f3?project=albumentations) · [Kornia CUDA decode and crop trial](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/cbf8f5f31dbfd3138343afe9d6b43b16095b8a9473b6c5dcb0db1c79b14d9fe2?project=albumentations)

### When does DALI crop during JPEG decoding?

We use DALI's region-of-interest decoder when the recipe starts with a compatible crop. For `RandomCrop224`, the adapter supplies a separate window for each image and adds centered zero padding on short axes. For `RandomResizedCrop`, DALI samples the crop in the decoder and resizes the result. [DALI execution path](benchmark_execution_contract.md#timing-boundary) · [DALI region-of-interest decoding](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.image_slice.html)

## Why apply TorchVision GPU transforms per image?

The recipes require an independent random draw for each image. A single call to a transform such as TorchVision v2 `RandomResizedCrop` on a batch tensor samples one set of crop parameters for that call. Images in the same batch would therefore share the crop. The adapter calls the transform separately for each image. This also lets it crop or resize variable-sized decoded images before stacking them. Two independent draws can still happen to select the same parameters.

Batching is appropriate when it preserves that independence. TorchCodec decodes the input JPEGs in one CUDA call, and TorchVision normalizes the assembled batch in one call. For JPEG recompression, the adapter draws an integer quality from 30 through 90 for each image, groups images that drew the same quality for encoding, and restores their original order after decoding. The CUDA encoder accepts one quality per call. [TorchVision v2 transform source](https://docs.pytorch.org/vision/stable/_modules/torchvision/transforms/v2/_transform.html) · [CUDA JPEG encoding](https://docs.pytorch.org/vision/0.29/generated/torchvision.io.encode_jpeg.html) · [Benchmark adapter](../src/augbench/implementations/torchvision_pipeline_impl.py)

## Why disable antialiasing for bilinear resize?

The resize and random-resized-crop recipes compare bilinear resampling. OpenCV exposes `INTER_LINEAR` without a separate antialiasing option. `INTER_AREA` is available for shrinking and uses a different resampling method. Filtering only on paths that enable it by default would change both the output and the amount of work being timed.

We request bilinear resampling without an antialiasing filter in these recipes. The Pillow paths use their native bilinear affine resampler because `Image.resize` with bilinear resampling filters when shrinking. Comparing antialiased resizing would require specifying a filter that all paths can match. [OpenCV resize modes](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) · [TorchVision resize options](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html) · [Resize details](benchmark_execution_contract.md#timing-boundary)

## How do we choose equivalent recipes across libraries?

Use the native AlbumentationsX recipe as the reference. Each AX stage calls a stock transform and lets its sampler run. TorchVision and Kornia run native operators with parameters chosen to approximate the AX effect. Their algorithms and their costs are part of the comparison. Pillow, Pillow-AllSIMD, and DALI can compose native primitives into the target effect.

Shared seeds align input order. The augmentation or its adapter draws parameters for each image during execution. For example, the TorchVision adapter selects a quarter-turn angle and passes it to native `rotate`. If the library lacks an operator for the target, the path is unsupported. This applies to Transpose, LetterBox, and Emboss in TorchVision and Kornia. R63 is called `RGB <-> BGR` because it always exchanges red and blue. R12 samples from all six RGB channel orders. [Recipe catalog](../catalog/recipes/rgb.yaml) · [Aggregation rules](benchmark_execution_contract.md#aggregation)

The adapter inventory shows where TorchVision and Kornia add sampling, convert parameters, or change how images enter a native call. It also traces JPEG decoding, transfers, and normalization. [TorchVision and Kornia adapter inventory](torchvision_kornia_adapters.md)

### What did the calibration match?

The figures below are CPU checks on 16 ImageNet validation crops at 224×224 with seed 137. For pixel effects, RMS measures how far the output moves from its own input, in 0–255 units. It does not measure pixel agreement between independent draws or throughput.

| Recipe | Calibration and remaining differences |
| --- | --- |
| GaussianNoise (R16) | AX and Kornia draw zero-mean noise with σ=0.44 on a 0–1 scale. AX clips and returns uint8. The Kornia adapter clamps the native output to [0, 1] before batch normalization. On 16 ImageNet images resized and center-cropped to 224×224, pixel RMS was 82.66 for AX and 82.92 for Kornia. |
| CLAHE (R29) | AX applies CLAHE to Lab lightness. Kornia applies it per RGB channel with clip range 1.0–3.5, and DALI applies it to luma. AX and Kornia changed Lab lightness by an average of 7.5 and 7.2 units respectively. The mean Lab chroma shift was 0.7 and 7.5 units respectively. |
| Corner and Gaussian Illumination (R35–36) | Kornia uses gain 0.006–0.12. Pixel RMS was 9.0 and 9.4 in AX and 8.8 and 9.2 in Kornia. The AX effects multiply pixel values. Kornia adds a field and can affect black pixels. |
| PlasmaBrightness and PlasmaShadow (R38, R40) | Tuning the native Kornia intensity and shadow threshold gave mean absolute changes of 23.0 and 27.9. AX measured 22.7 and 28.5. Kornia can change each RGB channel separately for brightness and uses a thresholded mask for shadow. |
| Rain (R41) | AX draws 74 thick, blurred additive streaks. Native Kornia overwrites pixels with 1,150 thin streaks of a similar length and slant. Pixel RMS was 73.5 in AX and 75.7 in Kornia. The streak texture differs. |
| ThinPlateSpline (R47) | Both paths use four corners and the center. AX draws Gaussian offsets. Kornia uses its own uniform sampler with scale 0.173. Both give a per-axis anchor displacement standard deviation of about 11.2 px. The native warps preserve their own map conventions. |
| Pillow blur and sharpness (R15, R32, R55–57) | The Pillow paths reflect-pad before filtering. GaussianBlur uses a sampled σ and a finite 5×5 kernel. At σ=2, both outputs matched on all 16 crops. At α=0.75, the edge and detail enhancement filters each differed by under 0.06 intensity units MAE. UnsharpMask at the five-tap setting and α=0.35 differed by 0.19 units. |

JPEG quality is a control for the selected codec. Kornia maps the sampled nominal qualities 30, 70, and 90 to about 24, 63, and 89 to match the strength of AX's compression artifacts. Kornia's differentiable simulator and DALI's artifact simulator retain their own decoding and quantization behavior. [Recipe catalog](../catalog/recipes/rgb.yaml)

For Perspective (R09), the sampled inner quadrilateral is the source. The native warp expands it into the output canvas. In a separate CPU check on 24 white inputs, AX, TorchVision, and Kornia all returned zero fully black pixels. With the previous corner order, TorchVision and Kornia left 20% and 22% of the canvas black. [Geometry contract](benchmark_execution_contract.md#timing-boundary)

Some native operators differ substantially after tuning. Kornia PlasmaContrast exposes roughness and has no contrast-strength parameter. Its pixel RMS was 36.5 versus 5.6 in AX.

### What is the reference for OpticalDistortion (R45)?

The reference is OpenCV's `cv2.fisheye` camera model. AX approximates its radial distortion with a signed polynomial. On a 224×224 grid, AX at `k=-0.30` differed by 3.7 px RMS from the OpenCV map with a centered principal point, a focal length of 120 px, and zero distortion coefficients. [OpenCV camera model](https://docs.opencv.org/doc/doxygen/html/d2/d14/namespacecv_1_1fisheye.html) · [AX radial map](https://github.com/albumentations-team/AlbumentationsX/blob/2.6.3/albumentations/augmentations/geometric/_functional_distortion.py)

AX samples its radial coefficient uniformly from −0.3 to 0.3. The sign allows inward and outward shifts, and this range keeps the radial map monotonic across the crop. The benchmark runs Kornia's native `RandomFisheye` on CPU and GPU as an approximate match. Kornia samples an exponent from 45 to 90 and moves source coordinates outward. On the 16 CPU calibration crops, 6.9% of AX output pixels and 27.1% of Kornia output pixels were fully black. [AX adapter](../src/augbench/implementations/albumentationsx_impl.py#L361) · [Kornia adapter](../src/augbench/implementations/kornia_impl.py#L437)

## Why normalize on the GPU and return FP16?

Normalizing on the GPU gives every path the same final device and output contract. A path that augments `uint8` images on the CPU can transfer one byte per channel value, then convert and normalize on CUDA. The cost of making the normalized FP16 batch ready for the model stays in the measurement. Intermediate dtypes and arithmetic follow each implementation's documented path. [Execution paths](benchmark_execution_contract.md#timing-boundary)

AX ends its CPU `Compose` with `ToTensorV2`, which converts each HWC `uint8` image to a CHW tensor. Workers stack the tensors into a BCHW batch. After the `uint8` batch transfer, AX's `InputNormalizer` normalizes it on CUDA with no layout change. The catalog's `Normalize` and `ToTensor` entries configure the batch result. The CPU image pipeline runs the listed augmentations followed by `ToTensorV2`. [AX batch path](../src/augbench/implementations/albumentationsx_pipeline_impl.py)

## Why randomize image order after reading the files into RAM?

Random access to a cold disk adds seek latency and random I/O throughput to the measurement. Those can change the result even when the image libraries do the same amount of work. Before timing, the runner reads the selected 10,000 JPEG files sequentially once. Their combined size is less than the 6.28 GiB ImageNet validation archive, and the VM has 64 GiB of host memory, so the Linux page cache can hold the selection. Each timed seed still visits files in a seeded random order without replacement. Timed file opens and page cache lookups remain, along with JPEG decoding and augmentation. The result describes a warm cache-resident workload. It does not describe cold-cache or remote-storage performance. [Dataset selection](../configs/families/rgb.yaml) · [Data access contract](benchmark_execution_contract.md#data-access)

## Why hold the machine, batch size, workers, and prefetch settings fixed?

All paths use the same file order for a given seed, a batch size of 256, and the same L4 machine type. The timer starts after one warm-up batch to keep pipeline setup out of the throughput number. We also fix the worker and prefetch settings for the DataLoader paths. DALI uses its native pipeline, so a prefetch depth of two there and a prefetch factor of two per DataLoader worker describe different queues. Fixing the settings gives a repeatable comparison at one operating point. Tuning the worker count, batch size, or accelerator for a particular deployment can change the ranking. [Fixed workload](benchmark_execution_contract.md#fixed-rgb-workload)

## Why measure throughput and GPU memory together?

Some pipelines use more VRAM to deliver batches faster. Measuring both in one pass ties the two numbers to the same decoder, settings, and batch stream. The process-level memory reading covers CUDA allocations made by the decoder and libraries as well as tensors tracked by PyTorch. The export includes both the baseline before pipeline construction and the observed peak. Polling can miss a short-lived spike. [Measurements per cell](benchmark_execution_contract.md#what-one-cell-measures)

## Why compare only shared recipes and report API coverage separately?

A speed ratio is meaningful when both paths execute the same workload. Summary charts use the recipes shared by the paths shown, and the notes describe where those implementations still differ. Coverage answers a second question with its own denominator. The census starts from selected AlbumentationsX operations and records counterparts in the other libraries, including DALI. It distinguishes direct operators, compositions, and approximate matches. This shows which effects have a counterpart even when they lack a timing row. [API census](../catalog/operations.yaml) · [Aggregation rules](benchmark_execution_contract.md#aggregation)
