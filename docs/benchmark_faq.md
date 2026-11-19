# Benchmark methodology FAQ

This FAQ explains the choices behind the RGB benchmark. The [execution contract](benchmark_execution_contract.md) specifies the current setup. The [result manifests](generated/results.json) identify the exact code and settings behind each published measurement, including older runs.

## Why measure from a JPEG on disk to a batch ready for the model?

A training loop needs ready batches. If the next batch arrives late, the model waits even when the augmentation kernel itself is fast. Libraries differ in where they decode, how they assemble batches, and when they transfer data to the GPU. The benchmark times those choices together, including file access, decoding, augmentation, batching, transfer, and normalization. For example, a pipeline that crops during JPEG decode can deliver a batch sooner even if its later transforms run at the same speed.

The output is a materialized CUDA FP16 batch in BCHW layout with three channels and 224×224 pixels. The runner synchronizes CUDA at the end of the timer so work already queued on the device counts. Training step time also depends on the model, its GPU use, and how training overlaps with input preparation. [Timing boundary](benchmark_execution_contract.md#timing-boundary)

## Why start with 8-bit RGB JPEGs from ImageNet validation?

JPEG input for RGB classification is a common workload with a large set of comparable operations across the libraries. ImageNet validation provides real files with varying dimensions and compression characteristics. A fixed selection from a checksum-verified archive gives each path the same bytes, and each seed gives all paths the same file order. We use the first 10,000 matching files in lexical order for reproducibility. This is one fixed ImageNet workload.

The 8-bit format describes the decoded source. A library can convert it to the dtype its operators expect, and every path delivers the same FP16 output. A workload that starts with FP32 or FP16 arrays changes which readers can participate, how many bytes move, and which kernels run. Nine-channel images, video, volumes, and sequences of volumes also need their own sample format, recipes, and batch size. Those comparisons have yet to be run. [RGB workload configuration](../configs/families/rgb.yaml) · [Requirements for a new workload](benchmark_execution_contract.md#adding-a-benchmark-family)

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

The comparisons below use warm-cache ImageNet JPEGs on a Standard GCP `g2-standard-16` with one NVIDIA L4. Each configuration uses 256 images per batch and reports the median across three seeds. The frozen run identifies the exact decoder, code, and settings for each measurement. [Fixed RGB workload](benchmark_execution_contract.md#fixed-rgb-workload) · [Published result manifests](generated/results.json)

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

### Why do TorchVision GPU and Kornia GPU use different JPEG pipelines?

TorchVision benefits from GPU decode because its transforms can consume the decoded `uint8` CUDA tensors directly. DataLoader workers read JPEG bytes and remove the EXIF orientation metadata so decoding preserves the stored pixel orientation. The main process passes the JPEGs to TorchCodec in one batch call. TorchVision applies Resize or Crop on the GPU, stacks the resulting 224×224 images, and normalizes the FP16 batch. This path keeps decoded RGB on the GPU from decode through normalization. [TorchVision adapter](../src/augbench/implementations/torchvision_pipeline_impl.py) · [TorchCodec JPEG decoding](https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/image_decoding.html)

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

Use the native AlbumentationsX recipe as the reference. Match its geometry, interpolation, boundary fill, probability of applying the operation, and parameter distribution in the other adapters. For example, the current Affine recipe specifies the same scale, rotation, shear, and translation so every path performs the full warp. Library-native compositions are useful when an API lacks a single operator for the requested effect. Shared seeds align the input order. Each library draws its own parameters from the declared distribution.

Pixel rounding and some native algorithms still differ. The contract calls out consequential differences. For example, JPEG quality 50 is the same nominal setting for a codec and a JPEG artifact simulator, but it does not guarantee the same distortion. If a path changes what is measured, its comparison needs new observations under the corrected recipe. [Recipe catalog](../catalog/recipes/rgb.yaml) · [Aggregation rules](benchmark_execution_contract.md#aggregation)

### How do we match Rain and ThinPlateSpline to AlbumentationsX?

**Rain (R41).** On a 224×224 crop, AlbumentationsX draws 74 streaks with a 28-pixel vertical displacement and a nominal 5-pixel line thickness. It blurs the rain layer with a 7×7 box filter, then adds it to the image. Kornia's default draws 1,000–2,000 thin lines and writes their color directly into the image. We set Kornia's density, height, and slant to match the AX recipe, then thicken, blur, and add its rain layer. In a CPU check over 24 black crops, AX and Kornia each changed about 54% of pixels and raised the mean level by about 57 and 56 points on a 0–255 scale. [AX adapter](../src/augbench/implementations/albumentationsx_impl.py) · [Kornia adapter](../src/augbench/implementations/kornia_impl.py)

**ThinPlateSpline (R47).** AX draws Gaussian offsets at four corners and the center, with standard deviation equal to 5% of the crop dimension, or 11.2 px on a 224×224 crop. Kornia now draws from that same distribution and fits its native spline from the fixed output anchors to the perturbed input coordinates. Both adapters draw inside the transform. In a CPU check with 24 sets of identical anchor offsets, their dense sampling maps agreed to less than 0.001 px RMS. Interpolation and rounding can still differ. [Recipe catalog](../catalog/recipes/rgb.yaml) · [Kornia adapter](../src/augbench/implementations/kornia_impl.py)

## Why normalize on the GPU and return FP16?

Normalizing on the GPU gives every path the same final device and output contract. A path that augments `uint8` images on the CPU can transfer one byte per channel value, then convert and normalize on CUDA. The cost of making the normalized FP16 batch ready for the model stays in the measurement. Intermediate dtypes and arithmetic follow each implementation's documented path. [Execution paths](benchmark_execution_contract.md#timing-boundary)

## Why randomize image order after reading the files into RAM?

Random access to a cold disk adds seek latency and random I/O throughput to the measurement. Those can change the result even when the image libraries do the same amount of work. Before timing, the runner reads the selected 10,000 JPEG files sequentially once. Their combined size is less than the 6.28 GiB ImageNet validation archive, and the VM has 64 GiB of host memory, so the Linux page cache can hold the selection. Each timed seed still visits files in a seeded random order without replacement. Timed file opens and page cache lookups remain, along with JPEG decoding and augmentation. The result describes a warm cache-resident workload. It does not describe cold-cache or remote-storage performance. [Dataset selection](../configs/families/rgb.yaml) · [Data access contract](benchmark_execution_contract.md#data-access)

## Why hold the machine, batch size, workers, and prefetch settings fixed?

All paths use the same file order for a given seed, a batch size of 256, and the same L4 machine type. The timer starts after one warm-up batch to keep pipeline setup out of the throughput number. We also fix the worker and prefetch settings for the DataLoader paths. DALI uses its native pipeline, so a prefetch depth of two there and a prefetch factor of two per DataLoader worker describe different queues. Fixing the settings gives a repeatable comparison at one operating point. Tuning the worker count, batch size, or accelerator for a particular deployment can change the ranking. [Fixed workload](benchmark_execution_contract.md#fixed-rgb-workload)

## Why measure throughput and GPU memory together?

Some pipelines use more VRAM to deliver batches faster. Measuring both in one pass ties the two numbers to the same decoder, settings, and batch stream. The process-level memory reading covers CUDA allocations made by the decoder and libraries as well as tensors tracked by PyTorch. The export includes both the baseline before pipeline construction and the observed peak. Polling can miss a short-lived spike. [Measurements per cell](benchmark_execution_contract.md#what-one-cell-measures)

## Why compare only shared recipes and report API coverage separately?

A speed ratio is meaningful when both paths execute the same workload. Summary charts use the recipes shared by the paths shown, and the notes describe where those implementations still differ. Coverage answers a second question with its own denominator. The census starts from selected AlbumentationsX operations and records counterparts in the other libraries, including DALI. It distinguishes direct operators, compositions, and approximate matches. This shows which effects have a counterpart even when they lack a timing row. [API census](../catalog/operations.yaml) · [Aggregation rules](benchmark_execution_contract.md#aggregation)
