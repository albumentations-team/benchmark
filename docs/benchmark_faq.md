# Benchmark methodology FAQ

This FAQ explains the choices behind the RGB benchmark. The [execution contract](benchmark_execution_contract.md) specifies the current setup. The [result manifests](generated/results.json) identify the exact code and settings behind each published measurement, including older runs.

## Why measure from a JPEG on disk to a batch ready for the model?

A training loop needs ready batches. If the next batch arrives late, the model waits even when the augmentation kernel itself is fast. Libraries differ in where they decode, how they assemble batches, and when they transfer data to the GPU. The benchmark times those choices together, including file access, decoding, augmentation, batching, transfer, and normalization. For example, a pipeline that crops during JPEG decode can deliver a batch sooner even if its later transforms run at the same speed.

The output is a materialized CUDA FP16 batch in BCHW layout with three channels and 224×224 pixels. The runner synchronizes CUDA at the end of the timer so work already queued on the device counts. Training step time also depends on the model, its GPU use, and how training overlaps with input preparation. [Timing boundary](benchmark_execution_contract.md#timing-boundary)

## Why start with 8-bit RGB JPEGs from ImageNet validation?

JPEG input for RGB classification is a common workload with a large set of comparable operations across the libraries. ImageNet validation provides real files with varying dimensions and compression characteristics. A fixed selection from a checksum-verified archive gives each path the same bytes, and each seed gives all paths the same file order. We use the first 10,000 matching files in lexical order for reproducibility. This is one fixed ImageNet workload.

The 8-bit format describes the decoded source. A library can convert it to the dtype its operators expect, and every path delivers the same FP16 output. A workload that starts with FP32 or FP16 arrays changes which readers can participate, how many bytes move, and which kernels run. Nine-channel images, video, volumes, and sequences of volumes also need their own sample format, recipes, and batch size. Those comparisons have yet to be run. [RGB workload configuration](../configs/families/rgb.yaml) · [Requirements for a new workload](benchmark_execution_contract.md#adding-a-benchmark-family)

## Why use native readers and decoders?

The reader and decoder affect the time a user waits for a batch. Sharing one decoder would hide differences such as decoding on CUDA or decoding just the region a crop needs. The current adapters use each library's reader where available, including Kornia's Rust-backed reader and TorchVision's batched CUDA JPEG decoder. When a library provides image transforms without its own JPEG reader, the path specifies its decoder explicitly. AlbumentationsX uses SimpleJPEG.

We use native decode-and-crop when it can produce the requested crop. For example, the DALI adapter supplies one window per image for a fixed-size random crop and adds centered zero padding on short axes. For a random resized crop, it uses the decoder's random crop followed by resize. The decode optimization still has to preserve the recipe's crop size, sampling, and padding. [Execution paths](benchmark_execution_contract.md#timing-boundary) · [DALI ROI decoding](https://docs.nvidia.com/deeplearning/dali/archives/dali_2_4_0/user-guide/operations/nvidia.dali.fn.decoders.image_slice.html)

## Why apply TorchVision GPU transforms per image?

The recipes require an independent random draw for each image. A single call to a transform such as TorchVision v2 `RandomResizedCrop` on a batch tensor samples one set of crop parameters for that call. Images in the same batch would therefore share the crop. The adapter calls the transform separately for each image. This also lets it crop or resize variable-sized decoded images before stacking them. Two independent draws can still happen to select the same parameters.

Batching is appropriate when it preserves that independence. TorchVision can decode the input JPEGs in one CUDA call and normalize the assembled batch in one call. For JPEG recompression, the adapter draws an integer quality from 30 through 90 for each image, groups images that drew the same quality for encoding, and restores their original order after decoding. The CUDA encoder accepts one quality per call. [TorchVision v2 transform source](https://docs.pytorch.org/vision/stable/_modules/torchvision/transforms/v2/_transform.html) · [CUDA JPEG encoding](https://docs.pytorch.org/vision/0.29/generated/torchvision.io.encode_jpeg.html) · [Benchmark adapter](../src/augbench/implementations/torchvision_pipeline_impl.py)

## Why disable antialiasing for bilinear resize?

The resize and random-resized-crop recipes compare bilinear resampling. OpenCV exposes `INTER_LINEAR` without a separate antialiasing option. `INTER_AREA` is available for shrinking and uses a different resampling method. Filtering only on paths that enable it by default would change both the output and the amount of work being timed.

We request bilinear resampling without an antialiasing filter in these recipes. The Pillow paths use their native bilinear affine resampler because `Image.resize` with bilinear resampling filters when shrinking. Comparing antialiased resizing would require specifying a filter that all paths can match. [OpenCV resize modes](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) · [TorchVision resize options](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html) · [Resize details](benchmark_execution_contract.md#timing-boundary)

## How do we choose equivalent recipes across libraries?

Start with the desired image effect. Match the geometry, interpolation, boundary fill, probability of applying the operation, and parameter distribution wherever they affect the work. For example, the current Affine recipe specifies the same scale, rotation, shear, and translation so every path performs the full warp. Library-native compositions are useful when an API lacks a single operator for the requested effect. Shared seeds align the input order. Each library draws its own parameters from the declared distribution.

Pixel rounding and some native algorithms still differ. The contract calls out consequential differences. For example, JPEG quality 50 is the same nominal setting for a codec and a JPEG artifact simulator, but it does not guarantee the same distortion. If a path changes what is measured, its comparison needs new observations under the corrected recipe. [Recipe catalog](../catalog/recipes/rgb.yaml) · [Aggregation rules](benchmark_execution_contract.md#aggregation)

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
