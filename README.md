# Image and Video Augmentation Library Benchmarks

A comprehensive benchmarking suite for comparing the performance of popular image and video augmentation libraries including [AlbumentationsX](https://albumentations.ai/), [torchvision](https://docs.pytorch.org/vision/stable/index.html), and [Kornia](https://kornia.readthedocs.io/en/latest/).

## GitAds Sponsored
[![Sponsored by GitAds](https://gitads.dev/v1/ad-serve?source=albumentations-team/benchmark@github)](https://gitads.dev/v1/ad-track?source=albumentations-team/benchmark@github)

<details>
<summary>Table of Contents</summary>

- [Image and Video Augmentation Library Benchmarks](#image-and-video-augmentation-library-benchmarks)
  - [Overview](#overview)
  - [Benchmark Types](#benchmark-types)
    - [Image Benchmarks](#image-benchmarks)
    - [Video Benchmarks](#video-benchmarks)
  - [Benchmark Results](#benchmark-results)
    - [Result Tables](#result-tables)
    - [RGB](#rgb)
    - [9-Channel](#9-channel)
    - [Video](#video)
  - [Requirements](#requirements)
  - [Supported Libraries](#supported-libraries)
  - [Setup](#setup)
    - [Getting Started](#getting-started)
    - [Using Your Own Data](#using-your-own-data)
  - [Running Benchmarks](#running-benchmarks)
    - [Google Cloud (detached)](#google-cloud-detached)
    - [RGB Image Benchmarks](#rgb-image-benchmarks-all-libraries)
    - [Video Benchmarks](#video-benchmarks-all-libraries)
  - [Architecture](#architecture)
  - [Methodology](#methodology)
  - [Contributing](#contributing)

</details>

## Overview

This benchmark suite measures the throughput and performance characteristics of common augmentation operations across different libraries. It features:

- Benchmarks for both image and video augmentation
- Adaptive warmup to ensure stable measurements
- Multiple runs for statistical significance
- Detailed performance metrics and system information
- Thread control settings for consistent performance
- Support for multiple image/video formats and loading methods

## Benchmark Types

### Image Benchmarks

The image benchmarks compare the performance of various libraries on standard image transformations. Interpret the tables by benchmark mode:

- **Micro / profiler benchmarks** preload decoded images and time augmentation only. These runs use one internal CPU thread for every library to measure single-stream transform cost. For tensor-native image libraries (`torchvision`, `kornia`), `--device cuda|mps|auto` preloads tensors on the selected device and times device-resident augmentation.
- **DataLoader benchmarks** use recipe-level training pipelines. `memory_dataloader_augment` preloads decoded samples and isolates worker/augmentation scaling; `decode_dataloader_augment` adds disk read/decode; `decode_dataloader_augment_batch_copy` additionally materializes the collated batch tensor and copies it to CUDA/MPS when requested. CPU image pipelines apply the full recipe inside the dataset path before collation. TorchVision and Kornia image GPU DataLoader rows split the recipe: workers use the same library on CPU for crop/pad shape preparation, then the collated batch is copied to GPU. Kornia runs the measured augmentation batched with `same_on_batch=False` plus normalization; TorchVision runs only the measured augmentation in a per-sample GPU loop to preserve per-image randomness, then applies normalization once to the whole batch. Pipeline recipes include `Normalize+ToTensor` in the library spec: AlbumentationsX uses `ToTensorV2`, Pillow uses `torchvision.transforms.PILToTensor` before normalization, and torchvision/Kornia already operate on tensors. All pipeline recipes return fixed-shape tensor outputs that PyTorch default collation can stack. These runs record worker counts, thread policy, device target, and whether decode/collate/device transfer were included.

The checked-in result tables use `2,000` ImageNet validation images for image micro benchmarks and `10,000` images for
image DataLoader benchmarks. Full `50,000`-image ImageNet sweeps are optional when validating a specific production
deployment.

### Video Benchmarks

Video benchmarks use fixed-length clips from UCF101. AlbumentationsX receives clips as NumPy arrays with shape
`(T, H, W, C)` and applies transforms through `transform(images=video)["images"]`, so parameters are sampled once per
clip and shared across frames. This matches the training-style semantics used by Kornia's `same_on_batch=True` path.

## Benchmark Results

<!-- PAPER_FIGURES_START -->

The figures and tables below are generated from checked-in benchmark data.

### Figure 1. Open production DataLoader category

![Figure 1. Open production DataLoader category](docs/paper_figures/open_dataloader_leaderboard.png)

CPU and GPU DataLoader implementations compete together over the same 57-recipe universe. Bars show median measured-row throughput; labels show full measured coverage and open-category wins. AlbumentationsX CPU wins 52 of 57 recipes and has the highest median throughput.

### Figure 2. Coverage breadth versus measured throughput

![Figure 2. Coverage breadth versus measured throughput](docs/paper_figures/coverage_vs_throughput.png)

DataLoader coverage and throughput are distinct benchmark axes. The x-axis is the count of full measured recipes over the canonical 57 CPU DataLoader recipes, and the y-axis is median throughput over measured rows only. The Elastic drill-down shows that GPU execution does not rescue a slow implementation of a hard transform.

### Figure 3. GPU DataLoader pipelines versus AlbumentationsX CPU

![Figure 3. GPU DataLoader pipelines versus AlbumentationsX CPU](docs/paper_figures/gpu_vs_albumentationsx_cpu_ratios.png)

Each point is a paired GPU DataLoader recipe divided by the AlbumentationsX CPU DataLoader throughput for the same recipe. The dashed line marks parity. Most GPU rows fall below parity once the full DataLoader path is measured.

### Figure 4. GPU memory consumed by augmentation pipelines

![Figure 4. GPU memory consumed by augmentation pipelines](docs/paper_figures/gpu_memory_vs_throughput.png)

GPU augmentation also consumes accelerator memory that would otherwise be available to model parameters, activations, optimizer state, or larger batches. Each point is a measured GPU DataLoader row with peak allocated memory recorded during the benchmark.

### Figure 5. Winner counts by benchmark regime

![Figure 5. Winner counts by benchmark regime](docs/paper_figures/winner_counts.png)

Measured winner counts among comparable measured transforms by regime. The conclusion changes when moving from augmentation-only microbenchmarks to production-style DataLoader measurements.

### Result Tables

The tables below summarize the checked-in benchmark results for RGB images, 9-channel images, and video clips. Image table values are medians with 95% confidence intervals when available; the video fallback table reports its own uncertainty in the column headers. Image tables report throughput in images/s; the video table reports clips/s. A dash means no full measured row is available.

### RGB

| Transform | AlbumentationsX<br>CPU micro | AlbumentationsX<br>CPU DataLoader | TorchVision<br>GPU micro | DALI<br>GPU DataLoader |
| --- | ---: | ---: | ---: | ---: |
| Affine | 871.8 ± 9.1 | **4527.7 ± 31.8** | 1316.9 ± 76.9 | 3806.0 ± 90.1 |
| AutoContrast | 1242.7 ± 21.7 | **4645.6 ± 90.1** | 3942.2 ± 531.8 | - |
| Blur | 4448.7 ± 19.5 | **5274.9 ± 195.9** | - | - |
| Brightness | **6912.2 ± 14.4** | 5230.9 ± 274.5 | 5706.8 ± 956.3 | 3797.8 ± 38.0 |
| CLAHE | 282.9 ± 1.3 | 3384.5 ± 192.0 | - | **3730.6 ± 158.7** |
| ChannelDropout | **6810.1 ± 73.5** | 5316.5 ± 101.4 | - | - |
| ChannelShuffle | 4337.4 ± 14.5 | 5086.6 ± 250.8 | **9557.4 ± 2514.8** | - |
| ColorJiggle | 639.3 ± 5.5 | **4255.1 ± 68.1** | 680.0 ± 18.7 | 3742.5 ± 38.5 |
| ColorJitter | 641.4 ± 0.8 | **4224.7 ± 261.0** | 687.0 ± 23.4 | 3817.8 ± 113.7 |
| Contrast | **6932.8 ± 34.1** | 5258.2 ± 162.6 | 3274.0 ± 360.2 | 3759.3 ± 83.0 |
| CornerIllumination | 424.6 ± 2.7 | **3824.1 ± 95.3** | - | - |
| Elastic | 191.0 ± 0.4 | **2954.0 ± 65.9** | - | - |
| EnhanceDetail | 2148.3 ± 14.5 | **5033.1 ± 63.7** | - | - |
| EnhanceEdge | 1373.3 ± 17.6 | **4923.4 ± 112.2** | - | - |
| Equalize | 807.4 ± 3.2 | **4304.4 ± 139.7** | 2015.9 ± 149.1 | 3823.9 ± 56.3 |
| Erasing | **9510.6 ± 83.9** | 5118.2 ± 296.1 | 2242.0 ± 183.9 | 3791.4 ± 50.9 |
| GaussianBlur | 2342.8 ± 4.3 | **5029.0 ± 106.1** | 2803.2 ± 279.1 | 3700.9 ± 94.7 |
| GaussianIllumination | 388.1 ± 1.4 | **3655.9 ± 47.9** | - | - |
| GaussianNoise | 225.1 ± 0.5 | 3321.3 ± 68.1 | - | **3820.0 ± 67.6** |
| Grayscale | 5193.9 ± 1.5 | 5263.2 ± 89.8 | **8863.8 ± 2122.0** | - |
| HorizontalFlip | 8416.0 ± 21.4 | 5218.1 ± 163.2 | **16083.8 ± 5064.7** | 3722.4 ± 86.3 |
| Hue | 966.9 ± 0.9 | **4698.0 ± 54.0** | - | 3784.2 ± 6.0 |
| Invert | 15094.9 ± 69.6 | 5491.9 ± 106.5 | **15936.1 ± 5092.4** | - |
| JpegCompression | 692.0 ± 7.5 | **4106.2 ± 170.9** | - | 3786.2 ± 26.7 |
| LinearIllumination | 520.7 ± 1.2 | **4076.3 ± 91.2** | - | - |
| LongestMaxSize | **2824.5 ± 47.9** | 1316.9 ± 32.7 | - | - |
| MedianBlur | 843.3 ± 4.2 | **4038.1 ± 113.5** | - | - |
| MotionBlur | 1952.5 ± 23.3 | **4614.8 ± 136.8** | - | - |
| OpticalDistortion | 274.4 ± 1.2 | **3556.4 ± 70.4** | - | - |
| Pad | 13181.0 ± 134.1 | 4866.6 ± 92.5 | **16609.6 ± 5327.4** | 3756.1 ± 88.2 |
| Perspective | 559.4 ± 2.0 | **3992.0 ± 131.9** | 760.6 ± 22.8 | - |
| PhotoMetricDistort | 580.9 ± 5.1 | **4149.0 ± 105.1** | 619.4 ± 15.7 | - |
| PlankianJitter | 2253.1 ± 19.2 | **4899.4 ± 85.3** | - | - |
| PlasmaBrightness | 267.0 ± 0.8 | **2672.0 ± 64.8** | - | - |
| PlasmaContrast | 142.8 ± 0.5 | **2155.6 ± 96.2** | - | - |
| PlasmaShadow | 419.8 ± 2.9 | **2795.5 ± 44.2** | - | - |
| Posterize | 14398.5 ± 65.3 | 5319.0 ± 73.7 | **15122.0 ± 4628.1** | - |
| RGBShift | 2292.1 ± 3.2 | **4830.7 ± 131.1** | - | - |
| Rain | 1258.8 ± 2.4 | **4528.5 ± 225.5** | - | - |
| RandomCrop224 | **38380.3 ± 217.1** | 5084.5 ± 122.8 | 15008.3 ± 4670.8 | 3589.1 ± 82.9 |
| RandomGamma | **9937.7 ± 51.6** | 5251.3 ± 153.4 | - | - |
| RandomJigsaw | **5172.0 ± 17.8** | 4868.3 ± 31.1 | - | - |
| RandomResizedCrop | **7150.4 ± 21.0** | 5056.2 ± 143.9 | 3886.9 ± 481.3 | 3898.1 ± 116.7 |
| RandomRotate90 | **5990.0 ± 95.9** | 5086.5 ± 48.6 | - | - |
| Resize | 2462.7 ± 41.7 | 1333.8 ± 15.7 | **6472.7 ± 1160.8** | 3523.1 ± 69.2 |
| Rotate | 1407.6 ± 45.8 | **4782.3 ± 137.3** | 1363.9 ± 59.6 | 3808.5 ± 71.7 |
| SaltAndPepper | 737.7 ± 11.3 | **4459.7 ± 45.9** | - | 3824.4 ± 85.8 |
| Saturation | 846.6 ± 19.1 | **4581.7 ± 110.3** | - | 3792.5 ± 33.6 |
| Sharpen | 1387.6 ± 5.2 | **4821.7 ± 141.8** | 3332.5 ± 389.0 | - |
| Shear | 784.4 ± 6.3 | **4261.2 ± 88.1** | - | 3771.7 ± 61.0 |
| SmallestMaxSize | **2017.5 ± 27.8** | 1328.7 ± 42.2 | - | - |
| Snow | 489.3 ± 3.3 | **4135.0 ± 171.2** | - | - |
| Solarize | 9759.5 ± 38.8 | 5338.8 ± 71.7 | **10111.6 ± 2582.8** | - |
| ThinPlateSpline | 51.7 ± 0.1 | **721.0 ± 66.3** | - | - |
| Transpose | 4626.8 ± 29.6 | **5230.6 ± 130.3** | - | - |
| UnsharpMask | 906.1 ± 2.2 | **4521.6 ± 78.0** | - | - |
| VerticalFlip | 14051.5 ± 61.9 | 5301.7 ± 165.5 | **16368.3 ± 5461.5** | 3766.2 ± 133.8 |

### 9-Channel

| Transform | AlbumentationsX<br>9ch CPU micro | AlbumentationsX<br>9ch CPU DataLoader | TorchVision<br>9ch GPU micro | TorchVision<br>9ch GPU DataLoader |
| --- | ---: | ---: | ---: | ---: |
| Affine | 229.8 ± 0.0 | **1617.5 ± 0.0** | 1187.5 ± 0.0 | 1085.2 ± 0.0 |
| AutoContrast | 316.6 ± 0.0 | **1749.3 ± 0.0** | 1329.1 ± 0.0 | 1024.6 ± 0.0 |
| Blur | 1385.1 ± 0.0 | **1998.8 ± 0.0** | - | - |
| Brightness | **2477.1 ± 0.0** | 2041.0 ± 0.0 | 1952.7 ± 0.0 | 1271.8 ± 0.0 |
| ChannelDropout | **3335.5 ± 0.0** | 2027.3 ± 0.0 | - | - |
| ChannelShuffle | 1447.9 ± 0.0 | 1946.7 ± 0.0 | **10344.1 ± 0.0** | 1341.8 ± 0.0 |
| Contrast | **2482.3 ± 0.0** | 2113.2 ± 0.0 | 1162.8 ± 0.0 | 920.2 ± 0.0 |
| CornerIllumination | 195.8 ± 0.0 | **1627.4 ± 0.0** | - | - |
| Elastic | 121.0 ± 0.0 | **1404.3 ± 0.0** | - | 99.3 ± 0.0 |
| Erasing | **3658.7 ± 0.0** | 2024.1 ± 0.0 | 1438.0 ± 0.0 | 1323.0 ± 0.0 |
| GaussianBlur | 747.5 ± 0.0 | 1952.9 ± 0.0 | **3044.7 ± 0.0** | 1259.0 ± 0.0 |
| GaussianIllumination | 189.4 ± 0.0 | **1583.1 ± 0.0** | - | - |
| GaussianNoise | 75.8 ± 0.0 | **1249.8 ± 0.0** | - | - |
| Grayscale | 177.7 ± 0.0 | 1559.5 ± 0.0 | **3042.9 ± 0.0** | 1275.9 ± 0.0 |
| HorizontalFlip | 837.3 ± 0.0 | 1801.2 ± 0.0 | **20436.0 ± 0.0** | 1482.7 ± 0.0 |
| Invert | 4622.5 ± 0.0 | 2026.3 ± 0.0 | **24578.9 ± 0.0** | 1524.5 ± 0.0 |
| JpegCompression | 103.5 ± 0.0 | **1287.1 ± 0.0** | - | - |
| LinearIllumination | 163.1 ± 0.0 | **1585.7 ± 0.0** | - | - |
| LongestMaxSize | **612.5 ± 0.0** | 469.9 ± 0.0 | - | - |
| MedianBlur | 290.1 ± 0.0 | **1542.1 ± 0.0** | - | - |
| MotionBlur | 776.7 ± 0.0 | **1854.2 ± 0.0** | - | - |
| OpticalDistortion | 140.0 ± 0.0 | **1491.5 ± 0.0** | - | - |
| Pad | 4373.1 ± 0.0 | 1797.5 ± 0.0 | **17954.9 ± 0.0** | 1443.3 ± 0.0 |
| Perspective | 208.6 ± 0.0 | **1580.8 ± 0.0** | 718.0 ± 0.0 | 759.7 ± 0.0 |
| PlasmaBrightness | 114.3 ± 0.0 | **1308.6 ± 0.0** | - | - |
| PlasmaContrast | 46.0 ± 0.0 | **873.9 ± 0.0** | - | - |
| PlasmaShadow | 235.5 ± 0.0 | **1391.6 ± 0.0** | - | - |
| Posterize | 4533.0 ± 0.0 | 2012.5 ± 0.0 | **20756.5 ± 0.0** | 1554.5 ± 0.0 |
| RandomCrop224 | **18067.7 ± 0.0** | 2004.5 ± 0.0 | 15069.7 ± 0.0 | 1577.2 ± 0.0 |
| RandomGamma | **3439.2 ± 0.0** | 2003.2 ± 0.0 | - | - |
| RandomJigsaw | **2852.1 ± 0.0** | 1952.4 ± 0.0 | - | - |
| RandomResizedCrop | 1870.7 ± 0.0 | 1782.6 ± 0.0 | **4337.0 ± 0.0** | 628.2 ± 0.0 |
| RandomRotate90 | 687.7 ± 0.0 | **1862.5 ± 0.0** | - | - |
| Resize | 543.3 ± 0.0 | 468.6 ± 0.0 | **4727.3 ± 0.0** | 1394.3 ± 0.0 |
| Rotate | 645.4 ± 0.0 | **1883.7 ± 0.0** | 1253.4 ± 0.0 | 1115.8 ± 0.0 |
| Sharpen | 479.0 ± 0.0 | **1831.2 ± 0.0** | 1204.8 ± 0.0 | 905.9 ± 0.0 |
| Shear | 181.0 ± 0.0 | **1576.7 ± 0.0** | - | - |
| SmallestMaxSize | 435.4 ± 0.0 | **467.1 ± 0.0** | - | - |
| Solarize | 3364.3 ± 0.0 | 2082.8 ± 0.0 | **12677.5 ± 0.0** | 1527.0 ± 0.0 |
| ThinPlateSpline | 44.4 ± 0.0 | **460.9 ± 0.0** | - | - |
| VerticalFlip | 4444.1 ± 0.0 | 2021.4 ± 0.0 | **23657.7 ± 0.0** | 1560.0 ± 0.0 |

### Video

| Transform                | AlbumentationsX (video) 2.1.1 [vid/s]   | kornia (video) 0.8.0 [vid/s]   | torchvision (video) 0.21.0 [vid/s]   | Speedup (albx / fastest, +/-1sd)   |
|:-------------------------|:----------------------------------------|:-------------------------------|:-------------------------------------|:-----------------------------------|
| AdditiveNoise            | **10 ± 0**                              | -                              | -                                    | N/A                                |
| AdvancedBlur             | **24 ± 1**                              | -                              | -                                    | N/A                                |
| Affine                   | 25 ± 0                                  | 21 ± 0                         | **453 ± 0**                          | 0.06x (0.06-0.06x)                 |
| AtmosphericFog           | **6 ± 0**                               | -                              | -                                    | N/A                                |
| AutoContrast             | 22 ± 0                                  | 21 ± 0                         | **578 ± 17**                         | 0.04x (0.04-0.04x)                 |
| Blur                     | **110 ± 1**                             | 21 ± 0                         | -                                    | 5.33x (5.29-5.37x)                 |
| Brightness               | 241 ± 2                                 | 22 ± 0                         | **756 ± 435**                        | 0.32x (0.20-0.76x)                 |
| CLAHE                    | **10 ± 0**                              | -                              | -                                    | N/A                                |
| CenterCrop128            | 975 ± 13                                | 70 ± 1                         | **1133 ± 235**                       | 0.86x (0.70-1.10x)                 |
| ChannelDropout           | **205 ± 1**                             | 22 ± 0                         | -                                    | 9.42x (9.37-9.47x)                 |
| ChannelShuffle           | 26 ± 0                                  | 20 ± 0                         | **958 ± 0**                          | 0.03x (0.03-0.03x)                 |
| ChannelSwap              | **24 ± 0**                              | -                              | -                                    | N/A                                |
| ChromaticAberration      | **9 ± 0**                               | -                              | -                                    | N/A                                |
| CoarseDropout            | **487 ± 6**                             | -                              | -                                    | N/A                                |
| ColorJitter              | 19 ± 1                                  | 19 ± 0                         | **69 ± 0**                           | 0.27x (0.26-0.29x)                 |
| ConstrainedCoarseDropout | **112591 ± 2961**                       | -                              | -                                    | N/A                                |
| Contrast                 | 239 ± 2                                 | 22 ± 0                         | **547 ± 13**                         | 0.44x (0.42-0.45x)                 |
| CornerIllumination       | **10 ± 0**                              | 3 ± 0                          | -                                    | 3.96x (3.79-4.13x)                 |
| CropAndPad               | **42 ± 2**                              | -                              | -                                    | N/A                                |
| Defocus                  | **2 ± 0**                               | -                              | -                                    | N/A                                |
| Dithering                | slow-skipped                            | -                              | -                                    | N/A                                |
| Downscale                | **83 ± 1**                              | -                              | -                                    | N/A                                |
| Elastic                  | 26 ± 0                                  | -                              | **127 ± 1**                          | 0.21x (0.20-0.21x)                 |
| Emboss                   | **47 ± 1**                              | -                              | -                                    | N/A                                |
| Equalize                 | 16 ± 0                                  | 4 ± 0                          | **192 ± 1**                          | 0.08x (0.08-0.08x)                 |
| Erasing                  | **458 ± 7**                             | -                              | 255 ± 7                              | 1.80x (1.73-1.88x)                 |
| FancyPCA                 | **2 ± 0**                               | -                              | -                                    | N/A                                |
| FilmGrain                | **5 ± 0**                               | -                              | -                                    | N/A                                |
| GaussianBlur             | 42 ± 1                                  | 22 ± 0                         | **543 ± 11**                         | 0.08x (0.07-0.08x)                 |
| GaussianIllumination     | 10 ± 0                                  | **20 ± 0**                     | -                                    | 0.50x (0.49-0.51x)                 |
| GaussianNoise            | 11 ± 0                                  | **22 ± 0**                     | -                                    | 0.51x (0.49-0.53x)                 |
| GlassBlur                | **1 ± 0**                               | -                              | -                                    | N/A                                |
| Grayscale                | 82 ± 0                                  | 22 ± 0                         | **838 ± 467**                        | 0.10x (0.06-0.22x)                 |
| GridDistortion           | **28 ± 0**                              | -                              | -                                    | N/A                                |
| GridDropout              | **93 ± 14**                             | -                              | -                                    | N/A                                |
| GridMask                 | **199 ± 3**                             | -                              | -                                    | N/A                                |
| HSV                      | **15 ± 1**                              | -                              | -                                    | N/A                                |
| Halftone                 | slow-skipped                            | -                              | -                                    | N/A                                |
| HorizontalFlip           | 30 ± 0                                  | 22 ± 0                         | **978 ± 49**                         | 0.03x (0.03-0.03x)                 |
| Hue                      | **26 ± 2**                              | 20 ± 0                         | -                                    | 1.33x (1.22-1.45x)                 |
| ISONoise                 | **9 ± 0**                               | -                              | -                                    | N/A                                |
| Invert                   | 467 ± 27                                | 22 ± 0                         | **843 ± 176**                        | 0.55x (0.43-0.74x)                 |
| JpegCompression          | **25 ± 0**                              | -                              | -                                    | N/A                                |
| LensFlare                | **7 ± 0**                               | -                              | -                                    | N/A                                |
| LinearIllumination       | **10 ± 0**                              | 4 ± 0                          | -                                    | 2.39x (2.25-2.54x)                 |
| LongestMaxSize           | **28 ± 0**                              | -                              | -                                    | N/A                                |
| MedianBlur               | **24 ± 0**                              | 8 ± 0                          | -                                    | 2.85x (2.79-2.91x)                 |
| Morphological            | **219 ± 2**                             | -                              | -                                    | N/A                                |
| MotionBlur               | **80 ± 2**                              | -                              | -                                    | N/A                                |
| MultiplicativeNoise      | **40 ± 0**                              | -                              | -                                    | N/A                                |
| Normalize                | 22 ± 0                                  | 22 ± 0                         | **461 ± 0**                          | 0.05x (0.05-0.05x)                 |
| OpticalDistortion        | **26 ± 0**                              | -                              | -                                    | N/A                                |
| Pad                      | 302 ± 11                                | -                              | **760 ± 338**                        | 0.40x (0.27-0.74x)                 |
| PadIfNeeded              | **17 ± 0**                              | -                              | -                                    | N/A                                |
| Perspective              | 22 ± 0                                  | -                              | **435 ± 0**                          | 0.05x (0.05-0.05x)                 |
| PhotoMetricDistort       | **16 ± 1**                              | -                              | -                                    | N/A                                |
| PiecewiseAffine          | **25 ± 0**                              | -                              | -                                    | N/A                                |
| PixelDropout             | **76 ± 0**                              | -                              | -                                    | N/A                                |
| PlankianJitter           | **59 ± 0**                              | 11 ± 0                         | -                                    | 5.41x (5.37-5.46x)                 |
| PlasmaBrightness         | 4 ± 0                                   | **17 ± 0**                     | -                                    | 0.26x (0.25-0.27x)                 |
| PlasmaContrast           | 3 ± 0                                   | **17 ± 0**                     | -                                    | 0.17x (0.17-0.17x)                 |
| PlasmaShadow             | 7 ± 0                                   | **19 ± 0**                     | -                                    | 0.36x (0.35-0.37x)                 |
| Posterize                | 240 ± 8                                 | -                              | **631 ± 15**                         | 0.38x (0.36-0.40x)                 |
| RGBShift                 | 9 ± 0                                   | **22 ± 0**                     | -                                    | 0.42x (0.42-0.43x)                 |
| Rain                     | **27 ± 1**                              | 4 ± 0                          | -                                    | 7.24x (7.07-7.41x)                 |
| RandomCrop128            | 933 ± 7                                 | 65 ± 0                         | **1133 ± 15**                        | 0.82x (0.81-0.84x)                 |
| RandomFog                | slow-skipped                            | -                              | -                                    | N/A                                |
| RandomGamma              | **238 ± 1**                             | 22 ± 0                         | -                                    | 10.98x (10.93-11.03x)              |
| RandomGravel             | **24 ± 1**                              | -                              | -                                    | N/A                                |
| RandomGridShuffle        | **11 ± 0**                              | -                              | -                                    | N/A                                |
| RandomResizedCrop        | 28 ± 0                                  | 6 ± 0                          | **182 ± 16**                         | 0.15x (0.14-0.17x)                 |
| RandomRotate90           | **41 ± 4**                              | -                              | -                                    | N/A                                |
| RandomScale              | **56 ± 1**                              | -                              | -                                    | N/A                                |
| RandomShadow             | **8 ± 1**                               | -                              | -                                    | N/A                                |
| RandomSizedCrop          | **24 ± 0**                              | -                              | -                                    | N/A                                |
| RandomSunFlare           | **5 ± 0**                               | -                              | -                                    | N/A                                |
| RandomToneCurve          | **239 ± 1**                             | -                              | -                                    | N/A                                |
| Resize                   | 26 ± 0                                  | 6 ± 0                          | **140 ± 35**                         | 0.18x (0.14-0.25x)                 |
| RingingOvershoot         | **3 ± 0**                               | -                              | -                                    | N/A                                |
| Rotate                   | 49 ± 0                                  | 22 ± 0                         | **534 ± 0**                          | 0.09x (0.09-0.09x)                 |
| SafeRotate               | **24 ± 0**                              | -                              | -                                    | N/A                                |
| SaltAndPepper            | **12 ± 0**                              | 9 ± 0                          | -                                    | 1.36x (1.34-1.38x)                 |
| Saturation               | 19 ± 1                                  | **37 ± 0**                     | -                                    | 0.52x (0.50-0.54x)                 |
| Sharpen                  | 38 ± 0                                  | 18 ± 0                         | **420 ± 9**                          | 0.09x (0.09-0.09x)                 |
| Shear                    | **23 ± 0**                              | -                              | -                                    | N/A                                |
| ShiftScaleRotate         | **24 ± 0**                              | -                              | -                                    | N/A                                |
| ShotNoise                | **1 ± 0**                               | -                              | -                                    | N/A                                |
| SmallestMaxSize          | **18 ± 0**                              | -                              | -                                    | N/A                                |
| Snow                     | **13 ± 0**                              | -                              | -                                    | N/A                                |
| Solarize                 | 249 ± 9                                 | 21 ± 0                         | **628 ± 6**                          | 0.40x (0.38-0.41x)                 |
| Spatter                  | **7 ± 0**                               | -                              | -                                    | N/A                                |
| SquareSymmetry           | **37 ± 3**                              | -                              | -                                    | N/A                                |
| Superpixels              | slow-skipped                            | -                              | -                                    | N/A                                |
| ThinPlateSpline          | 23 ± 0                                  | **45 ± 1**                     | -                                    | 0.51x (0.49-0.53x)                 |
| ToSepia                  | **135 ± 0**                             | -                              | -                                    | N/A                                |
| Transpose                | **28 ± 0**                              | -                              | -                                    | N/A                                |
| UnsharpMask              | **8 ± 0**                               | -                              | -                                    | N/A                                |
| VerticalFlip             | 591 ± 20                                | 22 ± 0                         | **978 ± 5**                          | 0.60x (0.58-0.63x)                 |
| Vignetting               | **10 ± 1**                              | -                              | -                                    | N/A                                |
| WaterRefraction          | **22 ± 0**                              | -                              | -                                    | N/A                                |
| ZoomBlur                 | **4 ± 0**                               | -                              | -                                    | N/A                                |

<!-- PAPER_FIGURES_END -->

## Requirements

The benchmark automatically creates isolated virtual environments for each library and installs the necessary dependencies. Base requirements:

- Python 3.10+
- uv (for fast package installation)
- Disk space for virtual environments
- Image/video dataset in a supported format

## Supported Libraries

- [AlbumentationsX](https://albumentations.ai/) (commercial/AGPL)
- [torchvision](https://docs.pytorch.org/vision/stable/index.html)
- [Kornia](https://kornia.readthedocs.io/en/latest/)

Each library's specific dependencies are managed through separate requirements files in the `requirements/` directory.

## Setup

### Getting Started

For testing and comparison purposes, you can use standard datasets:

**For image benchmarks:**
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mkdir -p imagenet/val
tar -xf ILSVRC2012_img_val.tar -C imagenet/val
```

This is the same ImageNet validation input convention used by `imread_benchmark`: download the official validation tar, unpack it locally, then point `--data-dir` at `imagenet/val`.

**For video benchmarks:**
```bash
# UCF101 dataset
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar -d /path/to/your/target/directory
```

For cloud runs, package datasets as a single tarball and upload that object to GCS. This is much faster and more reliable
than copying thousands of small files from your laptop to GCS and then from GCS to the VM.

```bash
# ImageNet validation directory -> tarball.
COPYFILE_DISABLE=1 tar --no-xattrs \
  --exclude="__MACOSX" \
  --exclude="*/__MACOSX/*" \
  --exclude=".DS_Store" \
  --exclude="*/.DS_Store" \
  --exclude="._*" \
  --exclude="*/._*" \
  -cf /tmp/imagenet-val.tar \
  -C /path/to/imagenet val

gcloud storage cp /tmp/imagenet-val.tar gs://my-bucket/datasets/imagenet/val.tar

# UCF101 directory -> tarball.
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
gcloud storage objects describe gs://imagenet_validation/ucf101/ucf101.tar \
  --format="yaml(size,crc32c,md5Hash,updated)"

# Optional sanity check: this should print nothing.
tar -tf /tmp/ucf101.tar | rg '(^__MACOSX/|/\.DS_Store$|^\.DS_Store$|/\._|^\._)'
```

The video cloud benchmark runs use `gs://imagenet_validation/ucf101/ucf101.tar`; the uploaded object was verified at
`14136559616` bytes.

### Using Your Own Data

We strongly recommend running the benchmarks on your own dataset that matches your use case:

- Use images/videos that are representative of your actual workload
- Consider sizes and formats you typically work with
- Include edge cases specific to your application

This will give you more relevant performance metrics for your specific use case.

## Running Benchmarks

All benchmarks use the unified CLI: `python -m benchmark.cli run`. Prefer checked-in YAML configs for benchmark and cloud
runs; CLI flags are override knobs for an existing config, not a second source of truth. Config files are validated with
Pydantic before work starts.
Named transform sets are expanded to concrete transform names, and the resolved config is written to
`resolved_config.yaml` in the output directory.

```bash
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml
python -m benchmark.cli plan --config configs/examples/local_rgb_dataloader_cpu.yaml
python -m benchmark.cli run --config configs/examples/local_rgb_dataloader_cpu.yaml --num-items 25
```

Use `benchmark plan --config ...` or `benchmark run --config ... --dry-run` to print the resolved config, generated jobs,
expected output files, and cloud VM settings without starting local measurements or creating a VM.

Flag-only benchmark execution is intentionally unsupported. Start from a checked-in YAML config, then use supported
overrides such as `--num-items`, `--num-runs`, `--device`, `--workers`,
`--batch-size`, and `--output` when you need quick local changes.

The CLI creates joined virtual environments for compatible libraries, for example `.venv_albumentationsx` for AlbumentationsX and `.venv_torch_stack` for torchvision, Kornia, and Pillow image benchmarks. By default, each run refreshes `requirements/*.txt` from `requirements/*.in` with the latest compatible package versions, then installs dependencies only when the resolved requirement files changed. Pass `--no-refresh-requirements` for offline/debug reruns that should reuse the existing lock files and venv cache.

For production image runs, prefer the checked-in `prod_*` configs. The first benchmark pass uses one run per row so the
full table can be covered quickly; top-up repeats can be merged later after coverage is validated.

Smoke configs remain available for path checks and fast reruns.

Pipeline result filenames include the key sweep parameters, for example
`albumentationsx_memory_dataloader_augment_n2000_r5_w8_b64_results.json` or
`torchvision_decode_dataloader_augment_batch_copy_nall_r5_w8_b64_dev-mps_results.json`.

Video DataLoader runs use dedicated recipe specs, not the transform-only video micro specs. For AlbumentationsX,
torchvision, and Kornia, the recipe shape is `crop + transform + Normalize + ToTensor` so DataLoader collation receives
fixed-shape tensor clips. This keeps video pipeline semantics aligned with RGB pipeline benchmarks while micro remains a
preloaded transform-only profiler.

Treat RGB micro results as an implementation profiler: preloaded decoded inputs, one process, one internal
library thread, augmentation only. They are useful for checking algorithmic implementation quality and regressions,
but they are intentionally artificial because they measure one CPU core instead of a production input pipeline.

The benchmark hardware set should focus on CPUs that resemble machines used to feed model training, not every available
cloud CPU family. For RGB micro/profiler runs, use a compact representative set:

- Apple Silicon laptop, e.g. MacBook M4, for local macOS Arm behavior.
- `c4-standard-16` for modern Intel x86.
- `c4d-standard-16` for modern AMD x86.
- `c4a-standard-16` for cloud Arm, if Arm portability is part of the claim.
- `g2-standard-16` for the host CPU used with L4 GPU training.
- `a2-highgpu-1g` for the host CPU used with A100 training.

Older/general-purpose machines such as `n2-standard-16` and `n2d-standard-16` are useful as historical baselines, but
they should not drive the headline benchmark claims. The more important benchmark rows are production-style DataLoader
runs for images, GPU image sanity checks for TorchVision/Kornia, and GPU video augmentation, especially torchvision video
paths on GPU.

Skip dependency lock refresh when you intentionally want the fastest local rerun from existing locks:

```bash
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml --no-refresh-requirements
```

### Benchmark execution policy

- The benchmark matrix lives in `benchmark/matrix.py`. Add scenario/library/mode support there first so spec files,
  requirement groups, transform sets, device support, pipeline scopes, and backend selection stay aligned.
- Shared image/video defaults live in `benchmark/policy.py`. Do not duplicate slow-skip thresholds, warmup item counts, or
  item labels separately in micro and pipeline runners.
- Command construction lives in `benchmark/jobs.py`, and backend dispatch lives in `benchmark/orchestrator.py`. The CLI
  should parse user intent and resolve scenarios, not grow backend-specific branches.
- Cloud runs stage one dataset tarball, such as `gs://.../val.tar` or `gs://.../ucf101.tar`, onto the VM and unpack it locally. Do not upload or copy thousands of individual images/videos for each run. Tarballs created on macOS should use `COPYFILE_DISABLE=1`, `--no-xattrs`, and excludes for `.DS_Store`, AppleDouble `._*`, and `__MACOSX`; the VM-side extractor also ignores those entries.
- Micro benchmarks preload the requested number of images or videos once per library into that library's native in-memory representation. Per-transform timing must not reread or decode media from disk.
- Micro benchmarks measure only the named transform in each library's native layout, then force the returned object into contiguous memory before timing stops. Do not add `Normalize`, `ToTensor`, axis conversion, or DataLoader collation work to micro specs.
- GPU image micro benchmarks are device-resident transform profilers for `torchvision` and `kornia`: samples and transforms are moved to CUDA/MPS before timing, and the timed loop synchronizes the selected device. They do not include host-to-device transfer.
- Kornia image GPU rows exclude `Shear` in micro and DataLoader modes because Kornia's current CUDA shear parameter
  generator can fail with mixed CPU/CUDA tensors when moved to GPU. Keep `Shear` in the image transform sets: it still
  runs for AlbumentationsX, Pillow, torchvision where supported, and Kornia CPU rows.
- Kornia 9-channel image GPU rows also exclude `MedianBlur`. On the L4 9-channel GPU micro run, Kornia's median-blur
  path requested a multi-GB temporary allocation after device-resident preload and OOMed. Keep `MedianBlur` in RGB GPU,
  CPU, and other-library rows; treat the exclusion as a Kornia 9-channel GPU memory limitation.
- Kornia RGB GPU DataLoader may record `GaussianIllumination` as unsupported because the current recipe path can hit a
  mixed CPU/CUDA tensor error. Keep this as a library/device limitation in the methodology rather than removing
  `GaussianIllumination` globally from CPU or other-library rows.
- Pyperf micro runs isolate transform measurements in subprocesses, but those subprocesses reuse the per-library media cache and lazily construct only the transform being measured.
- Libraries with lazy or partially lazy output objects must materialize their own result inside the timed call. Micro timing converts returned Pillow `Image.Image` objects to contiguous NumPy arrays and calls `.contiguous()` on tensor-like outputs so every measured transform produces realized contiguous output.
- Libraries should only be listed for direct per-transform rows when they support the named transform directly. Do not recreate missing transforms with extensive benchmark-side helper code just to fill a table cell. For example, Pillow can benchmark direct `Image` / `ImageOps` / `ImageFilter` operations, but should skip Albumentations-style composites such as `RandomResizedCrop`, `PadIfNeeded`, `SafeRotate`, `ShiftScaleRotate`, `LongestMaxSize`, and `SmallestMaxSize` in direct transform listings. Pipeline recipe benchmarks are the exception: they may include maintained Pillow equivalents for composite recipes when the goal is end-to-end pipeline comparison rather than claiming direct single-op support. When Pillow has a direct equivalent for an AlbumentationsX transform, keep the parameters exact.
- Compatible libraries share joined environments to avoid redundant dependency setup. Image benchmarks group torchvision, Kornia, and Pillow into the `torch_stack` environment; video benchmarks group torchvision and Kornia into `torch_video`.
- Environment setup is cached by resolved requirement files, Python version, media type, and environment group. Detached GCP runs can additionally reuse the GCS venv cache unless `--gcp-no-venv-cache` or `--gcp-force-venv-cache-rebuild` is set.
- Requirement lock refresh is expected once per library or joined-environment launch when refresh is enabled. Do not add extra cross-library refresh orchestration unless it removes real work without changing dependency freshness semantics; use `--no-refresh-requirements` for repeated local runs with fixed locks.
- Slow transforms are preflighted before exhaustive micro or DataLoader pipeline measurement. If an image transform is slower than the practical floor (`>=0.05 sec/image`, `<=20 img/s`), record an early-stop result instead of spending the full run budget. This prevents benchmark sweeps from getting stuck on transforms that are too slow for practical training use.
- Keep benchmark data local to the machine doing the timing. GCP runs should not benchmark against mounted buckets or network paths.
- Preserve single-thread micro timing for fair augmentation-only comparisons. Pipeline benchmarks use an explicit `--thread-policy`; the main production path is `pipeline-default`, and controlled comparison runs can use `pipeline-single-worker`.
- Pipeline specs, not `pipeline_runner.py`, own recipe-level tensor conversion. The runner should receive fixed-shape outputs and use PyTorch default collation; it should not repair channel layouts with benchmark-side heuristics.
- GPU image pipeline benchmarks are separate from CPU pipeline rows. For TorchVision and Kornia, `--device cuda|mps|auto` keeps decode/load and library-native crop/pad shape preparation in DataLoader workers on CPU, copies each fixed-shape collated batch to the selected device, applies the measured augmentation plus normalization on GPU, and includes synchronization in timing. Kornia uses batched augmentation with `same_on_batch=False`; TorchVision applies the measured augmentation in a per-sample GPU loop and then normalizes the whole batch because TorchVision v2 lacks a `same_on_batch=False` equivalent for batched transforms. AlbumentationsX and Pillow remain CPU-only for image benchmarks.
- TorchVision `JpegCompression` maps to `torchvision.transforms.v2.JPEG`, which requires `uint8` CPU input and is excluded from TorchVision GPU image rows. Keep it in CPU TorchVision rows and in other libraries that support it. Treat this as a JPEG-compression augmentation constraint when describing methodology.
- CUDA DataLoader rows record per-transform peak GPU memory during timed runs under `results.<transform>.gpu_memory`, including peak allocated/reserved bytes and before/after allocation snapshots. Pyperf micro rows do not report peak memory because their timed loops run inside pyperf worker processes.
- Benchmark code must be fair but fast: avoid repeated decode, loader construction, conversion, synchronization, checksums, materialization, or dependency work unless it is explicitly part of the named measurement scope or needed to make lazy work complete.

### Google Cloud (detached)

Run benchmarks on a **Compute Engine** VM that starts from your laptop, then keeps going after you disconnect. The default path is **detached**: the CLI uploads the repo and a typed job definition to **GCS**, creates a VM whose **startup script** downloads one dataset tarball such as `gs://.../val.tar` or `gs://.../ucf101.tar`, unpacks media files to **local disk** (benchmarks do not read from a mounted bucket), writes the typed run config to disk, runs `python -m benchmark.cli run --resolved-config /root/benchmark-work/job_config.yaml`, uploads **results**, **vm.log**, **exit_code.txt**, and **run_meta.json** under a unique prefix, and **deletes the VM** when finished (unless you set `cloud.keep_instance: true` or pass `--gcp-keep-instance` as an override).

The VM bootstrap stages the dataset before benchmark dependencies are installed. `benchmark/cloud/stage_dataset.py` must
therefore remain stdlib-only; Pydantic validation happens later inside the control venv and the per-library benchmark
venvs.

**Prerequisites**

- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud`) authenticated for your project.
- VM boot image must provide **Python 3.13+** (the package matches `requires-python` in `pytorch-latest-*` images only if that image already ships 3.13; otherwise use a custom image or install 3.13 in your startup flow—the bootstrap script fails fast with a clear error if `python3` is too old).
- A GCS bucket (or two) with:
  - A **dataset tarball** your VM can read, e.g. `gs://my-bucket/datasets/imagenet/val.tar` or `gs://my-bucket/datasets/ucf101/ucf101.tar`.
  - A **results base URI** where each run is written, e.g. `gs://my-bucket/benchmark-runs`.
- The default Compute Engine service account (or the one attached to the VM) needs **read** access to the dataset object and **read/write** to the results bucket. For the VM to **delete itself** after the run, that service account also needs permission to call **compute.instances.delete** on its own instance (e.g. `roles/compute.instanceAdmin.v1` on a dedicated benchmark project—tighten IAM for production).

**Submit a detached run**

Detached runs carry a typed `run_config` in `job.json`; the VM writes that config to disk and runs `benchmark.cli` with
`--resolved-config`. Point the real dataset at GCS in the YAML config:

```bash
python -m benchmark.cli plan --config configs/your_gcp_config.yaml
python -m benchmark.cli run --config configs/your_gcp_config.yaml --gcp-dry-run
python -m benchmark.cli run --config configs/your_gcp_config.yaml
```

After submission, open `./gcp_runs/gcp_last_run.json` for `run_prefix`, `instance_name`, and a suggested `gcloud storage cp` command to pull `results/` when the run finishes.

**Dry run (no upload, no VM)**

```bash
python -m benchmark.cli run --config configs/your_gcp_config.yaml --gcp-dry-run
```

If a GPU zone is stocked out, keep the config fixed and override only the zone that GCP suggests:

```bash
python -m benchmark.cli run --config configs/your_gcp_config.yaml --gcp-zone us-central1-a
```

**Attached / SSH mode (debug)**

Creates the VM, waits for SSH, uploads the repo, runs the benchmark in a live session, downloads results to `--output`, then deletes the VM. Requires a dataset path **on the VM** (you must stage data yourself):

```bash
python -m benchmark.cli run --config configs/your_gcp_config.yaml --gcp-attached --gcp-remote-data-dir /data/benchmark/videos
```

**Cost note:** GCS storage for a subset and JSON results is usually small compared to **GPU/CPU VM uptime**; the expensive mistake is leaving instances running. Detached runs terminate the VM by default after uploading artifacts.

### RGB image benchmarks (all libraries)

```bash
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml --data-dir /path/to/images --output /path/to/output
```

### RGB image benchmarks (single library)

```bash
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml --data-dir /path/to/images --output /path/to/output --libraries albumentationsx
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml --data-dir /path/to/images --output /path/to/output --libraries torchvision
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml --data-dir /path/to/images --output /path/to/output --libraries kornia
```

### Multi-channel image benchmarks (9ch, all libraries)

```bash
python -m benchmark.cli run --config configs/examples/local_9ch_micro_cpu.yaml --data-dir /path/to/images --output /path/to/output
```

### Multi-channel image benchmarks (9ch, single library)

```bash
python -m benchmark.cli run --config configs/examples/local_9ch_micro_cpu.yaml --data-dir /path/to/images --output /path/to/output --libraries albumentationsx
python -m benchmark.cli run --config configs/examples/local_9ch_micro_cpu.yaml --data-dir /path/to/images --output /path/to/output --libraries torchvision
python -m benchmark.cli run --config configs/examples/local_9ch_micro_cpu.yaml --data-dir /path/to/images --output /path/to/output --libraries kornia
```

### Video benchmarks (all libraries)

```bash
python -m benchmark.cli run --config configs/examples/local_video_micro_cpu.yaml --data-dir /path/to/videos --output /path/to/output
```

### Video benchmarks (single library)

```bash
python -m benchmark.cli run --config configs/examples/local_video_micro_cpu.yaml --data-dir /path/to/videos --output /path/to/output --libraries albumentationsx
python -m benchmark.cli run --config configs/examples/local_video_micro_cpu.yaml --data-dir /path/to/videos --output /path/to/output --libraries torchvision
python -m benchmark.cli run --config configs/examples/local_video_micro_cpu.yaml --data-dir /path/to/videos --output /path/to/output --libraries kornia
```

After running benchmarks, update the README tables with:

```bash
./tools/update_docs.sh
# Or with custom result dirs:
./tools/update_docs.sh --image-results output/ --video-results output_videos/
```

### Using Custom Transforms

To benchmark transforms, create a Python file defining `LIBRARY` and `CUSTOM_TRANSFORMS`:

```python
# my_transforms.py
import albumentations as A

# Specify the library
LIBRARY = "albumentationsx"

CUSTOM_TRANSFORMS = [
    # Test different parameters of the same transform
    A.ToGray(method="weighted_average", p=1),
    A.ToGray(method="pca", p=1),

    # Different noise levels
    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
    A.GaussNoise(var_limit=(100.0, 200.0), p=1),

    # Any other transforms...
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
]
```

Then reference it from a YAML config:

```bash
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml --spec my_transforms.py
```

The results will show each transform with all its parameters:
- `ToGray(method=weighted_average, p=1)`
- `ToGray(method=pca, p=1)`
- `GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1, per_channel=True)`

See `examples/custom_video_specs_template.py` and `example_direct_transforms.py` for more examples.

To analyze parametric results:

```bash
python tools/analyze_parametric_results.py parametric_results.json
```

This will show:
- Best and worst configurations for each transform
- Performance differences between parameter choices
- Optimal settings for your use case

## Architecture

The implementation is split between a control plane and timing engines:

- `benchmark/parser.py`: argument parsing and CLI override tracking.
- `benchmark/cli.py`: command handlers and typed config execution.
- `benchmark/matrix.py`: declarative scenario/library/mode matrix.
- `benchmark/policy.py`: shared media defaults and slow-transform policy.
- `benchmark/jobs.py`: immutable `BenchmarkJob` plus subprocess command construction.
- `benchmark/orchestrator.py`: backend dispatch, including DALI image/video pipeline jobs.
- `benchmark/envs.py`: virtualenvs, requirement refresh, and dependency cache keys.
- `benchmark/specs/load.py`: transform spec loading and validation.
- `benchmark/media/loaders.py`: RGB, 9-channel, and video media loading for micro benchmarks.
- `benchmark/pyperf_micro_runner.py`: production micro timing engine.
- `benchmark/pipeline_runner.py`: DataLoader/pipeline timing engine.
- `benchmark/runner.py`: compatibility/simple-timer runner.

See `docs/benchmark_architecture.md` for extension rules and the test files that protect this split.

## Methodology

The detailed methodology source is [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md). It describes the
measurement scopes, transform-set policy, environment isolation, media loading, micro timing, DataLoader timing, GPU and
DALI handling, slow-transform guard, result metadata, and cloud execution model.

In short: micro benchmarks are preloaded augmentation-only profilers, DataLoader benchmarks are production-style recipe
measurements, GPU rows are labeled separately with transfer/synchronization semantics, and unsupported or early-stopped
rows remain visible so coverage and throughput can be interpreted together.

## Contributing

Contributions are welcome! If you'd like to add support for a new library, improve the benchmarking methodology, or fix issues, please submit a pull request.

When contributing, please:
1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass

<!-- GitAds-Verify: ROVYUM6GM9I4GUYXL61ND2O2ZT2SVPGP -->
