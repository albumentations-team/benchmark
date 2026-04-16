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
    - [Multi-Channel Image Benchmarks (9ch)](#multi-channel-image-benchmarks-9ch)
    - [Video Benchmarks](#video-benchmarks)
  - [Performance Highlights](#performance-highlights)
    - [Image Augmentation Performance](#image-augmentation-performance)
    - [Video Augmentation Performance](#video-augmentation-performance)
  - [AlbumentationsX vs Albumentations (MIT)](#albumentationsx-vs-albumentations-mit)
    - [Image Benchmark (RGB uint8)](#image-benchmark-rgb-uint8)
    - [Multi-channel Image Benchmark (9 channels)](#multi-channel-image-benchmark-9-channels)
    - [Video Benchmark](#video-benchmark)
  - [Requirements](#requirements)
  - [Supported Libraries](#supported-libraries)
  - [Setup](#setup)
    - [Getting Started](#getting-started)
    - [Using Your Own Data](#using-your-own-data)
  - [Running Benchmarks](#running-benchmarks)
    - [Google Cloud (detached)](#google-cloud-detached)
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

The image benchmarks compare the performance of various libraries on standard image transformations. All benchmarks are run on a single CPU thread to ensure consistent and comparable results.

<!-- IMAGE_BENCHMARK_TABLE_START -->

| Transform            | AlbumentationsX 2.1.1 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx / fastest, +/-1sd)   |
|:---------------------|:--------------------------------|:-----------------------|:-----------------------------|:-----------------------------------|
| Affine               | **1455 ± 8**                    | -                      | 264 ± 16                     | 5.50x (5.17-5.87x)                 |
| AutoContrast         | **1716 ± 12**                   | 576 ± 18               | 178 ± 2                      | 2.98x (2.87-3.10x)                 |
| Blur                 | **7462 ± 243**                  | 365 ± 8                | -                            | 20.45x (19.35-21.60x)              |
| Brightness           | **9592 ± 302**                  | 2276 ± 169             | 1681 ± 21                    | 4.21x (3.80-4.70x)                 |
| CLAHE                | **654 ± 2**                     | 109 ± 2                | -                            | 6.00x (5.86-6.15x)                 |
| CenterCrop128        | 70851 ± 1429                    | -                      | **203348 ± 7429**            | 0.35x (0.33-0.37x)                 |
| ChannelDropout       | **11359 ± 529**                 | 3065 ± 179             | -                            | 3.71x (3.34-4.12x)                 |
| ChannelShuffle       | **7997 ± 234**                  | 1446 ± 115             | 4290 ± 303                   | 1.86x (1.69-2.06x)                 |
| ColorJitter          | **1228 ± 13**                   | 100 ± 3                | 88 ± 3                       | 12.29x (11.83-12.78x)              |
| Contrast             | **9485 ± 252**                  | 2159 ± 193             | 870 ± 26                     | 4.39x (3.93-4.95x)                 |
| CornerIllumination   | **496 ± 4**                     | 350 ± 4                | -                            | 1.42x (1.39-1.45x)                 |
| Equalize             | **1286 ± 6**                    | 310 ± 17               | 588 ± 17                     | 2.19x (2.11-2.26x)                 |
| Erasing              | **23858 ± 3409**                | 776 ± 45               | 10421 ± 629                  | 2.29x (1.85-2.78x)                 |
| GaussianBlur         | **2462 ± 8**                    | 353 ± 13               | 124 ± 17                     | 6.98x (6.71-7.28x)                 |
| GaussianIllumination | **761 ± 13**                    | 428 ± 16               | -                            | 1.78x (1.68-1.88x)                 |
| GaussianNoise        | **356 ± 2**                     | 121 ± 2                | -                            | 2.94x (2.88-3.00x)                 |
| Grayscale            | **19355 ± 1955**                | 1574 ± 77              | 2206 ± 179                   | 8.78x (7.30-10.52x)                |
| HorizontalFlip       | **12706 ± 397**                 | 1128 ± 42              | 2234 ± 27                    | 5.69x (5.44-5.94x)                 |
| Hue                  | **1928 ± 18**                   | 123 ± 7                | -                            | 15.64x (14.64-16.76x)              |
| Invert               | **36192 ± 7422**                | 4412 ± 293             | 22891 ± 2484                 | 1.58x (1.13-2.14x)                 |
| JpegCompression      | **1355 ± 10**                   | 117 ± 5                | 826 ± 11                     | 1.64x (1.61-1.68x)                 |
| LinearIllumination   | 565 ± 7                         | **849 ± 22**           | -                            | 0.67x (0.64-0.69x)                 |
| LongestMaxSize       | **3679 ± 97**                   | 481 ± 36               | -                            | 7.66x (6.94-8.50x)                 |
| MotionBlur           | **4365 ± 57**                   | 117 ± 6                | -                            | 37.37x (34.95-40.08x)              |
| Normalize            | **1650 ± 8**                    | 1173 ± 39              | 947 ± 33                     | 1.41x (1.35-1.46x)                 |
| OpticalDistortion    | **816 ± 2**                     | 193 ± 4                | -                            | 4.22x (4.12-4.33x)                 |
| Pad                  | **30472 ± 888**                 | -                      | 4480 ± 129                   | 6.80x (6.42-7.21x)                 |
| Perspective          | **1198 ± 4**                    | 170 ± 5                | 217 ± 8                      | 5.51x (5.31-5.73x)                 |
| PhotoMetricDistort   | **995 ± 13**                    | -                      | 80 ± 3                       | 12.39x (11.78-13.04x)              |
| PlankianJitter       | **3191 ± 46**                   | 1578 ± 100             | -                            | 2.02x (1.87-2.19x)                 |
| PlasmaBrightness     | **178 ± 5**                     | 76 ± 2                 | -                            | 2.34x (2.22-2.48x)                 |
| PlasmaContrast       | **140 ± 3**                     | 75 ± 6                 | -                            | 1.85x (1.68-2.06x)                 |
| PlasmaShadow         | 206 ± 3                         | **211 ± 5**            | -                            | 0.98x (0.94-1.01x)                 |
| Posterize            | 15033 ± 1371                    | 709 ± 27               | **17723 ± 1380**             | 0.85x (0.72-1.00x)                 |
| RGBShift             | **4908 ± 64**                   | 1787 ± 71              | -                            | 2.75x (2.61-2.90x)                 |
| Rain                 | **2089 ± 17**                   | 1591 ± 61              | -                            | 1.31x (1.25-1.38x)                 |
| RandomCrop128        | 67835 ± 527                     | 2802 ± 40              | **112838 ± 2384**            | 0.60x (0.58-0.62x)                 |
| RandomGamma          | **15432 ± 969**                 | 226 ± 5                | -                            | 68.14x (62.50-74.04x)              |
| RandomResizedCrop    | **4331 ± 12**                   | 579 ± 6                | 789 ± 27                     | 5.49x (5.29-5.69x)                 |
| Resize               | **3544 ± 17**                   | 648 ± 15               | 271 ± 4                      | 5.47x (5.31-5.62x)                 |
| Rotate               | **3003 ± 37**                   | 330 ± 7                | 319 ± 8                      | 9.08x (8.78-9.40x)                 |
| SaltAndPepper        | **626 ± 6**                     | 450 ± 5                | -                            | 1.39x (1.36-1.42x)                 |
| Saturation           | **1336 ± 26**                   | 132 ± 4                | -                            | 10.15x (9.68-10.65x)               |
| Sharpen              | **2278 ± 27**                   | 263 ± 14               | 274 ± 9                      | 8.30x (7.95-8.68x)                 |
| Shear                | **1308 ± 6**                    | 358 ± 11               | -                            | 3.65x (3.52-3.79x)                 |
| SmallestMaxSize      | **2485 ± 16**                   | 375 ± 10               | -                            | 6.63x (6.42-6.85x)                 |
| Snow                 | **736 ± 4**                     | 129 ± 4                | -                            | 5.70x (5.50-5.92x)                 |
| Solarize             | **15022 ± 984**                 | 262 ± 3                | 1117 ± 35                    | 13.45x (12.18-14.79x)              |
| ThinPlateSpline      | **83 ± 1**                      | 61 ± 2                 | -                            | 1.36x (1.31-1.42x)                 |
| VerticalFlip         | 26548 ± 2260                    | 2387 ± 58              | **26928 ± 4799**             | 0.99x (0.77-1.30x)                 |

<!-- IMAGE_BENCHMARK_TABLE_END -->

### Multi-Channel Image Benchmarks (9ch)

Benchmarks on 9-channel images (3x stacked RGB) to test OpenCV chunking and library support for >4 channels.

<!-- MULTICHANNEL_BENCHMARK_TABLE_START -->

| Transform            | AlbumentationsX 2.1.1 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx / fastest, +/-1sd)   |
|:---------------------|:--------------------------------|:-----------------------|:-----------------------------|:-----------------------------------|
| Affine               | **880 ± 20**                    | 228 ± 3                | 143 ± 3                      | 3.86x (3.72-4.00x)                 |
| AutoContrast         | **662 ± 19**                    | 374 ± 3                | -                            | 1.77x (1.70-1.84x)                 |
| Blur                 | **3165 ± 31**                   | 186 ± 3                | -                            | 16.98x (16.51-17.46x)              |
| Brightness           | **5682 ± 138**                  | 1350 ± 40              | -                            | 4.21x (3.99-4.44x)                 |
| CenterCrop128        | 74581 ± 12008                   | -                      | **223574 ± 5049**            | 0.33x (0.27-0.40x)                 |
| ChannelDropout       | **6556 ± 830**                  | 2179 ± 95              | -                            | 3.01x (2.52-3.54x)                 |
| ChannelShuffle       | **3117 ± 212**                  | 929 ± 25               | 1600 ± 41                    | 1.95x (1.77-2.13x)                 |
| Contrast             | **5576 ± 631**                  | 1346 ± 31              | -                            | 4.14x (3.59-4.72x)                 |
| CornerIllumination   | **275 ± 9**                     | 181 ± 3                | -                            | 1.52x (1.45-1.60x)                 |
| Erasing              | **10944 ± 368**                 | 426 ± 10               | 4321 ± 384                   | 2.53x (2.25-2.87x)                 |
| GaussianBlur         | **981 ± 19**                    | 188 ± 2                | 49 ± 6                       | 5.22x (5.06-5.39x)                 |
| GaussianIllumination | **296 ± 11**                    | 212 ± 15               | -                            | 1.40x (1.25-1.56x)                 |
| GaussianNoise        | **131 ± 7**                     | 65 ± 0                 | -                            | 2.01x (1.89-2.13x)                 |
| HorizontalFlip       | 3155 ± 71                       | 2286 ± 557             | **15102 ± 3640**             | 0.21x (0.16-0.28x)                 |
| Invert               | 9115 ± 1441                     | 2774 ± 169             | **15806 ± 3070**             | 0.58x (0.41-0.83x)                 |
| LinearIllumination   | 247 ± 5                         | **491 ± 12**           | -                            | 0.50x (0.48-0.53x)                 |
| LongestMaxSize       | **879 ± 12**                    | 376 ± 2                | -                            | 2.34x (2.29-2.39x)                 |
| MotionBlur           | **1978 ± 126**                  | 63 ± 1                 | -                            | 31.44x (29.17-33.75x)              |
| Normalize            | 477 ± 46                        | **1402 ± 64**          | 795 ± 22                     | 0.34x (0.29-0.39x)                 |
| OpticalDistortion    | **653 ± 14**                    | 157 ± 4                | -                            | 4.16x (3.97-4.35x)                 |
| Pad                  | **9499 ± 1173**                 | -                      | 9112 ± 704                   | 1.04x (0.85-1.27x)                 |
| Perspective          | **788 ± 10**                    | 149 ± 1                | 129 ± 2                      | 5.30x (5.19-5.41x)                 |
| PlasmaBrightness     | **126 ± 2**                     | 24 ± 1                 | -                            | 5.21x (4.93-5.51x)                 |
| PlasmaContrast       | **89 ± 1**                      | 24 ± 1                 | -                            | 3.69x (3.55-3.83x)                 |
| PlasmaShadow         | 177 ± 4                         | **224 ± 2**            | -                            | 0.79x (0.77-0.82x)                 |
| Posterize            | 5500 ± 485                      | 317 ± 16               | **12018 ± 1989**             | 0.46x (0.36-0.60x)                 |
| RandomCrop128        | 62467 ± 7734                    | 2566 ± 75              | **124539 ± 2345**            | 0.50x (0.43-0.57x)                 |
| RandomGamma          | **6328 ± 72**                   | 83 ± 0                 | -                            | 76.70x (75.37-78.04x)              |
| RandomResizedCrop    | **969 ± 109**                   | 309 ± 2                | 297 ± 3                      | 3.14x (2.77-3.51x)                 |
| Resize               | **802 ± 14**                    | 297 ± 3                | 194 ± 1                      | 2.70x (2.63-2.78x)                 |
| Rotate               | **2833 ± 141**                  | 172 ± 1                | 152 ± 10                     | 16.48x (15.53-17.44x)              |
| Sharpen              | **969 ± 17**                    | 140 ± 6                | -                            | 6.92x (6.51-7.35x)                 |
| Shear                | **931 ± 19**                    | 250 ± 2                | 163 ± 6                      | 3.73x (3.62-3.83x)                 |
| SmallestMaxSize      | **585 ± 5**                     | 187 ± 3                | -                            | 3.13x (3.06-3.21x)                 |
| Solarize             | **5822 ± 780**                  | 339 ± 4                | 456 ± 11                     | 12.77x (10.80-14.84x)              |
| ThinPlateSpline      | **97 ± 1**                      | 62 ± 0                 | -                            | 1.56x (1.55-1.58x)                 |
| VerticalFlip         | 11827 ± 1427                    | 2296 ± 118             | **15409 ± 890**              | 0.77x (0.64-0.91x)                 |

<!-- MULTICHANNEL_BENCHMARK_TABLE_END -->

### Video Benchmarks

The video benchmarks compare CPU-based processing (AlbumentationsX) with GPU-accelerated processing (Kornia) for video transformations. The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains realistic videos from 101 action categories.

For AlbumentationsX and Albumentations (MIT), each clip is a NumPy array `(T, H, W, C)`. The built-in spec files apply augmentations with `transform(images=video)["images"]`—Albumentations’ batch video API—so parameters are drawn once per clip and shared across frames, in line with typical video training and with Kornia’s `same_on_batch=True` for a fair comparison.

<!-- VIDEO_BENCHMARK_TABLE_START -->

| Transform            | AlbumentationsX (video) 2.1.1 [vid/s]   | kornia (video) 0.8.0 [vid/s]   | torchvision (video) 0.21.0 [vid/s]   | Speedup (albx / fastest, +/-1sd)   |
|:---------------------|:----------------------------------------|:-------------------------------|:-------------------------------------|:-----------------------------------|
| Affine               | 19 ± 0                                  | 21 ± 0                         | **453 ± 0**                          | 0.04x (0.04-0.04x)                 |
| AutoContrast         | 18 ± 0                                  | 21 ± 0                         | **578 ± 17**                         | 0.03x (0.03-0.03x)                 |
| Blur                 | **61 ± 1**                              | 21 ± 0                         | -                                    | 2.97x (2.91-3.02x)                 |
| Brightness           | 82 ± 4                                  | 22 ± 0                         | **756 ± 435**                        | 0.11x (0.07-0.27x)                 |
| CenterCrop128        | 387 ± 16                                | 70 ± 1                         | **1133 ± 235**                       | 0.34x (0.27-0.45x)                 |
| ChannelDropout       | **77 ± 0**                              | 22 ± 0                         | -                                    | 3.54x (3.52-3.56x)                 |
| ChannelShuffle       | 67 ± 4                                  | 20 ± 0                         | **958 ± 0**                          | 0.07x (0.07-0.07x)                 |
| ColorJitter          | 15 ± 0                                  | 19 ± 0                         | **69 ± 0**                           | 0.21x (0.20-0.22x)                 |
| Contrast             | 76 ± 1                                  | 22 ± 0                         | **547 ± 13**                         | 0.14x (0.14-0.14x)                 |
| CornerIllumination   | **6 ± 0**                               | 3 ± 0                          | -                                    | 2.42x (2.31-2.55x)                 |
| Elastic              | 7 ± 0                                   | -                              | **127 ± 1**                          | 0.05x (0.05-0.06x)                 |
| Equalize             | 12 ± 0                                  | 4 ± 0                          | **192 ± 1**                          | 0.06x (0.06-0.06x)                 |
| Erasing              | 86 ± 2                                  | -                              | **255 ± 7**                          | 0.34x (0.32-0.35x)                 |
| GaussianBlur         | 30 ± 1                                  | 22 ± 0                         | **543 ± 11**                         | 0.06x (0.05-0.06x)                 |
| GaussianIllumination | 9 ± 0                                   | **20 ± 0**                     | -                                    | 0.43x (0.40-0.45x)                 |
| GaussianNoise        | 4 ± 0                                   | **22 ± 0**                     | -                                    | 0.18x (0.18-0.19x)                 |
| Grayscale            | 87 ± 6                                  | 22 ± 0                         | **838 ± 467**                        | 0.10x (0.06-0.25x)                 |
| HorizontalFlip       | 73 ± 1                                  | 22 ± 0                         | **978 ± 49**                         | 0.07x (0.07-0.08x)                 |
| Hue                  | **20 ± 0**                              | 20 ± 0                         | -                                    | 1.00x (0.98-1.02x)                 |
| Invert               | 101 ± 1                                 | 22 ± 0                         | **843 ± 176**                        | 0.12x (0.10-0.15x)                 |
| LinearIllumination   | **7 ± 0**                               | 4 ± 0                          | -                                    | 1.68x (1.59-1.77x)                 |
| MedianBlur           | **20 ± 0**                              | 8 ± 0                          | -                                    | 2.33x (2.29-2.37x)                 |
| Normalize            | 17 ± 1                                  | 22 ± 0                         | **461 ± 0**                          | 0.04x (0.04-0.04x)                 |
| Pad                  | 80 ± 2                                  | -                              | **760 ± 338**                        | 0.10x (0.07-0.19x)                 |
| Perspective          | 16 ± 0                                  | -                              | **435 ± 0**                          | 0.04x (0.04-0.04x)                 |
| PlankianJitter       | **35 ± 1**                              | 11 ± 0                         | -                                    | 3.21x (3.12-3.29x)                 |
| PlasmaBrightness     | 2 ± 0                                   | **17 ± 0**                     | -                                    | 0.09x (0.09-0.10x)                 |
| PlasmaContrast       | 1 ± 0                                   | **17 ± 0**                     | -                                    | 0.07x (0.07-0.07x)                 |
| PlasmaShadow         | 2 ± 0                                   | **19 ± 0**                     | -                                    | 0.09x (0.09-0.10x)                 |
| Posterize            | 85 ± 7                                  | -                              | **631 ± 15**                         | 0.13x (0.12-0.15x)                 |
| RGBShift             | **47 ± 2**                              | 22 ± 0                         | -                                    | 2.10x (1.99-2.21x)                 |
| Rain                 | **27 ± 0**                              | 4 ± 0                          | -                                    | 7.10x (7.03-7.17x)                 |
| RandomCrop128        | 375 ± 11                                | 65 ± 0                         | **1133 ± 15**                        | 0.33x (0.32-0.35x)                 |
| RandomGamma          | **78 ± 5**                              | 22 ± 0                         | -                                    | 3.59x (3.33-3.85x)                 |
| RandomResizedCrop    | 20 ± 1                                  | 6 ± 0                          | **182 ± 16**                         | 0.11x (0.10-0.13x)                 |
| Resize               | 19 ± 0                                  | 6 ± 0                          | **140 ± 35**                         | 0.14x (0.11-0.18x)                 |
| Rotate               | 30 ± 0                                  | 22 ± 0                         | **534 ± 0**                          | 0.06x (0.06-0.06x)                 |
| SaltAndPepper        | **10 ± 0**                              | 9 ± 0                          | -                                    | 1.17x (1.13-1.22x)                 |
| Saturation           | 11 ± 1                                  | **37 ± 0**                     | -                                    | 0.30x (0.28-0.31x)                 |
| Sharpen              | 30 ± 1                                  | 18 ± 0                         | **420 ± 9**                          | 0.07x (0.07-0.07x)                 |
| Solarize             | 88 ± 3                                  | 21 ± 0                         | **628 ± 6**                          | 0.14x (0.14-0.15x)                 |
| ThinPlateSpline      | 1 ± 0                                   | **45 ± 1**                     | -                                    | 0.03x (0.03-0.03x)                 |
| VerticalFlip         | 90 ± 4                                  | 22 ± 0                         | **978 ± 5**                          | 0.09x (0.09-0.10x)                 |

<!-- VIDEO_BENCHMARK_TABLE_END -->

## Performance Highlights

### Image Augmentation Performance

<!-- IMAGE_SPEEDUP_SUMMARY_START -->

See the full benchmark table above for image results.

<!-- IMAGE_SPEEDUP_SUMMARY_END -->

### Video Augmentation Performance

<!-- VIDEO_SPEEDUP_SUMMARY_START -->

See the full benchmark table above for video results.

<!-- VIDEO_SPEEDUP_SUMMARY_END -->

## AlbumentationsX vs Albumentations (MIT)

Head-to-head comparison between AlbumentationsX (commercial/AGPL) and Albumentations 2.0.8 (MIT) across the full shared transform API.

### Image Benchmark (RGB uint8)

<!-- ALBX_VS_ALB_IMAGE_TABLE_START -->

| Transform                | Albumentations (MIT) 2.0.8 [img/s]   | AlbumentationsX 2.1.1 [img/s]   | Speedup (albx / MIT, +/-1sd)   |
|:-------------------------|:-------------------------------------|:--------------------------------|:-------------------------------|
| AdditiveNoise            | 246 ± 2                              | **265 ± 3**                     | 1.08x (1.06-1.10x)             |
| AdvancedBlur             | 1192 ± 42                            | **1267 ± 58**                   | 1.06x (0.98-1.15x)             |
| Affine                   | 1438 ± 15                            | **1455 ± 8**                    | 1.01x (1.00-1.03x)             |
| AutoContrast             | 1684 ± 53                            | **1716 ± 12**                   | 1.02x (0.98-1.06x)             |
| Blur                     | **7699 ± 348**                       | 7462 ± 243                      | 0.97x (0.90-1.05x)             |
| Brightness               | **11407 ± 992**                      | 9592 ± 302                      | 0.84x (0.75-0.95x)             |
| CLAHE                    | 640 ± 12                             | **654 ± 2**                     | 1.02x (1.00-1.05x)             |
| CenterCrop128            | **118961 ± 4000**                    | 70851 ± 1429                    | 0.60x (0.56-0.63x)             |
| ChannelDropout           | **12059 ± 1067**                     | 11359 ± 529                     | 0.94x (0.83-1.08x)             |
| ChannelShuffle           | **8076 ± 235**                       | 7997 ± 234                      | 0.99x (0.93-1.05x)             |
| ChromaticAberration      | 500 ± 12                             | **516 ± 8**                     | 1.03x (0.99-1.07x)             |
| CoarseDropout            | **20042 ± 2999**                     | 16647 ± 1134                    | 0.83x (0.67-1.04x)             |
| ColorJitter              | 1012 ± 12                            | **1228 ± 13**                   | 1.21x (1.19-1.24x)             |
| ConstrainedCoarseDropout | **380677 ± 4476**                    | 108677 ± 1894                   | 0.29x (0.28-0.29x)             |
| Contrast                 | **11873 ± 954**                      | 9485 ± 252                      | 0.80x (0.72-0.89x)             |
| CornerIllumination       | 457 ± 21                             | **496 ± 4**                     | 1.09x (1.03-1.15x)             |
| CropAndPad               | **3019 ± 49**                        | 2770 ± 57                       | 0.92x (0.88-0.95x)             |
| Defocus                  | 129 ± 0                              | **130 ± 2**                     | 1.01x (0.99-1.02x)             |
| Downscale                | 5203 ± 108                           | **5284 ± 197**                  | 1.02x (0.96-1.08x)             |
| Elastic                  | 316 ± 6                              | **448 ± 1**                     | 1.42x (1.39-1.45x)             |
| Emboss                   | 2695 ± 79                            | **2790 ± 30**                   | 1.04x (1.00-1.08x)             |
| Equalize                 | 1274 ± 13                            | **1286 ± 6**                    | 1.01x (0.99-1.02x)             |
| Erasing                  | 23703 ± 7611                         | **23858 ± 3409**                | 1.01x (0.65-1.69x)             |
| FancyPCA                 | 105 ± 5                              | **105 ± 3**                     | 1.00x (0.93-1.08x)             |
| GaussianBlur             | **2474 ± 23**                        | 2462 ± 8                        | 1.00x (0.98-1.01x)             |
| GaussianIllumination     | 685 ± 51                             | **761 ± 13**                    | 1.11x (1.02-1.22x)             |
| GaussianNoise            | 335 ± 13                             | **356 ± 2**                     | 1.07x (1.02-1.12x)             |
| GlassBlur                | **37 ± 1**                           | 35 ± 0                          | 0.96x (0.93-1.00x)             |
| Grayscale                | 18109 ± 520                          | **19355 ± 1955**                | 1.07x (0.93-1.21x)             |
| GridDistortion           | 796 ± 50                             | **1334 ± 31**                   | 1.68x (1.54-1.83x)             |
| GridDropout              | **80 ± 5**                           | 77 ± 3                          | 0.96x (0.88-1.06x)             |
| HSV                      | 1086 ± 27                            | **1121 ± 10**                   | 1.03x (1.00-1.07x)             |
| HorizontalFlip           | **14182 ± 318**                      | 12706 ± 397                     | 0.90x (0.85-0.95x)             |
| Hue                      | 1837 ± 18                            | **1928 ± 18**                   | 1.05x (1.03-1.07x)             |
| ISONoise                 | 167 ± 4                              | **185 ± 1**                     | 1.11x (1.08-1.14x)             |
| Invert                   | 26181 ± 9229                         | **36192 ± 7422**                | 1.38x (0.81-2.57x)             |
| JpegCompression          | 1351 ± 20                            | **1355 ± 10**                   | 1.00x (0.98-1.03x)             |
| LinearIllumination       | 454 ± 28                             | **565 ± 7**                     | 1.24x (1.16-1.34x)             |
| LongestMaxSize           | **3705 ± 62**                        | 3679 ± 97                       | 0.99x (0.95-1.04x)             |
| MedianBlur               | 1569 ± 14                            | **1579 ± 9**                    | 1.01x (0.99-1.02x)             |
| Morphological            | 14215 ± 936                          | **14539 ± 998**                 | 1.02x (0.89-1.17x)             |
| MotionBlur               | **4484 ± 65**                        | 4365 ± 57                       | 0.97x (0.95-1.00x)             |
| MultiplicativeNoise      | 3202 ± 77                            | **4908 ± 135**                  | 1.53x (1.46-1.61x)             |
| Normalize                | 1630 ± 108                           | **1650 ± 8**                    | 1.01x (0.94-1.09x)             |
| OpticalDistortion        | 601 ± 7                              | **816 ± 2**                     | 1.36x (1.34-1.38x)             |
| Pad                      | **48440 ± 519**                      | 30472 ± 888                     | 0.63x (0.60-0.65x)             |
| PadIfNeeded              | **19659 ± 4560**                     | 13792 ± 1061                    | 0.70x (0.53-0.98x)             |
| Perspective              | 1197 ± 13                            | **1198 ± 4**                    | 1.00x (0.99-1.01x)             |
| PhotoMetricDistort       | 917 ± 20                             | **995 ± 13**                    | 1.09x (1.05-1.12x)             |
| PixelDropout             | **393 ± 10**                         | 371 ± 4                         | 0.94x (0.91-0.98x)             |
| PlankianJitter           | 3132 ± 63                            | **3191 ± 46**                   | 1.02x (0.98-1.05x)             |
| PlasmaBrightness         | 169 ± 3                              | **178 ± 5**                     | 1.05x (1.01-1.10x)             |
| PlasmaContrast           | **148 ± 4**                          | 140 ± 3                         | 0.95x (0.91-0.99x)             |
| PlasmaShadow             | 191 ± 7                              | **206 ± 3**                     | 1.08x (1.03-1.13x)             |
| Posterize                | 13473 ± 1217                         | **15033 ± 1371**                | 1.12x (0.93-1.34x)             |
| RGBShift                 | 3436 ± 26                            | **4908 ± 64**                   | 1.43x (1.40-1.46x)             |
| Rain                     | 1946 ± 26                            | **2089 ± 17**                   | 1.07x (1.05-1.10x)             |
| RandomCrop128            | **113738 ± 2162**                    | 67835 ± 527                     | 0.60x (0.58-0.61x)             |
| RandomFog                | **9 ± 0**                            | 9 ± 0                           | 1.00x (0.97-1.03x)             |
| RandomGamma              | 12076 ± 855                          | **15432 ± 969**                 | 1.28x (1.12-1.46x)             |
| RandomGravel             | **1376 ± 10**                        | 1303 ± 22                       | 0.95x (0.92-0.97x)             |
| RandomGridShuffle        | 9290 ± 437                           | **10399 ± 627**                 | 1.12x (1.00-1.25x)             |
| RandomResizedCrop        | **4379 ± 12**                        | 4331 ± 12                       | 0.99x (0.98-0.99x)             |
| RandomRotate90           | 1876 ± 53                            | **1957 ± 46**                   | 1.04x (0.99-1.10x)             |
| RandomScale              | 3140 ± 55                            | **3405 ± 41**                   | 1.08x (1.05-1.12x)             |
| RandomShadow             | 418 ± 7                              | **516 ± 12**                    | 1.24x (1.19-1.29x)             |
| RandomSizedCrop          | **3954 ± 23**                        | 3638 ± 39                       | 0.92x (0.90-0.94x)             |
| RandomSunFlare           | 334 ± 3                              | **348 ± 2**                     | 1.04x (1.03-1.06x)             |
| RandomToneCurve          | 10625 ± 797                          | **11951 ± 403**                 | 1.12x (1.01-1.26x)             |
| Resize                   | 3535 ± 11                            | **3544 ± 17**                   | 1.00x (0.99-1.01x)             |
| RingingOvershoot         | 156 ± 1                              | **157 ± 1**                     | 1.00x (0.99-1.02x)             |
| Rotate                   | 2938 ± 29                            | **3003 ± 37**                   | 1.02x (1.00-1.04x)             |
| SafeRotate               | 1293 ± 12                            | **1381 ± 7**                    | 1.07x (1.05-1.08x)             |
| SaltAndPepper            | 578 ± 9                              | **626 ± 6**                     | 1.08x (1.06-1.11x)             |
| Saturation               | 1169 ± 40                            | **1336 ± 26**                   | 1.14x (1.08-1.21x)             |
| Sharpen                  | **2337 ± 11**                        | 2278 ± 27                       | 0.98x (0.96-0.99x)             |
| Shear                    | 1257 ± 13                            | **1308 ± 6**                    | 1.04x (1.03-1.06x)             |
| ShiftScaleRotate         | 1269 ± 15                            | **1370 ± 2**                    | 1.08x (1.06-1.09x)             |
| ShotNoise                | 41 ± 0                               | **46 ± 0**                      | 1.10x (1.09-1.12x)             |
| SmallestMaxSize          | 2483 ± 39                            | **2485 ± 16**                   | 1.00x (0.98-1.02x)             |
| Snow                     | **758 ± 11**                         | 736 ± 4                         | 0.97x (0.95-0.99x)             |
| Solarize                 | 11616 ± 807                          | **15022 ± 984**                 | 1.29x (1.13-1.48x)             |
| Spatter                  | 111 ± 2                              | **114 ± 2**                     | 1.03x (0.99-1.08x)             |
| SquareSymmetry           | 2080 ± 59                            | **2232 ± 76**                   | 1.07x (1.01-1.14x)             |
| Superpixels              | **19 ± 0**                           | 18 ± 0                          | 0.95x (0.93-0.96x)             |
| ThinPlateSpline          | 78 ± 2                               | **83 ± 1**                      | 1.08x (1.04-1.11x)             |
| ToSepia                  | **7672 ± 463**                       | 6731 ± 194                      | 0.88x (0.80-0.96x)             |
| Transpose                | 1513 ± 20                            | **1601 ± 13**                   | 1.06x (1.04-1.08x)             |
| UnsharpMask              | 201 ± 4                              | **357 ± 12**                    | 1.77x (1.68-1.86x)             |
| VerticalFlip             | **31529 ± 519**                      | 26548 ± 2260                    | 0.84x (0.76-0.93x)             |
| ZoomBlur                 | 186 ± 10                             | **211 ± 3**                     | 1.13x (1.06-1.21x)             |

<!-- ALBX_VS_ALB_IMAGE_TABLE_END -->

### Multi-channel Image Benchmark (9 channels)

<!-- ALBX_VS_ALB_MULTICHANNEL_TABLE_START -->

| Transform                | Albumentations (MIT) 2.0.8 [img/s]   | AlbumentationsX 2.1.1 [img/s]   | Speedup (albx / MIT, +/-1sd)   |
|:-------------------------|:-------------------------------------|:--------------------------------|:-------------------------------|
| AdditiveNoise            | 87 ± 1                               | **112 ± 1**                     | 1.29x (1.26-1.33x)             |
| AdvancedBlur             | 199 ± 3                              | **818 ± 305**                   | 4.11x (2.54-5.74x)             |
| Affine                   | 237 ± 2                              | **880 ± 20**                    | 3.71x (3.59-3.82x)             |
| AutoContrast             | 472 ± 3                              | **662 ± 19**                    | 1.40x (1.35-1.45x)             |
| Blur                     | 312 ± 6                              | **3165 ± 31**                   | 10.15x (9.88-10.44x)           |
| Brightness               | 4268 ± 39                            | **5682 ± 138**                  | 1.33x (1.29-1.38x)             |
| CenterCrop128            | 51105 ± 1026                         | **74581 ± 12008**               | 1.46x (1.20-1.73x)             |
| ChannelDropout           | **7176 ± 221**                       | 6556 ± 830                      | 0.91x (0.77-1.06x)             |
| ChannelShuffle           | 2619 ± 14                            | **3117 ± 212**                  | 1.19x (1.10-1.28x)             |
| CoarseDropout            | 8698 ± 259                           | **9047 ± 339**                  | 1.04x (0.97-1.11x)             |
| ConstrainedCoarseDropout | **387459 ± 3701**                    | 119167 ± 26242                  | 0.31x (0.24-0.38x)             |
| Contrast                 | 4257 ± 69                            | **5576 ± 631**                  | 1.31x (1.14-1.48x)             |
| CornerIllumination       | 223 ± 2                              | **275 ± 9**                     | 1.23x (1.18-1.28x)             |
| CropAndPad               | 167 ± 0                              | **868 ± 10**                    | 5.19x (5.12-5.26x)             |
| Defocus                  | 40 ± 0                               | **52 ± 0**                      | 1.31x (1.30-1.32x)             |
| Downscale                | 2746 ± 81                            | **3916 ± 136**                  | 1.43x (1.34-1.52x)             |
| ElasticTransform         | 139 ± 3                              | **418 ± 7**                     | 3.00x (2.88-3.13x)             |
| Emboss                   | 262 ± 1                              | **1205 ± 14**                   | 4.61x (4.54-4.67x)             |
| Erasing                  | 10476 ± 417                          | **10944 ± 368**                 | 1.04x (0.97-1.12x)             |
| GaussianBlur             | 248 ± 8                              | **981 ± 19**                    | 3.96x (3.77-4.16x)             |
| GaussianIllumination     | 263 ± 4                              | **296 ± 11**                    | 1.13x (1.07-1.19x)             |
| GaussianNoise            | 111 ± 1                              | **131 ± 7**                     | 1.18x (1.11-1.25x)             |
| Grayscale                | 431 ± 2                              | **515 ± 4**                     | 1.20x (1.18-1.21x)             |
| GridDistortion           | 212 ± 1                              | **810 ± 9**                     | 3.82x (3.76-3.88x)             |
| GridDropout              | 80 ± 4                               | **81 ± 78**                     | 1.01x (0.03-2.09x)             |
| HorizontalFlip           | 2650 ± 133                           | **3155 ± 71**                   | 1.19x (1.11-1.28x)             |
| Invert                   | **10552 ± 497**                      | 9115 ± 1441                     | 0.86x (0.69-1.05x)             |
| JpegCompression          | 162 ± 0                              | **205 ± 3**                     | 1.27x (1.25-1.29x)             |
| LinearIllumination       | 160 ± 2                              | **247 ± 5**                     | 1.54x (1.49-1.59x)             |
| LongestMaxSize           | 329 ± 6                              | **879 ± 12**                    | 2.67x (2.59-2.76x)             |
| MedianBlur               | 212 ± 1                              | **546 ± 15**                    | 2.57x (2.49-2.65x)             |
| Morphological            | 5724 ± 68                            | **6610 ± 429**                  | 1.15x (1.07-1.24x)             |
| MotionBlur               | 296 ± 3                              | **1978 ± 126**                  | 6.69x (6.21-7.18x)             |
| MultiplicativeNoise      | 805 ± 5                              | **2089 ± 71**                   | 2.60x (2.49-2.70x)             |
| Normalize                | **487 ± 9**                          | 477 ± 46                        | 0.98x (0.87-1.09x)             |
| OpticalDistortion        | 181 ± 6                              | **653 ± 14**                    | 3.60x (3.41-3.82x)             |
| Pad                      | 321 ± 8                              | **9499 ± 1173**                 | 29.56x (25.27-34.08x)          |
| PadIfNeeded              | 122 ± 1                              | **8504 ± 352**                  | 69.83x (66.60-73.08x)          |
| Perspective              | 212 ± 2                              | **788 ± 10**                    | 3.71x (3.63-3.80x)             |
| PixelDropout             | 305 ± 1                              | **371 ± 7**                     | 1.22x (1.19-1.25x)             |
| PlasmaBrightness         | 94 ± 0                               | **126 ± 2**                     | 1.34x (1.31-1.37x)             |
| PlasmaContrast           | 75 ± 1                               | **89 ± 1**                      | 1.19x (1.16-1.21x)             |
| PlasmaShadow             | 139 ± 2                              | **177 ± 4**                     | 1.28x (1.23-1.32x)             |
| Posterize                | 4341 ± 76                            | **5500 ± 485**                  | 1.27x (1.14-1.40x)             |
| RandomCrop128            | 49737 ± 691                          | **62467 ± 7734**                | 1.26x (1.09-1.43x)             |
| RandomGamma              | 4359 ± 42                            | **6328 ± 72**                   | 1.45x (1.42-1.48x)             |
| RandomGridShuffle        | **5037 ± 196**                       | 4668 ± 657                      | 0.93x (0.77-1.10x)             |
| RandomResizedCrop        | 387 ± 5                              | **969 ± 109**                   | 2.51x (2.19-2.83x)             |
| RandomRotate90           | 1484 ± 27                            | **2137 ± 1277**                 | 1.44x (0.57-2.34x)             |
| RandomScale              | 319 ± 8                              | **882 ± 32**                    | 2.77x (2.60-2.95x)             |
| RandomSizedCrop          | 378 ± 4                              | **828 ± 37**                    | 2.19x (2.07-2.31x)             |
| Resize                   | 289 ± 4                              | **802 ± 14**                    | 2.77x (2.69-2.86x)             |
| RingingOvershoot         | 46 ± 0                               | **61 ± 32**                     | 1.33x (0.62-2.06x)             |
| Rotate                   | 293 ± 6                              | **2833 ± 141**                  | 9.68x (9.00-10.38x)            |
| SafeRotate               | 232 ± 1                              | **983 ± 76**                    | 4.23x (3.89-4.57x)             |
| SaltAndPepper            | 438 ± 9                              | **530 ± 9**                     | 1.21x (1.16-1.26x)             |
| Sharpen                  | 249 ± 0                              | **969 ± 17**                    | 3.90x (3.82-3.97x)             |
| Shear                    | 230 ± 1                              | **931 ± 19**                    | 4.04x (3.94-4.14x)             |
| ShiftScaleRotate         | 231 ± 1                              | **926 ± 44**                    | 4.01x (3.80-4.22x)             |
| SmallestMaxSize          | 249 ± 2                              | **585 ± 5**                     | 2.35x (2.31-2.39x)             |
| Solarize                 | 4217 ± 39                            | **5822 ± 780**                  | 1.38x (1.18-1.58x)             |
| SquareSymmetry           | 1491 ± 25                            | **2325 ± 1080**                 | 1.56x (0.82-2.32x)             |
| ThinPlateSpline          | 62 ± 1                               | **97 ± 1**                      | 1.57x (1.54-1.61x)             |
| Transpose                | 1459 ± 33                            | **2081 ± 88**                   | 1.43x (1.34-1.52x)             |
| UnsharpMask              | 48 ± 1                               | **143 ± 5**                     | 3.00x (2.84-3.16x)             |
| VerticalFlip             | 10223 ± 69                           | **11827 ± 1427**                | 1.16x (1.01-1.31x)             |

<!-- ALBX_VS_ALB_MULTICHANNEL_TABLE_END -->

### Video Benchmark

<!-- ALBX_VS_ALB_VIDEO_TABLE_START -->

| Transform                | Albumentations (MIT) (video) 2.0.8 [vid/s]   | AlbumentationsX (video) 2.1.1 [vid/s]   | Speedup (albx / MIT, +/-1sd)   |
|:-------------------------|:---------------------------------------------|:----------------------------------------|:-------------------------------|
| AdditiveNoise            | **4 ± 0**                                    | 4 ± 0                                   | 0.96x (0.94-0.98x)             |
| AdvancedBlur             | 18 ± 0                                       | **18 ± 0**                              | 1.01x (0.98-1.05x)             |
| Affine                   | 18 ± 0                                       | **19 ± 0**                              | 1.03x (1.01-1.05x)             |
| AutoContrast             | 17 ± 0                                       | **18 ± 0**                              | 1.04x (1.02-1.06x)             |
| Blur                     | 52 ± 3                                       | **61 ± 1**                              | 1.17x (1.10-1.25x)             |
| Brightness               | 68 ± 2                                       | **82 ± 4**                              | 1.21x (1.12-1.30x)             |
| CLAHE                    | **10 ± 0**                                   | 10 ± 0                                  | 0.99x (0.94-1.04x)             |
| CenterCrop128            | 381 ± 28                                     | **387 ± 16**                            | 1.01x (0.90-1.14x)             |
| ChannelDropout           | 76 ± 8                                       | **77 ± 0**                              | 1.02x (0.92-1.14x)             |
| ChannelShuffle           | 57 ± 6                                       | **67 ± 4**                              | 1.18x (1.00-1.41x)             |
| ChromaticAberration      | **7 ± 0**                                    | 7 ± 0                                   | 1.00x (0.95-1.05x)             |
| CoarseDropout            | 62 ± 2                                       | **69 ± 3**                              | 1.10x (1.02-1.19x)             |
| ColorJitter              | 13 ± 0                                       | **15 ± 0**                              | 1.14x (1.10-1.19x)             |
| ConstrainedCoarseDropout | **566 ± 28**                                 | 362 ± 15                                | 0.64x (0.58-0.70x)             |
| Contrast                 | 64 ± 2                                       | **76 ± 1**                              | 1.19x (1.14-1.25x)             |
| CornerIllumination       | 6 ± 0                                        | **6 ± 0**                               | 1.06x (1.02-1.10x)             |
| CropAndPad               | 37 ± 0                                       | **37 ± 1**                              | 1.01x (0.98-1.05x)             |
| Defocus                  | 2 ± 0                                        | **2 ± 0**                               | 1.04x (1.03-1.06x)             |
| Downscale                | **50 ± 0**                                   | 49 ± 1                                  | 0.98x (0.95-1.01x)             |
| Elastic                  | 4 ± 0                                        | **7 ± 0**                               | 1.58x (1.47-1.69x)             |
| Emboss                   | 34 ± 1                                       | **34 ± 1**                              | 1.00x (0.94-1.07x)             |
| Equalize                 | **15 ± 0**                                   | 12 ± 0                                  | 0.81x (0.79-0.83x)             |
| Erasing                  | 76 ± 1                                       | **86 ± 2**                              | 1.14x (1.10-1.18x)             |
| FancyPCA                 | 2 ± 0                                        | **2 ± 0**                               | 1.09x (1.07-1.11x)             |
| GaussianBlur             | 30 ± 0                                       | **30 ± 1**                              | 1.00x (0.96-1.05x)             |
| GaussianIllumination     | 8 ± 0                                        | **9 ± 0**                               | 1.06x (0.98-1.14x)             |
| GaussianNoise            | 4 ± 0                                        | **4 ± 0**                               | 1.04x (1.01-1.08x)             |
| Grayscale                | 80 ± 3                                       | **87 ± 6**                              | 1.09x (0.98-1.20x)             |
| GridDistortion           | 14 ± 0                                       | **15 ± 1**                              | 1.10x (1.01-1.19x)             |
| GridDropout              | **1 ± 0**                                    | 1 ± 0                                   | 0.99x (0.96-1.02x)             |
| HSV                      | 8 ± 0                                        | **9 ± 0**                               | 1.16x (1.08-1.26x)             |
| HorizontalFlip           | 66 ± 1                                       | **73 ± 1**                              | 1.11x (1.08-1.13x)             |
| Hue                      | 17 ± 1                                       | **20 ± 0**                              | 1.15x (1.09-1.22x)             |
| ISONoise                 | 3 ± 0                                        | **3 ± 0**                               | 1.05x (1.02-1.09x)             |
| Invert                   | 86 ± 6                                       | **101 ± 1**                             | 1.18x (1.10-1.28x)             |
| JpegCompression          | 23 ± 0                                       | **23 ± 0**                              | 1.02x (0.99-1.06x)             |
| LinearIllumination       | 6 ± 0                                        | **7 ± 0**                               | 1.29x (1.26-1.32x)             |
| LongestMaxSize           | **21 ± 0**                                   | 19 ± 1                                  | 0.93x (0.86-1.01x)             |
| MedianBlur               | **20 ± 0**                                   | 20 ± 0                                  | 0.99x (0.97-1.02x)             |
| Morphological            | **90 ± 9**                                   | 71 ± 6                                  | 0.79x (0.65-0.96x)             |
| MotionBlur               | **39 ± 2**                                   | 37 ± 1                                  | 0.96x (0.89-1.04x)             |
| MultiplicativeNoise      | 40 ± 2                                       | **52 ± 1**                              | 1.30x (1.22-1.38x)             |
| Normalize                | **19 ± 0**                                   | 17 ± 1                                  | 0.91x (0.86-0.96x)             |
| OpticalDistortion        | 10 ± 0                                       | **11 ± 0**                              | 1.14x (1.07-1.22x)             |
| Pad                      | 75 ± 2                                       | **80 ± 2**                              | 1.07x (1.02-1.12x)             |
| PadIfNeeded              | 11 ± 0                                       | **13 ± 1**                              | 1.20x (1.11-1.31x)             |
| Perspective              | **17 ± 0**                                   | 16 ± 0                                  | 0.99x (0.95-1.02x)             |
| PhotoMetricDistort       | 13 ± 0                                       | **14 ± 1**                              | 1.07x (0.99-1.16x)             |
| PixelDropout             | **6 ± 0**                                    | 6 ± 0                                   | 1.00x (0.96-1.03x)             |
| PlankianJitter           | 26 ± 4                                       | **35 ± 1**                              | 1.33x (1.14-1.58x)             |
| PlasmaBrightness         | 2 ± 0                                        | **2 ± 0**                               | 1.03x (1.00-1.06x)             |
| PlasmaContrast           | **1 ± 0**                                    | 1 ± 0                                   | 0.82x (0.81-0.84x)             |
| PlasmaShadow             | 2 ± 0                                        | **2 ± 0**                               | 1.01x (1.00-1.03x)             |
| Posterize                | 70 ± 6                                       | **85 ± 7**                              | 1.21x (1.01-1.45x)             |
| RGBShift                 | 36 ± 2                                       | **47 ± 2**                              | 1.31x (1.19-1.44x)             |
| Rain                     | 27 ± 0                                       | **27 ± 0**                              | 1.01x (0.99-1.03x)             |
| RandomCrop128            | 371 ± 4                                      | **375 ± 11**                            | 1.01x (0.97-1.05x)             |
| RandomGamma              | 74 ± 3                                       | **78 ± 5**                              | 1.04x (0.93-1.16x)             |
| RandomGravel             | 20 ± 0                                       | **21 ± 0**                              | 1.05x (1.02-1.09x)             |
| RandomGridShuffle        | **55 ± 3**                                   | 55 ± 1                                  | 0.99x (0.93-1.07x)             |
| RandomResizedCrop        | 18 ± 1                                       | **20 ± 1**                              | 1.09x (1.03-1.17x)             |
| RandomShadow             | 6 ± 0                                        | **7 ± 0**                               | 1.22x (1.18-1.27x)             |
| RandomSizedCrop          | **17 ± 0**                                   | 17 ± 1                                  | 1.00x (0.91-1.09x)             |
| RandomSunFlare           | **5 ± 0**                                    | 5 ± 0                                   | 1.00x (0.99-1.00x)             |
| RandomToneCurve          | **74 ± 2**                                   | 72 ± 4                                  | 0.97x (0.90-1.05x)             |
| Resize                   | 18 ± 0                                       | **19 ± 0**                              | 1.08x (1.04-1.12x)             |
| RingingOvershoot         | 2 ± 0                                        | **2 ± 0**                               | 1.03x (1.02-1.04x)             |
| Rotate                   | 28 ± 1                                       | **30 ± 0**                              | 1.08x (1.04-1.11x)             |
| SafeRotate               | **18 ± 0**                                   | 17 ± 1                                  | 0.93x (0.84-1.02x)             |
| SaltAndPepper            | 10 ± 0                                       | **10 ± 0**                              | 1.01x (0.98-1.05x)             |
| Saturation               | 11 ± 1                                       | **11 ± 1**                              | 1.02x (0.91-1.15x)             |
| Sharpen                  | 29 ± 1                                       | **30 ± 1**                              | 1.05x (1.01-1.09x)             |
| Shear                    | 17 ± 0                                       | **18 ± 0**                              | 1.05x (1.01-1.08x)             |
| ShiftScaleRotate         | **18 ± 0**                                   | 17 ± 0                                  | 0.98x (0.94-1.03x)             |
| ShotNoise                | **1 ± 0**                                    | 1 ± 0                                   | 1.00x (0.98-1.02x)             |
| SmallestMaxSize          | 13 ± 0                                       | **14 ± 0**                              | 1.07x (1.01-1.13x)             |
| Snow                     | **14 ± 0**                                   | 10 ± 0                                  | 0.72x (0.68-0.77x)             |
| Solarize                 | 64 ± 1                                       | **88 ± 3**                              | 1.37x (1.32-1.43x)             |
| Spatter                  | **2 ± 0**                                    | 2 ± 0                                   | 0.86x (0.84-0.87x)             |
| ThinPlateSpline          | **1 ± 0**                                    | 1 ± 0                                   | 0.87x (0.85-0.89x)             |
| ToSepia                  | **71 ± 1**                                   | 67 ± 3                                  | 0.94x (0.88-0.99x)             |
| Transpose                | 21 ± 1                                       | **27 ± 1**                              | 1.29x (1.22-1.36x)             |
| UnsharpMask              | 3 ± 0                                        | **6 ± 0**                               | 1.83x (1.75-1.91x)             |
| VerticalFlip             | 80 ± 1                                       | **90 ± 4**                              | 1.12x (1.06-1.19x)             |
| ZoomBlur                 | **4 ± 0**                                    | 4 ± 0                                   | 0.98x (0.95-1.01x)             |

<!-- ALBX_VS_ALB_VIDEO_TABLE_END -->

## Requirements

The benchmark automatically creates isolated virtual environments for each library and installs the necessary dependencies. Base requirements:

- Python 3.10+
- uv (for fast package installation)
- Disk space for virtual environments
- Image/video dataset in a supported format

## Supported Libraries

- [AlbumentationsX](https://albumentations.ai/) (commercial/AGPL)
- [Albumentations](https://pypi.org/project/albumentations/2.0.8/) 2.0.8 (MIT)
- [torchvision](https://docs.pytorch.org/vision/stable/index.html)
- [Kornia](https://kornia.readthedocs.io/en/latest/)

Each library's specific dependencies are managed through separate requirements files in the `requirements/` directory.

## Setup

### Getting Started

For testing and comparison purposes, you can use standard datasets:

**For image benchmarks:**
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xf ILSVRC2012_img_val.tar -C /path/to/your/target/directory
```

**For video benchmarks:**
```bash
# UCF101 dataset
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar -d /path/to/your/target/directory
```

### Using Your Own Data

We strongly recommend running the benchmarks on your own dataset that matches your use case:

- Use images/videos that are representative of your actual workload
- Consider sizes and formats you typically work with
- Include edge cases specific to your application

This will give you more relevant performance metrics for your specific use case.

## Running Benchmarks

All benchmarks use the unified CLI: `python -m benchmark.cli run`. Use `--media` for image vs video, `--multichannel` for 9-channel image benchmarks, and `--libraries` to restrict to one or more libraries.

### Google Cloud (detached)

Run benchmarks on a **Compute Engine** VM that starts from your laptop, then keeps going after you disconnect. The default path is **detached**: the CLI uploads the repo and a job definition to **GCS**, creates a VM whose **startup script** stages data from a `gs://` dataset prefix to **local disk** (benchmarks do not read from a mounted bucket), runs `python -m benchmark.cli run` with the same flags you would use locally (including `--spec`, `--multichannel`, warmup options, etc.), uploads **results**, **vm.log**, **exit_code.txt**, and **run_meta.json** under a unique prefix, and **deletes the VM** when finished (unless you pass `--gcp-keep-instance`).

**Prerequisites**

- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud`) authenticated for your project.
- VM boot image must provide **Python 3.13+** (the package matches `requires-python` in `pytorch-latest-*` images only if that image already ships 3.13; otherwise use a custom image or install 3.13 in your startup flow—the bootstrap script fails fast with a clear error if `python3` is too old).
- A GCS bucket (or two) with:
  - A **dataset prefix** your VM can read, e.g. `gs://my-bucket/datasets/ucf101-subset/` (only the files under that prefix are rsynced to the VM).
  - A **results base URI** where each run is written, e.g. `gs://my-bucket/benchmark-runs`.
- The default Compute Engine service account (or the one attached to the VM) needs **read** access to the data prefix and **read/write** to the results bucket. For the VM to **delete itself** after the run, that service account also needs permission to call **compute.instances.delete** on its own instance (e.g. `roles/compute.instanceAdmin.v1` on a dedicated benchmark project—tighten IAM for production).

**Submit a detached run**

`--data-dir` and `--output` are still required by the parser; for detached mode they are only used locally to write `gcp_last_run.json` (and as a hint path for copying results). Point the real dataset at GCS:

```bash
python -m benchmark.cli run \
  --cloud gcp \
  --gcp-project my-gcp-project \
  --gcp-zone us-central1-a \
  --gcp-machine-type n1-standard-8 \
  --gcp-gcs-data-uri gs://my-bucket/datasets/video-50 \
  --gcp-gcs-results-uri gs://my-bucket/benchmark-runs \
  --data-dir /tmp/unused \
  --output ./gcp_runs \
  --media video \
  --libraries albumentationsx kornia torchvision \
  --num-items 50
```

After submission, open `./gcp_runs/gcp_last_run.json` for `run_prefix`, `instance_name`, and a suggested `gcloud storage cp` command to pull `results/` when the run finishes.

**Dry run (no upload, no VM)**

```bash
python -m benchmark.cli run --cloud gcp ... --gcp-dry-run
```

**Attached / SSH mode (debug)**

Creates the VM, waits for SSH, uploads the repo, runs the benchmark in a live session, downloads results to `--output`, then deletes the VM. Requires a dataset path **on the VM** (you must stage data yourself):

```bash
python -m benchmark.cli run \
  --cloud gcp --gcp-attached \
  --gcp-project my-gcp-project \
  --gcp-remote-data-dir /data/benchmark/videos \
  --data-dir /tmp/unused \
  --output ./results \
  --media video
```

**Cost note:** GCS storage for a subset and JSON results is usually small compared to **GPU/CPU VM uptime**; the expensive mistake is leaving instances running. Detached runs terminate the VM by default after uploading artifacts.

### RGB image benchmarks (all libraries)

```bash
python -m benchmark.cli run -d /path/to/images -o /path/to/output
```

### RGB image benchmarks (single library)

```bash
python -m benchmark.cli run -d /path/to/images -o /path/to/output --libraries albumentationsx
python -m benchmark.cli run -d /path/to/images -o /path/to/output --libraries torchvision
python -m benchmark.cli run -d /path/to/images -o /path/to/output --libraries kornia
```

### Multi-channel image benchmarks (9ch, all libraries)

```bash
python -m benchmark.cli run -d /path/to/images -o /path/to/output --multichannel
```

### Multi-channel image benchmarks (9ch, single library)

```bash
python -m benchmark.cli run -d /path/to/images -o /path/to/output --multichannel --libraries albumentationsx
python -m benchmark.cli run -d /path/to/images -o /path/to/output --multichannel --libraries torchvision
python -m benchmark.cli run -d /path/to/images -o /path/to/output --multichannel --libraries kornia
```

### Video benchmarks (all libraries)

```bash
python -m benchmark.cli run -d /path/to/videos -o /path/to/output --media video
```

### Video benchmarks (single library)

```bash
python -m benchmark.cli run -d /path/to/videos -o /path/to/output --media video --libraries albumentationsx
python -m benchmark.cli run -d /path/to/videos -o /path/to/output --media video --libraries torchvision
python -m benchmark.cli run -d /path/to/videos -o /path/to/output --media video --libraries kornia
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

Then run:

```bash
python -m benchmark.cli run -d /path/to/videos -o output/ --media video --spec my_transforms.py
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

## Methodology

The benchmark methodology is designed to ensure fair and reproducible comparisons:

1. **Data Loading**: Data is loaded using library-specific loaders to ensure optimal format compatibility
2. **Warmup Phase**: Adaptive warmup until performance variance stabilizes
3. **Measurement Phase**: Multiple runs with statistical analysis
4. **Environment Control**: Consistent thread settings and hardware utilization

## Contributing

Contributions are welcome! If you'd like to add support for a new library, improve the benchmarking methodology, or fix issues, please submit a pull request.

When contributing, please:
1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass

<!-- GitAds-Verify: ROVYUM6GM9I4GUYXL61ND2O2ZT2SVPGP -->
