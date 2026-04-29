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
  - [Requirements](#requirements)
  - [Supported Libraries](#supported-libraries)
  - [Setup](#setup)
    - [Getting Started](#getting-started)
    - [Using Your Own Data](#using-your-own-data)
  - [Running Benchmarks](#running-benchmarks)
    - [Google Cloud (detached)](#google-cloud-detached)
    - [RGB Image Benchmarks](#rgb-image-benchmarks-all-libraries)
    - [Video Benchmarks](#video-benchmarks-all-libraries)
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

- **Micro / profiler benchmarks** preload decoded images and time augmentation only. These runs use one internal CPU thread for every library to measure single-stream transform cost.
- **Pipeline benchmarks** read from disk through a dataloader and measure realistic input-pipeline throughput. These runs should use production-style settings: AlbumentationsX scales through dataloader workers, while torchvision, Kornia, OpenCV, and other libraries may use their normal threading/execution model. The result metadata records worker counts and thread settings.

For paper-quality RGB image results, use `2,000` ImageNet validation images for micro benchmarks and the full `50,000` ImageNet validation set for pipeline benchmarks.

<!-- IMAGE_BENCHMARK_TABLE_START -->

| Transform            | AlbumentationsX 2.2.5 [img/s]   | kornia 0.8.2 [img/s]   | pillow 12.2.0 [img/s]   | torchvision 0.26.0 [img/s]   | Speedup (albx / fastest, +/-1sd)   |
|:---------------------|:--------------------------------|:-----------------------|:------------------------|:-----------------------------|:-----------------------------------|
| Affine               | **1456 ± 23**                   | 456 ± 2                | 613 ± 1                 | 331 ± 7                      | 2.37x (2.33-2.41x)                 |
| AutoContrast         | 1619 ± 44                       | 739 ± 14               | **2239 ± 3**            | 399 ± 2                      | 0.72x (0.70-0.74x)                 |
| Blur                 | **7544 ± 134**                  | 745 ± 2                | 1870 ± 13               | -                            | 4.03x (3.93-4.13x)                 |
| Brightness           | **9849 ± 99**                   | 2992 ± 8               | 1340 ± 6                | 1989 ± 9                     | 3.29x (3.25-3.33x)                 |
| CLAHE                | **644 ± 5**                     | 206 ± 1                | -                       | -                            | 3.13x (3.10-3.17x)                 |
| CenterCrop128        | **95346 ± 1281**                | 5226 ± 57              | -                       | 34279 ± 307                  | 2.78x (2.72-2.84x)                 |
| ChannelDropout       | **11971 ± 434**                 | 2878 ± 11              | -                       | -                            | 4.16x (3.99-4.33x)                 |
| ChannelShuffle       | **8235 ± 86**                   | 1729 ± 3               | -                       | 5383 ± 12                    | 1.53x (1.51-1.55x)                 |
| ColorJiggle          | **1208 ± 16**                   | 107 ± 0                | -                       | 133 ± 1                      | 9.07x (8.90-9.23x)                 |
| ColorJitter          | **1221 ± 10**                   | 169 ± 1                | -                       | 131 ± 2                      | 7.23x (7.12-7.35x)                 |
| Colorize             | **3858 ± 11**                   | -                      | 3697 ± 18               | -                            | 1.04x (1.04-1.05x)                 |
| Contrast             | **10045 ± 119**                 | 2983 ± 11              | 1055 ± 6                | 1228 ± 4                     | 3.37x (3.32-3.42x)                 |
| CornerIllumination   | **866 ± 28**                    | 596 ± 3                | -                       | -                            | 1.45x (1.40-1.51x)                 |
| Dithering            | ≤10 img/s                       | -                      | **1426 ± 16**           | -                            | N/A                                |
| Elastic              | **453 ± 2**                     | ≤10 img/s              | -                       | ≤10 img/s                    | N/A                                |
| Equalize             | 1086 ± 12                       | 390 ± 6                | **2204 ± 2**            | 946 ± 5                      | 0.49x (0.49-0.50x)                 |
| Erasing              | **27849 ± 4028**                | 1228 ± 8               | -                       | 4175 ± 10                    | 6.67x (5.69-7.65x)                 |
| GaussianBlur         | **2462 ± 11**                   | 717 ± 11               | 765 ± 3                 | 336 ± 4                      | 3.22x (3.19-3.24x)                 |
| GaussianIllumination | **773 ± 21**                    | 680 ± 13               | -                       | -                            | 1.14x (1.08-1.19x)                 |
| GaussianNoise        | **328 ± 20**                    | 133 ± 1                | -                       | -                            | 2.47x (2.31-2.63x)                 |
| Grayscale            | **19593 ± 350**                 | 1679 ± 16              | 19267 ± 61              | 3409 ± 8                     | 1.02x (1.00-1.04x)                 |
| HorizontalFlip       | 13200 ± 430                     | 1352 ± 8               | **14680 ± 194**         | 1678 ± 15                    | 0.90x (0.86-0.94x)                 |
| Hue                  | **1908 ± 18**                   | 204 ± 1                | -                       | -                            | 9.35x (9.23-9.46x)                 |
| Invert               | **31753 ± 1327**                | 3718 ± 23              | 5503 ± 17               | 5195 ± 84                    | 5.77x (5.51-6.03x)                 |
| JpegCompression      | **1351 ± 11**                   | 202 ± 2                | 1305 ± 5                | 925 ± 3                      | 1.04x (1.02-1.05x)                 |
| LinearIllumination   | 557 ± 18                        | **1195 ± 7**           | -                       | -                            | 0.47x (0.45-0.48x)                 |
| LongestMaxSize       | **3847 ± 62**                   | 855 ± 7                | -                       | -                            | 4.50x (4.39-4.61x)                 |
| MedianBlur           | **1546 ± 16**                   | ≤10 img/s              | 11 ± 0                  | -                            | 143.29x (140.11-146.54x)           |
| MotionBlur           | **3847 ± 49**                   | 322 ± 6                | -                       | -                            | 11.95x (11.59-12.32x)              |
| Normalize            | **1642 ± 26**                   | 1226 ± 13              | -                       | 1091 ± 6                     | 1.34x (1.30-1.37x)                 |
| OpticalDistortion    | **395 ± 4**                     | 243 ± 1                | -                       | -                            | 1.63x (1.61-1.64x)                 |
| Pad                  | **34979 ± 3274**                | -                      | 27167 ± 282             | 5072 ± 68                    | 1.29x (1.16-1.42x)                 |
| Perspective          | **1185 ± 9**                    | 214 ± 2                | -                       | 268 ± 6                      | 4.43x (4.30-4.57x)                 |
| PhotoMetricDistort   | **1070 ± 19**                   | -                      | -                       | 129 ± 0                      | 8.31x (8.14-8.47x)                 |
| PlankianJitter       | **3278 ± 13**                   | 2996 ± 26              | -                       | -                            | 1.09x (1.08-1.11x)                 |
| PlasmaBrightness     | **394 ± 9**                     | 115 ± 0                | -                       | -                            | 3.44x (3.35-3.52x)                 |
| PlasmaContrast       | **250 ± 6**                     | 117 ± 0                | -                       | -                            | 2.14x (2.09-2.19x)                 |
| PlasmaShadow         | **526 ± 8**                     | 281 ± 1                | -                       | -                            | 1.87x (1.83-1.90x)                 |
| Posterize            | **28724 ± 3259**                | 1080 ± 67              | 5429 ± 10               | 5162 ± 6                     | 5.29x (4.68-5.90x)                 |
| RGBShift             | **5025 ± 48**                   | 1710 ± 15              | -                       | -                            | 2.94x (2.89-2.99x)                 |
| Rain                 | **2169 ± 27**                   | 1725 ± 3               | -                       | -                            | 1.26x (1.24-1.28x)                 |
| RandomCrop128        | **93574 ± 1964**                | 3223 ± 17              | -                       | 31208 ± 564                  | 3.00x (2.88-3.12x)                 |
| RandomGamma          | **14482 ± 424**                 | 252 ± 4                | -                       | -                            | 57.48x (54.88-60.17x)              |
| RandomJigsaw         | **9413 ± 136**                  | 638 ± 2                | -                       | -                            | 14.74x (14.48-15.01x)              |
| RandomResizedCrop    | **4354 ± 22**                   | 653 ± 6                | -                       | 939 ± 23                     | 4.64x (4.50-4.78x)                 |
| RandomRotate90       | **8652 ± 167**                  | 464 ± 2                | -                       | -                            | 18.63x (18.19-19.08x)              |
| Resize               | **3542 ± 11**                   | 677 ± 7                | 1087 ± 9                | 288 ± 2                      | 3.26x (3.22-3.30x)                 |
| Rotate               | 2996 ± 12                       | 442 ± 3                | **4101 ± 119**          | 451 ± 9                      | 0.73x (0.71-0.76x)                 |
| SaltAndPepper        | **946 ± 4**                     | 510 ± 1                | -                       | -                            | 1.85x (1.84-1.86x)                 |
| Saturation           | **1389 ± 27**                   | 216 ± 1                | 1324 ± 6                | -                            | 1.05x (1.02-1.07x)                 |
| Sharpen              | **2221 ± 35**                   | 434 ± 6                | -                       | 525 ± 1                      | 4.23x (4.15-4.31x)                 |
| Shear                | **1322 ± 7**                    | 482 ± 4                | 502 ± 2                 | -                            | 2.64x (2.61-2.66x)                 |
| SmallestMaxSize      | **2676 ± 7**                    | 537 ± 1                | -                       | -                            | 4.99x (4.96-5.01x)                 |
| Snow                 | **754 ± 4**                     | 188 ± 0                | -                       | -                            | 4.01x (3.98-4.04x)                 |
| Solarize             | **13505 ± 442**                 | 695 ± 18               | 5403 ± 13               | 1297 ± 2                     | 2.50x (2.41-2.59x)                 |
| ThinPlateSpline      | **92 ± 1**                      | 78 ± 0                 | -                       | -                            | 1.18x (1.16-1.20x)                 |
| Transpose            | 8184 ± 199                      | -                      | **11038 ± 172**         | -                            | 0.74x (0.71-0.77x)                 |
| UnsharpMask          | **3063 ± 37**                   | -                      | 478 ± 2                 | -                            | 6.41x (6.31-6.51x)                 |
| VerticalFlip         | 29169 ± 2657                    | 3409 ± 3               | **41794 ± 189**         | 5023 ± 29                    | 0.70x (0.63-0.76x)                 |

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

For AlbumentationsX, each clip is a NumPy array `(T, H, W, C)`. The built-in spec files apply augmentations with `transform(images=video)["images"]`—Albumentations’ batch video API—so parameters are drawn once per clip and shared across frames, in line with typical video training and with Kornia’s `same_on_batch=True` for a fair comparison.

<!-- VIDEO_BENCHMARK_TABLE_START -->

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

### Using Your Own Data

We strongly recommend running the benchmarks on your own dataset that matches your use case:

- Use images/videos that are representative of your actual workload
- Consider sizes and formats you typically work with
- Include edge cases specific to your application

This will give you more relevant performance metrics for your specific use case.

## Running Benchmarks

All benchmarks use the unified CLI: `python -m benchmark.cli run`. Use `--media` for image vs video, `--multichannel` for 9-channel image benchmarks, and `--libraries` to restrict to one or more libraries.

The CLI creates joined virtual environments for compatible libraries, for example `.venv_albumentationsx` for AlbumentationsX and `.venv_torch_stack` for torchvision, Kornia, and Pillow image benchmarks. By default, each run refreshes `requirements/*.txt` from `requirements/*.in` with the latest compatible package versions, then installs dependencies only when the resolved requirement files changed. Pass `--no-refresh-requirements` for offline/debug reruns that should reuse the existing lock files and venv cache.

For RGB image paper runs, prefer scenario mode:

```bash
# Micro/profiler: one internal thread per library, preloaded images, augmentation only.
python -m benchmark.cli run \
  --scenario image-rgb \
  --mode micro \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro \
  --num-items 2000 \
  --timer pyperf

# Pipeline/user guidance: full ImageNet validation from disk, production-style threading.
python -m benchmark.cli run \
  --scenario image-rgb \
  --mode pipeline \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_pipeline \
  --batch-size 64 \
  --workers 8 \
  --min-time 30
```

Treat RGB micro results as an implementation profiler: preloaded decoded inputs, one process, one internal
library thread, augmentation only. They are useful for checking algorithmic implementation quality and regressions,
but they are intentionally artificial because they measure one CPU core instead of a production input pipeline.

The paper hardware set should focus on CPUs that resemble machines used to feed model training, not every available
cloud CPU family. For RGB micro/profiler runs, use a compact representative set:

- Apple Silicon laptop, e.g. MacBook M4, for local macOS Arm behavior.
- `c4-standard-16` for modern Intel x86.
- `c4d-standard-16` for modern AMD x86.
- `c4a-standard-16` for cloud Arm, if Arm portability is part of the claim.
- `g2-standard-16` for the host CPU used with L4 GPU training.
- `a2-highgpu-1g` for the host CPU used with A100 training.

Older/general-purpose machines such as `n2-standard-16` and `n2d-standard-16` are useful as historical baselines, but
they should not drive the main paper claims. The more important paper benchmarks are production-style DataLoader runs
for images and GPU video augmentation, especially torchvision video paths on GPU.

Skip dependency lock refresh when you intentionally want the fastest local rerun from existing locks:

```bash
python -m benchmark.cli run \
  --scenario image-rgb \
  --mode micro \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro \
  --libraries albumentationsx \
  --no-refresh-requirements
```

### Benchmark execution policy

- Cloud runs stage one compressed dataset object, such as `gs://.../val.tar`, onto the VM and unpack it locally. Do not upload or copy thousands of individual images for each run.
- Micro benchmarks preload the requested number of images or videos once per library into that library's native in-memory representation. Per-transform timing must not reread or decode media from disk.
- Pyperf micro runs isolate transform measurements in subprocesses, but those subprocesses reuse the per-library media cache and lazily construct only the transform being measured.
- Libraries with lazy or partially lazy output objects must materialize their own result inside the timed call. For Pillow, call `Image.load()` on returned `Image.Image` objects. Do not force fairness by converting outputs to NumPy arrays or computing checksums in the timed benchmark; those are diagnostic-only operations and change the library contract being measured.
- Libraries should only be listed for transforms they support directly. Do not recreate missing transforms with extensive benchmark-side helper code just to fill a table cell. For example, Pillow can benchmark direct `Image` / `ImageOps` / `ImageFilter` operations, but should skip Albumentations-style composites such as `RandomResizedCrop`, `PadIfNeeded`, `SafeRotate`, `ShiftScaleRotate`, `LongestMaxSize`, and `SmallestMaxSize`. When Pillow has a direct equivalent for an AlbumentationsX transform, keep the parameters exact; `Dithering` maps to grayscale, 2-color Floyd-Steinberg dithering, not palette quantization.
- Compatible libraries share joined environments to avoid redundant dependency setup. Image benchmarks group torchvision, Kornia, and Pillow into the `torch_stack` environment; video benchmarks group torchvision and Kornia into `torch_video`.
- Environment setup is cached by resolved requirement files, Python version, media type, and environment group. Detached GCP runs can additionally reuse the GCS venv cache unless `--gcp-no-venv-cache` or `--gcp-force-venv-cache-rebuild` is set.
- Slow transforms are preflighted before exhaustive pyperf measurement. If a transform crosses the slow threshold, record an early-stop result instead of spending the full run budget.
- Keep benchmark data local to the machine doing the timing. GCP runs should not benchmark against mounted buckets or network paths.
- Preserve single-thread micro timing for fair augmentation-only comparisons. Pipeline benchmarks can use production-style workers and thread settings, but those settings must be recorded in metadata.

### Google Cloud (detached)

Run benchmarks on a **Compute Engine** VM that starts from your laptop, then keeps going after you disconnect. The default path is **detached**: the CLI uploads the repo and a job definition to **GCS**, creates a VM whose **startup script** downloads one dataset archive/object such as `gs://.../val.tar`, unpacks it to **local disk** (benchmarks do not read from a mounted bucket), runs `python -m benchmark.cli run` with the same flags you would use locally (including `--spec`, `--multichannel`, warmup options, etc.), uploads **results**, **vm.log**, **exit_code.txt**, and **run_meta.json** under a unique prefix, and **deletes the VM** when finished (unless you pass `--gcp-keep-instance`).

**Prerequisites**

- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud`) authenticated for your project.
- VM boot image must provide **Python 3.13+** (the package matches `requires-python` in `pytorch-latest-*` images only if that image already ships 3.13; otherwise use a custom image or install 3.13 in your startup flow—the bootstrap script fails fast with a clear error if `python3` is too old).
- A GCS bucket (or two) with:
  - A **dataset archive/object** your VM can read, e.g. `gs://my-bucket/datasets/imagenet/val.tar`.
  - A **results base URI** where each run is written, e.g. `gs://my-bucket/benchmark-runs`.
- The default Compute Engine service account (or the one attached to the VM) needs **read** access to the dataset object and **read/write** to the results bucket. For the VM to **delete itself** after the run, that service account also needs permission to call **compute.instances.delete** on its own instance (e.g. `roles/compute.instanceAdmin.v1` on a dedicated benchmark project—tighten IAM for production).

**Submit a detached run**

`--data-dir` and `--output` are still required by the parser; for detached mode they are only used locally to write `gcp_last_run.json` (and as a hint path for copying results). Point the real dataset at GCS:

```bash
python -m benchmark.cli run \
  --cloud gcp \
  --gcp-project my-gcp-project \
  --gcp-zone us-central1-a \
  --gcp-machine-type n1-standard-8 \
  --gcp-gcs-data-uri gs://my-bucket/datasets/imagenet/val.tar \
  --gcp-gcs-results-uri gs://my-bucket/benchmark-runs \
  --data-dir /tmp/unused \
  --output ./gcp_runs \
  --scenario image-rgb \
  --mode micro \
  --libraries albumentationsx torchvision kornia pillow \
  --num-items 2000 \
  --timer pyperf
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

1. **Measurement scope**: Micro benchmarks measure augmentation-only cost from preloaded data. Pipeline benchmarks measure realistic dataloader throughput from disk.
2. **Threading policy**: Micro benchmarks force one internal thread for every library. Pipeline benchmarks use production-style settings and record both dataloader workers and library thread settings.
3. **Dataset size**: RGB micro paper runs use `2,000` images from the unpacked ImageNet validation tar. RGB pipeline paper runs use the full `50,000`-image ImageNet validation set.
4. **Warmup and statistics**: Runs report robust summary statistics, coefficient of variation, confidence intervals, and unstable-result flags.
5. **Environment metadata**: Results record CPU/GPU metadata, package versions, git state, timing backend, dataset fingerprint, batch size, workers, and whether decode/collate/GPU transfer are included.

## Contributing

Contributions are welcome! If you'd like to add support for a new library, improve the benchmarking methodology, or fix issues, please submit a pull request.

When contributing, please:
1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass

<!-- GitAds-Verify: ROVYUM6GM9I4GUYXL61ND2O2ZT2SVPGP -->
