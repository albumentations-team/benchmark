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

| Transform                | AlbumentationsX 2.2.2 [img/s]   | kornia 0.8.2 [img/s]   | pillow 12.2.0 [img/s]   | torchvision 0.26.0 [img/s]   | Speedup (albx / fastest, +/-1sd)   |
|:-------------------------|:--------------------------------|:-----------------------|:------------------------|:-----------------------------|:-----------------------------------|
| AdditiveNoise            | **261 ± 2**                     | -                      | -                       | -                            | N/A                                |
| AdvancedBlur             | **1292 ± 33**                   | -                      | -                       | -                            | N/A                                |
| Affine                   | **1479 ± 4**                    | 451 ± 4                | 616 ± 3                 | 355 ± 1                      | 2.40x (2.38-2.42x)                 |
| AtmosphericFog           | **354 ± 7**                     | -                      | -                       | -                            | N/A                                |
| AutoContrast             | 1598 ± 23                       | 824 ± 21               | **2252 ± 7**            | 428 ± 3                      | 0.71x (0.70-0.72x)                 |
| Blur                     | **7879 ± 252**                  | 744 ± 3                | 1884 ± 2                | -                            | 4.18x (4.04-4.32x)                 |
| Brightness               | **10673 ± 562**                 | 6289 ± 65              | 1354 ± 4                | 3002 ± 38                    | 1.70x (1.59-1.81x)                 |
| CLAHE                    | **661 ± 1**                     | 193 ± 3                | -                       | -                            | 3.43x (3.38-3.49x)                 |
| CenterCrop128            | 69933 ± 230                     | 5170 ± 66              | 124355 ± 824            | **230859 ± 910**             | 0.30x (0.30-0.31x)                 |
| ChannelDropout           | **12416 ± 561**                 | 5878 ± 257             | 5717 ± 4                | -                            | 2.11x (1.93-2.31x)                 |
| ChannelShuffle           | **8185 ± 302**                  | 2293 ± 68              | 5936 ± 3                | 5182 ± 100                   | 1.38x (1.33-1.43x)                 |
| ChannelSwap              | **8256 ± 224**                  | -                      | 5929 ± 10               | -                            | 1.39x (1.35-1.43x)                 |
| ChromaticAberration      | **544 ± 3**                     | -                      | -                       | -                            | N/A                                |
| CoarseDropout            | 17841 ± 498                     | -                      | **31998 ± 299**         | -                            | 0.56x (0.54-0.58x)                 |
| ColorJiggle              | **1204 ± 23**                   | 106 ± 0                | 525 ± 1                 | 132 ± 1                      | 2.29x (2.25-2.34x)                 |
| ColorJitter              | **1211 ± 24**                   | 164 ± 4                | 461 ± 3                 | 133 ± 3                      | 2.62x (2.55-2.70x)                 |
| Colorize                 | **3896 ± 75**                   | -                      | 3774 ± 9                | -                            | 1.03x (1.01-1.05x)                 |
| ConstrainedCoarseDropout | **280 ± 5**                     | -                      | -                       | -                            | N/A                                |
| Contrast                 | **10696 ± 565**                 | 6209 ± 60              | 1062 ± 2                | 1514 ± 18                    | 1.72x (1.62-1.83x)                 |
| CopyAndPaste             | **152293 ± 1416**               | -                      | -                       | -                            | N/A                                |
| CornerIllumination       | 473 ± 4                         | **658 ± 4**            | -                       | -                            | 0.72x (0.71-0.73x)                 |
| CropAndPad               | 3098 ± 10                       | -                      | **18242 ± 105**         | -                            | 0.17x (0.17-0.17x)                 |
| Defocus                  | 136 ± 0                         | -                      | **774 ± 2**             | -                            | 0.18x (0.17-0.18x)                 |
| Dithering                | <10 img/s                       | -                      | **339 ± 1**             | -                            | N/A                                |
| Downscale                | 5860 ± 79                       | -                      | **10897 ± 33**          | -                            | 0.54x (0.53-0.55x)                 |
| Elastic                  | **427 ± 2**                     | <10 img/s              | -                       | <10 img/s                    | N/A                                |
| Emboss                   | **2894 ± 5**                    | -                      | 662 ± 1                 | -                            | 4.37x (4.36-4.38x)                 |
| EnhanceDetail            | **3768 ± 5**                    | -                      | 822 ± 2                 | -                            | 4.58x (4.57-4.60x)                 |
| EnhanceEdge              | **2368 ± 84**                   | -                      | 531 ± 1                 | -                            | 4.46x (4.29-4.62x)                 |
| Equalize                 | 1264 ± 13                       | 368 ± 2                | **2226 ± 16**           | 931 ± 44                     | 0.57x (0.56-0.58x)                 |
| Erasing                  | 22342 ± 6927                    | 944 ± 160              | **31615 ± 501**         | 15242 ± 494                  | 0.71x (0.48-0.94x)                 |
| FancyPCA                 | **107 ± 1**                     | -                      | -                       | -                            | N/A                                |
| FilmGrain                | **275 ± 4**                     | -                      | 137 ± 0                 | -                            | 2.01x (1.98-2.04x)                 |
| GaussianBlur             | **2499 ± 8**                    | 710 ± 21               | 765 ± 1                 | 342 ± 7                      | 3.27x (3.25-3.28x)                 |
| GaussianIllumination     | 719 ± 16                        | **756 ± 17**           | -                       | -                            | 0.95x (0.91-1.00x)                 |
| GaussianNoise            | **316 ± 11**                    | 133 ± 1                | 136 ± 0                 | -                            | 2.32x (2.24-2.40x)                 |
| GlassBlur                | **39 ± 0**                      | -                      | -                       | -                            | N/A                                |
| Grayscale                | 18661 ± 376                     | 2389 ± 15              | **19331 ± 59**          | 3484 ± 32                    | 0.97x (0.94-0.99x)                 |
| GridDistortion           | **1296 ± 37**                   | -                      | -                       | -                            | N/A                                |
| GridDropout              | **89 ± 5**                      | -                      | -                       | -                            | N/A                                |
| GridMask                 | **16793 ± 1135**                | -                      | -                       | -                            | N/A                                |
| HSV                      | **1050 ± 41**                   | -                      | 145 ± 0                 | -                            | 7.22x (6.93-7.51x)                 |
| Halftone                 | **30 ± 1**                      | -                      | -                       | -                            | N/A                                |
| HorizontalFlip           | 12126 ± 909                     | 1344 ± 96              | **14777 ± 49**          | 2542 ± 4                     | 0.82x (0.76-0.89x)                 |
| Hue                      | **1805 ± 54**                   | 196 ± 2                | 145 ± 0                 | -                            | 9.20x (8.82-9.60x)                 |
| ISONoise                 | **181 ± 1**                     | -                      | -                       | -                            | N/A                                |
| Invert                   | 35173 ± 2558                    | 10114 ± 229            | 5526 ± 15               | **55114 ± 380**              | 0.64x (0.59-0.69x)                 |
| JpegCompression          | **1384 ± 6**                    | 189 ± 7                | 1300 ± 2                | 907 ± 10                     | 1.07x (1.06-1.07x)                 |
| LensFlare                | **266 ± 7**                     | -                      | -                       | -                            | N/A                                |
| LinearIllumination       | 520 ± 21                        | **1478 ± 24**          | -                       | -                            | 0.35x (0.33-0.37x)                 |
| LongestMaxSize           | 3997 ± 50                       | 848 ± 22               | **8088 ± 17**           | -                            | 0.49x (0.49-0.50x)                 |
| MedianBlur               | **1594 ± 2**                    | <10 img/s              | 11 ± 0                  | -                            | 147.47x (146.96-147.99x)           |
| ModeFilter               | <10 img/s                       | -                      | **16 ± 0**              | -                            | N/A                                |
| Morphological            | **17361 ± 1482**                | -                      | 54 ± 0                  | -                            | 323.57x (294.12-353.39x)           |
| MotionBlur               | **3881 ± 14**                   | 317 ± 6                | -                       | -                            | 12.25x (11.99-12.51x)              |
| MultiplicativeNoise      | **5124 ± 49**                   | -                      | -                       | -                            | N/A                                |
| Normalize                | **1558 ± 35**                   | 1447 ± 28              | 450 ± 1                 | 1355 ± 3                     | 1.08x (1.03-1.12x)                 |
| OpticalDistortion        | **816 ± 10**                    | 243 ± 1                | -                       | -                            | 3.35x (3.30-3.41x)                 |
| Pad                      | **28177 ± 389**                 | -                      | 27214 ± 275             | 5340 ± 73                    | 1.04x (1.01-1.06x)                 |
| PadIfNeeded              | 11304 ± 5360                    | -                      | **12411 ± 101**         | -                            | 0.91x (0.48-1.35x)                 |
| Perspective              | **1192 ± 5**                    | 211 ± 1                | 477 ± 1                 | 274 ± 1                      | 2.50x (2.48-2.52x)                 |
| PhotoMetricDistort       | **1072 ± 8**                    | -                      | -                       | 129 ± 0                      | 8.33x (8.25-8.40x)                 |
| PiecewiseAffine          | **177 ± 1**                     | -                      | -                       | -                            | N/A                                |
| PixelDropout             | 427 ± 3                         | -                      | **952 ± 2**             | -                            | 0.45x (0.44-0.45x)                 |
| PixelSpread              | **598 ± 14**                    | -                      | -                       | -                            | N/A                                |
| PlankianJitter           | 2783 ± 611                      | **2798 ± 105**         | -                       | -                            | 0.99x (0.75-1.26x)                 |
| PlasmaBrightness         | **182 ± 1**                     | 112 ± 2                | -                       | -                            | 1.62x (1.59-1.66x)                 |
| PlasmaContrast           | 114 ± 9                         | **120 ± 0**            | -                       | -                            | 0.95x (0.87-1.03x)                 |
| PlasmaShadow             | 198 ± 4                         | **296 ± 2**            | -                       | -                            | 0.67x (0.65-0.68x)                 |
| Posterize                | 14839 ± 354                     | 1302 ± 27              | 5462 ± 44               | **51951 ± 330**              | 0.29x (0.28-0.29x)                 |
| RGBShift                 | **5066 ± 14**                   | 2445 ± 5               | 1537 ± 5                | -                            | 2.07x (2.06-2.08x)                 |
| Rain                     | 2113 ± 25                       | **2460 ± 23**          | -                       | -                            | 0.86x (0.84-0.88x)                 |
| RandomCrop128            | 66528 ± 732                     | 3108 ± 57              | 108216 ± 5091           | **131889 ± 981**             | 0.50x (0.50-0.51x)                 |
| RandomFog                | **9 ± 0**                       | -                      | -                       | -                            | N/A                                |
| RandomGamma              | **14442 ± 1278**                | 259 ± 6                | 4989 ± 5                | -                            | 2.89x (2.64-3.15x)                 |
| RandomGravel             | **1442 ± 3**                    | -                      | -                       | -                            | N/A                                |
| RandomGridShuffle        | **11942 ± 904**                 | -                      | -                       | -                            | N/A                                |
| RandomJigsaw             | **9940 ± 597**                  | 675 ± 1                | -                       | -                            | 14.72x (13.82-15.63x)              |
| RandomResizedCrop        | **4352 ± 38**                   | 640 ± 18               | 1282 ± 4                | 978 ± 8                      | 3.39x (3.35-3.44x)                 |
| RandomRotate90           | 2012 ± 68                       | 459 ± 2                | **16739 ± 1404**        | 3885 ± 164                   | 0.12x (0.11-0.14x)                 |
| RandomScale              | **3460 ± 25**                   | -                      | 1173 ± 6                | -                            | 2.95x (2.91-2.99x)                 |
| RandomShadow             | **553 ± 7**                     | -                      | -                       | -                            | N/A                                |
| RandomSizedCrop          | **3911 ± 38**                   | -                      | 1241 ± 11               | -                            | 3.15x (3.09-3.21x)                 |
| RandomSunFlare           | **358 ± 3**                     | -                      | -                       | -                            | N/A                                |
| RandomToneCurve          | **12668 ± 1068**                | -                      | 4784 ± 51               | -                            | 2.65x (2.40-2.90x)                 |
| RingingOvershoot         | **160 ± 2**                     | -                      | -                       | -                            | N/A                                |
| Rotate                   | 2992 ± 21                       | 410 ± 4                | **4477 ± 281**          | 469 ± 2                      | 0.67x (0.62-0.72x)                 |
| SafeRotate               | **1416 ± 5**                    | -                      | 272 ± 1                 | -                            | 5.20x (5.16-5.23x)                 |
| SaltAndPepper            | 632 ± 14                        | 575 ± 4                | **2904 ± 29**           | -                            | 0.22x (0.21-0.22x)                 |
| Saturation               | 1218 ± 85                       | 213 ± 1                | **1343 ± 6**            | -                            | 0.91x (0.84-0.97x)                 |
| Sharpen                  | **2339 ± 10**                   | 408 ± 4                | 591 ± 1                 | 575 ± 2                      | 3.95x (3.93-3.98x)                 |
| Shear                    | **1350 ± 3**                    | 486 ± 0                | 509 ± 1                 | -                            | 2.65x (2.64-2.66x)                 |
| ShiftScaleRotate         | **1415 ± 4**                    | -                      | 508 ± 2                 | -                            | 2.79x (2.77-2.81x)                 |
| ShotNoise                | **45 ± 0**                      | -                      | -                       | -                            | N/A                                |
| SmallestMaxSize          | **2719 ± 4**                    | 536 ± 4                | 962 ± 2                 | -                            | 2.83x (2.82-2.84x)                 |
| Snow                     | **769 ± 16**                    | 187 ± 0                | -                       | -                            | 4.11x (4.02-4.20x)                 |
| Solarize                 | **14480 ± 365**                 | 779 ± 11               | 5422 ± 8                | 1685 ± 3                     | 2.67x (2.60-2.74x)                 |
| Spatter                  | **114 ± 1**                     | -                      | -                       | -                            | N/A                                |
| SquareSymmetry           | 2265 ± 46                       | -                      | **13239 ± 725**         | -                            | 0.17x (0.16-0.18x)                 |
| Superpixels              | **19 ± 0**                      | -                      | -                       | -                            | N/A                                |
| ThinPlateSpline          | **84 ± 0**                      | 78 ± 0                 | -                       | -                            | 1.08x (1.07-1.09x)                 |
| ToSepia                  | **8218 ± 210**                  | -                      | 349 ± 0                 | -                            | 23.53x (22.90-24.17x)              |
| Transpose                | 1649 ± 3                        | -                      | **11516 ± 97**          | -                            | 0.14x (0.14-0.14x)                 |
| UnsharpMask              | 376 ± 10                        | -                      | **482 ± 2**             | -                            | 0.78x (0.76-0.80x)                 |
| VerticalFlip             | 23891 ± 580                     | 3107 ± 21              | 41174 ± 168             | **44325 ± 1050**             | 0.54x (0.51-0.57x)                 |
| Vignetting               | **624 ± 17**                    | -                      | 480 ± 2                 | -                            | 1.30x (1.26-1.34x)                 |
| WaterRefraction          | **129 ± 5**                     | -                      | -                       | -                            | N/A                                |
| ZoomBlur                 | **183 ± 5**                     | -                      | -                       | -                            | N/A                                |

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
