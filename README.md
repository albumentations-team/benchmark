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
  - [Performance Highlights](#performance-highlights)
    - [Image Augmentation Performance](#image-augmentation-performance)
    - [Video Augmentation Performance](#video-augmentation-performance)
  - [Requirements](#requirements)
  - [Supported Libraries](#supported-libraries)
  - [Setup](#setup)
    - [Getting Started](#getting-started)
    - [Using Your Own Data](#using-your-own-data)
  - [Running Benchmarks](#running-benchmarks)
    - [Image Benchmarks](#running-image-benchmarks)
    - [Video Benchmarks](#running-video-benchmarks)
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
| Transform            | albumentationsx 2.1.1 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx/fastest other)   |
|:---------------------|:--------------------------------|:-----------------------|:-----------------------------|:-------------------------------|
| Affine               | **1423 ± 7**                    | -                      | 264 ± 16                     | 5.38x                          |
| AutoContrast         | **1670 ± 15**                   | 576 ± 18               | 178 ± 2                      | 2.90x                          |
| Blur                 | **7402 ± 234**                  | 365 ± 8                | -                            | 20.28x                         |
| Brightness           | **9963 ± 339**                  | 2276 ± 169             | 1681 ± 21                    | 4.38x                          |
| CLAHE                | **652 ± 2**                     | 109 ± 2                | -                            | 5.98x                          |
| CenterCrop128        | 65127 ± 1358                    | -                      | **203348 ± 7429**            | 0.32x                          |
| ChannelDropout       | **11932 ± 441**                 | 3065 ± 179             | -                            | 3.89x                          |
| ChannelShuffle       | **7779 ± 52**                   | 1446 ± 115             | 4290 ± 303                   | 1.81x                          |
| ColorJitter          | **1199 ± 25**                   | 100 ± 3                | 88 ± 3                       | 12.00x                         |
| Contrast             | **9695 ± 276**                  | 2159 ± 193             | 870 ± 26                     | 4.49x                          |
| CornerIllumination   | **500 ± 3**                     | 350 ± 4                | -                            | 1.43x                          |
| Equalize             | **1273 ± 6**                    | 310 ± 17               | 588 ± 17                     | 2.16x                          |
| Erasing              | **26700 ± 3002**                | 776 ± 45               | 10421 ± 629                  | 2.56x                          |
| GaussianBlur         | **2344 ± 46**                   | 353 ± 13               | 124 ± 17                     | 6.65x                          |
| GaussianIllumination | **765 ± 11**                    | 428 ± 16               | -                            | 1.79x                          |
| GaussianNoise        | **319 ± 8**                     | 121 ± 2                | -                            | 2.62x                          |
| Grayscale            | **17827 ± 1101**                | 1574 ± 77              | 2206 ± 179                   | 8.08x                          |
| HorizontalFlip       | **12908 ± 422**                 | 1128 ± 42              | 2234 ± 27                    | 5.78x                          |
| Hue                  | **2001 ± 9**                    | 123 ± 7                | -                            | 16.23x                         |
| Invert               | **33252 ± 3185**                | 4412 ± 293             | 22891 ± 2484                 | 1.45x                          |
| JpegCompression      | **1353 ± 1**                    | 117 ± 5                | 826 ± 11                     | 1.64x                          |
| LinearIllumination   | 568 ± 5                         | **849 ± 22**           | -                            | 0.67x                          |
| LongestMaxSize       | **3908 ± 47**                   | 481 ± 36               | -                            | 8.13x                          |
| MotionBlur           | **4265 ± 48**                   | 117 ± 6                | -                            | 36.52x                         |
| Normalize            | **1621 ± 10**                   | 1173 ± 39              | 947 ± 33                     | 1.38x                          |
| OpticalDistortion    | **774 ± 13**                    | 193 ± 4                | -                            | 4.01x                          |
| Pad                  | **38630 ± 673**                 | -                      | 4480 ± 129                   | 8.62x                          |
| Perspective          | **1167 ± 14**                   | 170 ± 5                | 217 ± 8                      | 5.37x                          |
| PhotoMetricDistort   | **1070 ± 8**                    | -                      | 80 ± 3                       | 13.33x                         |
| PlankianJitter       | **3215 ± 25**                   | 1578 ± 100             | -                            | 2.04x                          |
| PlasmaBrightness     | **186 ± 1**                     | 76 ± 2                 | -                            | 2.45x                          |
| PlasmaContrast       | **134 ± 4**                     | 75 ± 6                 | -                            | 1.77x                          |
| PlasmaShadow         | 205 ± 7                         | **211 ± 5**            | -                            | 0.97x                          |
| Posterize            | 13288 ± 667                     | 709 ± 27               | **17723 ± 1380**             | 0.75x                          |
| RGBShift             | **4736 ± 123**                  | 1787 ± 71              | -                            | 2.65x                          |
| Rain                 | **2174 ± 37**                   | 1591 ± 61              | -                            | 1.37x                          |
| RandomCrop128        | 64659 ± 1175                    | 2802 ± 40              | **112838 ± 2384**            | 0.57x                          |
| RandomGamma          | **14281 ± 1142**                | 226 ± 5                | -                            | 63.06x                         |
| RandomResizedCrop    | **4226 ± 17**                   | 579 ± 6                | 789 ± 27                     | 5.35x                          |
| Resize               | **3477 ± 30**                   | 648 ± 15               | 271 ± 4                      | 5.36x                          |
| Rotate               | **2914 ± 21**                   | 330 ± 7                | 319 ± 8                      | 8.82x                          |
| SaltAndPepper        | **599 ± 15**                    | 450 ± 5                | -                            | 1.33x                          |
| Saturation           | **1307 ± 38**                   | 132 ± 4                | -                            | 9.93x                          |
| Sharpen              | **2226 ± 44**                   | 263 ± 14               | 274 ± 9                      | 8.11x                          |
| Shear                | **1266 ± 10**                   | 358 ± 11               | -                            | 3.53x                          |
| SmallestMaxSize      | **2728 ± 24**                   | 375 ± 10               | -                            | 7.27x                          |
| Snow                 | **707 ± 11**                    | 129 ± 4                | -                            | 5.48x                          |
| Solarize             | **13474 ± 919**                 | 262 ± 3                | 1117 ± 35                    | 12.06x                         |
| ThinPlateSpline      | **84 ± 5**                      | 61 ± 2                 | -                            | 1.37x                          |
| VerticalFlip         | 26665 ± 1851                    | 2387 ± 58              | **26928 ± 4799**             | 0.99x                          |
<!-- IMAGE_BENCHMARK_TABLE_END -->

### Multi-Channel Image Benchmarks (9ch)

Benchmarks on 9-channel images (3x stacked RGB) to test OpenCV chunking and library support for >4 channels.

<!-- MULTICHANNEL_BENCHMARK_TABLE_START -->
| Transform            | albumentationsx 2.1.1 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx/fastest other)   |
|:---------------------|:--------------------------------|:-----------------------|:-----------------------------|:-------------------------------|
| Affine               | **880 ± 20**                    | 228 ± 3                | 143 ± 3                      | 3.86x                          |
| AutoContrast         | **662 ± 19**                    | 374 ± 3                | -                            | 1.77x                          |
| Blur                 | **3165 ± 31**                   | 186 ± 3                | -                            | 16.98x                         |
| Brightness           | **5682 ± 138**                  | 1350 ± 40              | -                            | 4.21x                          |
| CenterCrop128        | 74581 ± 12008                   | -                      | **223574 ± 5049**            | 0.33x                          |
| ChannelDropout       | **6556 ± 830**                  | 2179 ± 95              | -                            | 3.01x                          |
| ChannelShuffle       | **3117 ± 212**                  | 929 ± 25               | 1600 ± 41                    | 1.95x                          |
| Contrast             | **5576 ± 631**                  | 1346 ± 31              | -                            | 4.14x                          |
| CornerIllumination   | **275 ± 9**                     | 181 ± 3                | -                            | 1.52x                          |
| Erasing              | **10944 ± 368**                 | 426 ± 10               | 4321 ± 384                   | 2.53x                          |
| GaussianBlur         | **981 ± 19**                    | 188 ± 2                | 49 ± 6                       | 5.22x                          |
| GaussianIllumination | **296 ± 11**                    | 212 ± 15               | -                            | 1.40x                          |
| GaussianNoise        | **131 ± 7**                     | 65 ± 0                 | -                            | 2.01x                          |
| HorizontalFlip       | 3155 ± 71                       | 2286 ± 557             | **15102 ± 3640**             | 0.21x                          |
| Invert               | 9115 ± 1441                     | 2774 ± 169             | **15806 ± 3070**             | 0.58x                          |
| LinearIllumination   | 247 ± 5                         | **491 ± 12**           | -                            | 0.50x                          |
| LongestMaxSize       | **879 ± 12**                    | 376 ± 2                | -                            | 2.34x                          |
| MotionBlur           | **1978 ± 126**                  | 63 ± 1                 | -                            | 31.44x                         |
| Normalize            | 477 ± 46                        | **1402 ± 64**          | 795 ± 22                     | 0.34x                          |
| OpticalDistortion    | **653 ± 14**                    | 157 ± 4                | -                            | 4.16x                          |
| Pad                  | **9499 ± 1173**                 | -                      | 9112 ± 704                   | 1.04x                          |
| Perspective          | **788 ± 10**                    | 149 ± 1                | 129 ± 2                      | 5.30x                          |
| PlasmaBrightness     | **126 ± 2**                     | 24 ± 1                 | -                            | 5.21x                          |
| PlasmaContrast       | **89 ± 1**                      | 24 ± 1                 | -                            | 3.69x                          |
| PlasmaShadow         | 177 ± 4                         | **224 ± 2**            | -                            | 0.79x                          |
| Posterize            | 5500 ± 485                      | 317 ± 16               | **12018 ± 1989**             | 0.46x                          |
| RandomCrop128        | 62467 ± 7734                    | 2566 ± 75              | **124539 ± 2345**            | 0.50x                          |
| RandomGamma          | **6328 ± 72**                   | 83 ± 0                 | -                            | 76.70x                         |
| RandomResizedCrop    | **969 ± 109**                   | 309 ± 2                | 297 ± 3                      | 3.14x                          |
| Resize               | **802 ± 14**                    | 297 ± 3                | 194 ± 1                      | 2.70x                          |
| Rotate               | **2833 ± 141**                  | 172 ± 1                | 152 ± 10                     | 16.48x                         |
| Sharpen              | **969 ± 17**                    | 140 ± 6                | -                            | 6.92x                          |
| Shear                | **931 ± 19**                    | 250 ± 2                | 163 ± 6                      | 3.73x                          |
| SmallestMaxSize      | **585 ± 5**                     | 187 ± 3                | -                            | 3.13x                          |
| Solarize             | **5822 ± 780**                  | 339 ± 4                | 456 ± 11                     | 12.77x                         |
| ThinPlateSpline      | **97 ± 1**                      | 62 ± 0                 | -                            | 1.56x                          |
| VerticalFlip         | 11827 ± 1427                    | 2296 ± 118             | **15409 ± 890**              | 0.77x                          |
<!-- MULTICHANNEL_BENCHMARK_TABLE_END -->

### Video Benchmarks

The video benchmarks compare CPU-based processing (AlbumentationsX) with GPU-accelerated processing (Kornia) for video transformations. The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains realistic videos from 101 action categories.

<!-- VIDEO_BENCHMARK_TABLE_START -->
| Transform            | albumentationsx (video) 2.1.1 [vid/s]   | kornia (video) 0.8.0 [vid/s]   | torchvision (video) 0.21.0 [vid/s]   | Speedup (albx/fastest other)   |
|:---------------------|:----------------------------------------|:-------------------------------|:-------------------------------------|:-------------------------------|
| Affine               | 19 ± 0                                  | 21 ± 0                         | **453 ± 0**                          | 0.04x                          |
| AutoContrast         | 18 ± 0                                  | 21 ± 0                         | **578 ± 17**                         | 0.03x                          |
| Blur                 | **61 ± 1**                              | 21 ± 0                         | -                                    | 2.97x                          |
| Brightness           | 82 ± 4                                  | 22 ± 0                         | **756 ± 435**                        | 0.11x                          |
| CenterCrop128        | 387 ± 16                                | 70 ± 1                         | **1133 ± 235**                       | 0.34x                          |
| ChannelDropout       | **77 ± 0**                              | 22 ± 0                         | -                                    | 3.54x                          |
| ChannelShuffle       | 67 ± 4                                  | 20 ± 0                         | **958 ± 0**                          | 0.07x                          |
| ColorJitter          | 15 ± 0                                  | 19 ± 0                         | **69 ± 0**                           | 0.21x                          |
| Contrast             | 76 ± 1                                  | 22 ± 0                         | **547 ± 13**                         | 0.14x                          |
| CornerIllumination   | **6 ± 0**                               | 3 ± 0                          | -                                    | 2.42x                          |
| Elastic              | 7 ± 0                                   | -                              | **127 ± 1**                          | 0.05x                          |
| Equalize             | 12 ± 0                                  | 4 ± 0                          | **192 ± 1**                          | 0.06x                          |
| Erasing              | 86 ± 2                                  | -                              | **255 ± 7**                          | 0.34x                          |
| GaussianBlur         | 30 ± 1                                  | 22 ± 0                         | **543 ± 11**                         | 0.06x                          |
| GaussianIllumination | 9 ± 0                                   | **20 ± 0**                     | -                                    | 0.43x                          |
| GaussianNoise        | 4 ± 0                                   | **22 ± 0**                     | -                                    | 0.18x                          |
| Grayscale            | 87 ± 6                                  | 22 ± 0                         | **838 ± 467**                        | 0.10x                          |
| HorizontalFlip       | 73 ± 1                                  | 22 ± 0                         | **978 ± 49**                         | 0.07x                          |
| Hue                  | **20 ± 0**                              | 20 ± 0                         | -                                    | 1.00x                          |
| Invert               | 101 ± 1                                 | 22 ± 0                         | **843 ± 176**                        | 0.12x                          |
| LinearIllumination   | **7 ± 0**                               | 4 ± 0                          | -                                    | 1.68x                          |
| MedianBlur           | **20 ± 0**                              | 8 ± 0                          | -                                    | 2.33x                          |
| Normalize            | 17 ± 1                                  | 22 ± 0                         | **461 ± 0**                          | 0.04x                          |
| Pad                  | 80 ± 2                                  | -                              | **760 ± 338**                        | 0.10x                          |
| Perspective          | 16 ± 0                                  | -                              | **435 ± 0**                          | 0.04x                          |
| PlankianJitter       | **35 ± 1**                              | 11 ± 0                         | -                                    | 3.21x                          |
| PlasmaBrightness     | 2 ± 0                                   | **17 ± 0**                     | -                                    | 0.09x                          |
| PlasmaContrast       | 1 ± 0                                   | **17 ± 0**                     | -                                    | 0.07x                          |
| PlasmaShadow         | 2 ± 0                                   | **19 ± 0**                     | -                                    | 0.09x                          |
| Posterize            | 85 ± 7                                  | -                              | **631 ± 15**                         | 0.13x                          |
| RGBShift             | **47 ± 2**                              | 22 ± 0                         | -                                    | 2.10x                          |
| Rain                 | **27 ± 0**                              | 4 ± 0                          | -                                    | 7.10x                          |
| RandomCrop128        | 375 ± 11                                | 65 ± 0                         | **1133 ± 15**                        | 0.33x                          |
| RandomGamma          | **78 ± 5**                              | 22 ± 0                         | -                                    | 3.59x                          |
| RandomResizedCrop    | 20 ± 1                                  | 6 ± 0                          | **182 ± 16**                         | 0.11x                          |
| Resize               | 19 ± 0                                  | 6 ± 0                          | **140 ± 35**                         | 0.14x                          |
| Rotate               | 30 ± 0                                  | 22 ± 0                         | **534 ± 0**                          | 0.06x                          |
| SaltAndPepper        | **10 ± 0**                              | 9 ± 0                          | -                                    | 1.17x                          |
| Saturation           | 11 ± 1                                  | **37 ± 0**                     | -                                    | 0.30x                          |
| Sharpen              | 30 ± 1                                  | 18 ± 0                         | **420 ± 9**                          | 0.07x                          |
| Solarize             | 88 ± 3                                  | 21 ± 0                         | **628 ± 6**                          | 0.14x                          |
| ThinPlateSpline      | 1 ± 0                                   | **45 ± 1**                     | -                                    | 0.03x                          |
| VerticalFlip         | 90 ± 4                                  | 22 ± 0                         | **978 ± 5**                          | 0.09x                          |
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

<!-- ALBX_VS_ALB_IMAGE_TABLE_END -->

### Multi-channel Image Benchmark (9 channels)

<!-- ALBX_VS_ALB_MULTICHANNEL_TABLE_START -->

<!-- ALBX_VS_ALB_MULTICHANNEL_TABLE_END -->

### Video Benchmark

<!-- ALBX_VS_ALB_VIDEO_TABLE_START -->

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
