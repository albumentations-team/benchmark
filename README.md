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
| Transform            | albumentationsx 2.1.0 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx/fastest other)   |
|:---------------------|:--------------------------------|:-----------------------|:-----------------------------|:-------------------------------|
| Affine               | **1458 ± 4**                    | -                      | 264 ± 16                     | 5.51x                          |
| AutoContrast         | **1698 ± 21**                   | 576 ± 18               | 178 ± 2                      | 2.95x                          |
| Blur                 | **7575 ± 180**                  | 365 ± 8                | -                            | 20.76x                         |
| Brightness           | **13221 ± 769**                 | 2276 ± 169             | 1681 ± 21                    | 5.81x                          |
| CLAHE                | **646 ± 10**                    | 109 ± 2                | -                            | 5.93x                          |
| CenterCrop128        | 122809 ± 3985                   | -                      | **203348 ± 7429**            | 0.60x                          |
| ChannelDropout       | **11699 ± 821**                 | 3065 ± 179             | -                            | 3.82x                          |
| ChannelShuffle       | **7922 ± 231**                  | 1446 ± 115             | 4290 ± 303                   | 1.85x                          |
| ColorJitter          | **1128 ± 9**                    | 100 ± 3                | 88 ± 3                       | 11.28x                         |
| Contrast             | **13842 ± 1087**                | 2159 ± 193             | 870 ± 26                     | 6.41x                          |
| CornerIllumination   | **455 ± 5**                     | 350 ± 4                | -                            | 1.30x                          |
| Equalize             | **1291 ± 17**                   | 310 ± 17               | 588 ± 17                     | 2.19x                          |
| Erasing              | **38298 ± 4226**                | 776 ± 45               | 10421 ± 629                  | 3.67x                          |
| GaussianBlur         | **2441 ± 37**                   | 353 ± 13               | 124 ± 17                     | 6.92x                          |
| GaussianIllumination | **712 ± 20**                    | 428 ± 16               | -                            | 1.66x                          |
| GaussianNoise        | **333 ± 5**                     | 121 ± 2                | -                            | 2.74x                          |
| Grayscale            | **20176 ± 2242**                | 1574 ± 77              | 2206 ± 179                   | 9.15x                          |
| HorizontalFlip       | **13434 ± 457**                 | 1128 ± 42              | 2234 ± 27                    | 6.01x                          |
| Hue                  | **1897 ± 36**                   | 123 ± 7                | -                            | 15.38x                         |
| Invert               | **46680 ± 10088**               | 4412 ± 293             | 22891 ± 2484                 | 2.04x                          |
| JpegCompression      | **1350 ± 17**                   | 117 ± 5                | 826 ± 11                     | 1.63x                          |
| LinearIllumination   | 459 ± 9                         | **849 ± 22**           | -                            | 0.54x                          |
| LongestMaxSize       | **3920 ± 68**                   | 481 ± 36               | -                            | 8.16x                          |
| MotionBlur           | **4511 ± 147**                  | 117 ± 6                | -                            | 38.62x                         |
| Normalize            | **1594 ± 29**                   | 1173 ± 39              | 947 ± 33                     | 1.36x                          |
| OpticalDistortion    | **834 ± 4**                     | 193 ± 4                | -                            | 4.31x                          |
| Pad                  | **43750 ± 2855**                | -                      | 4480 ± 129                   | 9.77x                          |
| Perspective          | **1200 ± 6**                    | 170 ± 5                | 217 ± 8                      | 5.52x                          |
| PhotoMetricDistort   | **992 ± 11**                    | -                      | 80 ± 3                       | 12.36x                         |
| PlankianJitter       | **3158 ± 29**                   | 1578 ± 100             | -                            | 2.00x                          |
| PlasmaBrightness     | **170 ± 3**                     | 76 ± 2                 | -                            | 2.25x                          |
| PlasmaContrast       | **154 ± 1**                     | 75 ± 6                 | -                            | 2.04x                          |
| PlasmaShadow         | 202 ± 2                         | **211 ± 5**            | -                            | 0.96x                          |
| Posterize            | 13426 ± 363                     | 709 ± 27               | **17723 ± 1380**             | 0.76x                          |
| RGBShift             | **2332 ± 14**                   | 1787 ± 71              | -                            | 1.31x                          |
| Rain                 | **2178 ± 28**                   | 1591 ± 61              | -                            | 1.37x                          |
| RandomCrop128        | **117121 ± 1637**               | 2802 ± 40              | 112838 ± 2384                | 1.04x                          |
| RandomGamma          | **13533 ± 408**                 | 226 ± 5                | -                            | 59.76x                         |
| RandomResizedCrop    | **4439 ± 44**                   | 579 ± 6                | 789 ± 27                     | 5.62x                          |
| Resize               | **3571 ± 44**                   | 648 ± 15               | 271 ± 4                      | 5.51x                          |
| Rotate               | **3046 ± 20**                   | 330 ± 7                | 319 ± 8                      | 9.22x                          |
| SaltAndPepper        | **631 ± 9**                     | 450 ± 5                | -                            | 1.40x                          |
| Saturation           | **1362 ± 33**                   | 132 ± 4                | -                            | 10.34x                         |
| Sharpen              | **2305 ± 7**                    | 263 ± 14               | 274 ± 9                      | 8.40x                          |
| Shear                | **1358 ± 3**                    | 358 ± 11               | -                            | 3.79x                          |
| SmallestMaxSize      | **2616 ± 58**                   | 375 ± 10               | -                            | 6.97x                          |
| Snow                 | **754 ± 8**                     | 129 ± 4                | -                            | 5.83x                          |
| Solarize             | **12744 ± 792**                 | 262 ± 3                | 1117 ± 35                    | 11.41x                         |
| ThinPlateSpline      | **94 ± 0**                      | 61 ± 2                 | -                            | 1.53x                          |
| VerticalFlip         | **28222 ± 3488**                | 2387 ± 58              | 26928 ± 4799                 | 1.05x                          |
<!-- IMAGE_BENCHMARK_TABLE_END -->

### Multi-Channel Image Benchmarks (9ch)

Benchmarks on 9-channel images (3x stacked RGB) to test OpenCV chunking and library support for >4 channels.

<!-- MULTICHANNEL_BENCHMARK_TABLE_START -->
| Transform            | albumentationsx 2.1.0 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx/fastest other)   |
|:---------------------|:--------------------------------|:-----------------------|:-----------------------------|:-------------------------------|
| Affine               | **667 ± 3**                     | 228 ± 3                | 143 ± 3                      | 2.92x                          |
| AutoContrast         | **474 ± 5**                     | 374 ± 3                | -                            | 1.27x                          |
| Blur                 | **2531 ± 31**                   | 186 ± 3                | -                            | 13.57x                         |
| Brightness           | **4317 ± 32**                   | 1350 ± 40              | -                            | 3.20x                          |
| CenterCrop128        | 50828 ± 1979                    | -                      | **223574 ± 5049**            | 0.23x                          |
| ChannelDropout       | **6730 ± 125**                  | 2179 ± 95              | -                            | 3.09x                          |
| ChannelShuffle       | **2428 ± 68**                   | 929 ± 25               | 1600 ± 41                    | 1.52x                          |
| Contrast             | **4295 ± 87**                   | 1346 ± 31              | -                            | 3.19x                          |
| CornerIllumination   | **220 ± 2**                     | 181 ± 3                | -                            | 1.22x                          |
| Erasing              | **11536 ± 304**                 | 426 ± 10               | 4321 ± 384                   | 2.67x                          |
| GaussianBlur         | **803 ± 2**                     | 188 ± 2                | 49 ± 6                       | 4.27x                          |
| GaussianIllumination | **271 ± 3**                     | 212 ± 15               | -                            | 1.28x                          |
| GaussianNoise        | **105 ± 3**                     | 65 ± 0                 | -                            | 1.62x                          |
| HorizontalFlip       | 2556 ± 152                      | 2286 ± 557             | **15102 ± 3640**             | 0.17x                          |
| Invert               | 10568 ± 727                     | 2774 ± 169             | **15806 ± 3070**             | 0.67x                          |
| LinearIllumination   | 165 ± 2                         | **491 ± 12**           | -                            | 0.34x                          |
| LongestMaxSize       | **840 ± 10**                    | 376 ± 2                | -                            | 2.24x                          |
| MotionBlur           | **1599 ± 24**                   | 63 ± 1                 | -                            | 25.42x                         |
| Normalize            | 412 ± 16                        | **1402 ± 64**          | 795 ± 22                     | 0.29x                          |
| OpticalDistortion    | **486 ± 3**                     | 157 ± 4                | -                            | 3.10x                          |
| Pad                  | 9098 ± 75                       | -                      | **9112 ± 704**               | 1.00x                          |
| Perspective          | **600 ± 4**                     | 149 ± 1                | 129 ± 2                      | 4.04x                          |
| PlasmaBrightness     | **100 ± 1**                     | 24 ± 1                 | -                            | 4.13x                          |
| PlasmaContrast       | **79 ± 2**                      | 24 ± 1                 | -                            | 3.27x                          |
| PlasmaShadow         | 146 ± 1                         | **224 ± 2**            | -                            | 0.65x                          |
| Posterize            | 4335 ± 81                       | 317 ± 16               | **12018 ± 1989**             | 0.36x                          |
| RandomCrop128        | 48969 ± 1848                    | 2566 ± 75              | **124539 ± 2345**            | 0.39x                          |
| RandomGamma          | **4131 ± 107**                  | 83 ± 0                 | -                            | 50.07x                         |
| RandomResizedCrop    | **1017 ± 8**                    | 309 ± 2                | 297 ± 3                      | 3.29x                          |
| Resize               | **775 ± 7**                     | 297 ± 3                | 194 ± 1                      | 2.61x                          |
| Rotate               | **1764 ± 17**                   | 172 ± 1                | 152 ± 10                     | 10.26x                         |
| Sharpen              | **774 ± 1**                     | 140 ± 6                | -                            | 5.52x                          |
| Shear                | **674 ± 6**                     | 250 ± 2                | 163 ± 6                      | 2.70x                          |
| SmallestMaxSize      | **586 ± 4**                     | 187 ± 3                | -                            | 3.14x                          |
| Solarize             | **4032 ± 103**                  | 339 ± 4                | 456 ± 11                     | 8.84x                          |
| ThinPlateSpline      | **88 ± 2**                      | 62 ± 0                 | -                            | 1.41x                          |
| VerticalFlip         | 9213 ± 180                      | 2296 ± 118             | **15409 ± 890**              | 0.60x                          |
<!-- MULTICHANNEL_BENCHMARK_TABLE_END -->

### Video Benchmarks

The video benchmarks compare CPU-based processing (AlbumentationsX) with GPU-accelerated processing (Kornia) for video transformations. The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains realistic videos from 101 action categories.

<!-- VIDEO_BENCHMARK_TABLE_START -->
| Transform            | albumentationsx (video) 2.1.0 [vid/s]   | kornia (video) 0.8.0 [vid/s]   | torchvision (video) 0.21.0 [vid/s]   | Speedup (albx/fastest other)   |
|:---------------------|:----------------------------------------|:-------------------------------|:-------------------------------------|:-------------------------------|
| Affine               | 17 ± 1                                  | 21 ± 0                         | **453 ± 0**                          | 0.04x                          |
| AutoContrast         | 17 ± 0                                  | 21 ± 0                         | **578 ± 17**                         | 0.03x                          |
| Blur                 | **57 ± 4**                              | 21 ± 0                         | -                                    | 2.75x                          |
| Brightness           | 70 ± 0                                  | 22 ± 0                         | **756 ± 435**                        | 0.09x                          |
| CenterCrop128        | 517 ± 22                                | 70 ± 1                         | **1133 ± 235**                       | 0.46x                          |
| ChannelDropout       | **70 ± 0**                              | 22 ± 0                         | -                                    | 3.23x                          |
| ChannelShuffle       | 59 ± 1                                  | 20 ± 0                         | **958 ± 0**                          | 0.06x                          |
| ColorJitter          | 13 ± 0                                  | 19 ± 0                         | **69 ± 0**                           | 0.19x                          |
| Contrast             | 71 ± 1                                  | 22 ± 0                         | **547 ± 13**                         | 0.13x                          |
| CornerIllumination   | **5 ± 0**                               | 3 ± 0                          | -                                    | 2.10x                          |
| Elastic              | 7 ± 0                                   | -                              | **127 ± 1**                          | 0.06x                          |
| Equalize             | 12 ± 1                                  | 4 ± 0                          | **192 ± 1**                          | 0.06x                          |
| Erasing              | 79 ± 1                                  | -                              | **255 ± 7**                          | 0.31x                          |
| GaussianBlur         | 29 ± 1                                  | 22 ± 0                         | **543 ± 11**                         | 0.05x                          |
| GaussianIllumination | 8 ± 1                                   | **20 ± 0**                     | -                                    | 0.37x                          |
| GaussianNoise        | 4 ± 0                                   | **22 ± 0**                     | -                                    | 0.16x                          |
| Grayscale            | 78 ± 2                                  | 22 ± 0                         | **838 ± 467**                        | 0.09x                          |
| HorizontalFlip       | 54 ± 4                                  | 22 ± 0                         | **978 ± 49**                         | 0.05x                          |
| Hue                  | 16 ± 1                                  | **20 ± 0**                     | -                                    | 0.83x                          |
| Invert               | 73 ± 7                                  | 22 ± 0                         | **843 ± 176**                        | 0.09x                          |
| LinearIllumination   | **5 ± 0**                               | 4 ± 0                          | -                                    | 1.17x                          |
| MedianBlur           | **19 ± 1**                              | 8 ± 0                          | -                                    | 2.28x                          |
| Normalize            | 16 ± 1                                  | 22 ± 0                         | **461 ± 0**                          | 0.04x                          |
| Pad                  | 55 ± 4                                  | -                              | **760 ± 338**                        | 0.07x                          |
| Perspective          | 15 ± 0                                  | -                              | **435 ± 0**                          | 0.03x                          |
| PlankianJitter       | **26 ± 3**                              | 11 ± 0                         | -                                    | 2.35x                          |
| PlasmaBrightness     | 2 ± 0                                   | **17 ± 0**                     | -                                    | 0.09x                          |
| PlasmaContrast       | 1 ± 0                                   | **17 ± 0**                     | -                                    | 0.08x                          |
| PlasmaShadow         | 2 ± 0                                   | **19 ± 0**                     | -                                    | 0.08x                          |
| Posterize            | 68 ± 7                                  | -                              | **631 ± 15**                         | 0.11x                          |
| RGBShift             | **28 ± 1**                              | 22 ± 0                         | -                                    | 1.26x                          |
| Rain                 | **24 ± 1**                              | 4 ± 0                          | -                                    | 6.26x                          |
| RandomCrop128        | 605 ± 6                                 | 65 ± 0                         | **1133 ± 15**                        | 0.53x                          |
| RandomGamma          | **70 ± 1**                              | 22 ± 0                         | -                                    | 3.23x                          |
| RandomResizedCrop    | 17 ± 1                                  | 6 ± 0                          | **182 ± 16**                         | 0.09x                          |
| Resize               | 17 ± 1                                  | 6 ± 0                          | **140 ± 35**                         | 0.12x                          |
| Rotate               | 25 ± 1                                  | 22 ± 0                         | **534 ± 0**                          | 0.05x                          |
| SaltAndPepper        | 9 ± 1                                   | **9 ± 0**                      | -                                    | 0.98x                          |
| Saturation           | 9 ± 0                                   | **37 ± 0**                     | -                                    | 0.25x                          |
| Sharpen              | 29 ± 0                                  | 18 ± 0                         | **420 ± 9**                          | 0.07x                          |
| Solarize             | 67 ± 9                                  | 21 ± 0                         | **628 ± 6**                          | 0.11x                          |
| ThinPlateSpline      | 1 ± 0                                   | **45 ± 1**                     | -                                    | 0.03x                          |
| VerticalFlip         | 76 ± 4                                  | 22 ± 0                         | **978 ± 5**                          | 0.08x                          |
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

- [AlbumentationsX](https://albumentations.ai/)
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
