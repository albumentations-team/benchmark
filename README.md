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
| Transform            | albumentationsx 2.0.18 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx/fastest other)   |
|:---------------------|:---------------------------------|:-----------------------|:-----------------------------|:-------------------------------|
| Affine               | **1428 ± 2**                     | -                      | 264 ± 16                     | 5.40x                          |
| AutoContrast         | **1666 ± 15**                    | 576 ± 18               | 178 ± 2                      | 2.89x                          |
| Blur                 | **7592 ± 285**                   | 365 ± 8                | -                            | 20.80x                         |
| Brightness           | **12784 ± 1017**                 | 2276 ± 169             | 1681 ± 21                    | 5.62x                          |
| CLAHE                | **633 ± 3**                      | 109 ± 2                | -                            | 5.81x                          |
| CenterCrop128        | 115895 ± 4274                    | -                      | **203348 ± 7429**            | 0.57x                          |
| ChannelDropout       | **12420 ± 866**                  | 3065 ± 179             | -                            | 4.05x                          |
| ChannelShuffle       | **8075 ± 291**                   | 1446 ± 115             | 4290 ± 303                   | 1.88x                          |
| ColorJitter          | **1132 ± 23**                    | 100 ± 3                | 88 ± 3                       | 11.33x                         |
| Contrast             | **14165 ± 104**                  | 2159 ± 193             | 870 ± 26                     | 6.56x                          |
| CornerIllumination   | **468 ± 11**                     | 350 ± 4                | -                            | 1.34x                          |
| Equalize             | **1243 ± 6**                     | 310 ± 17               | 588 ± 17                     | 2.11x                          |
| Erasing              | **26411 ± 4926**                 | 776 ± 45               | 10421 ± 629                  | 2.53x                          |
| GaussianBlur         | **2429 ± 9**                     | 353 ± 13               | 124 ± 17                     | 6.89x                          |
| GaussianIllumination | **772 ± 17**                     | 428 ± 16               | -                            | 1.80x                          |
| GaussianNoise        | **343 ± 4**                      | 121 ± 2                | -                            | 2.82x                          |
| Grayscale            | **20430 ± 2245**                 | 1574 ± 77              | 2206 ± 179                   | 9.26x                          |
| HorizontalFlip       | **13654 ± 353**                  | 1128 ± 42              | 2234 ± 27                    | 6.11x                          |
| Hue                  | **1917 ± 31**                    | 123 ± 7                | -                            | 15.55x                         |
| Invert               | **32495 ± 6354**                 | 4412 ± 293             | 22891 ± 2484                 | 1.42x                          |
| JpegCompression      | **1321 ± 9**                     | 117 ± 5                | 826 ± 11                     | 1.60x                          |
| LinearIllumination   | 485 ± 9                          | **849 ± 22**           | -                            | 0.57x                          |
| LongestMaxSize       | **3840 ± 68**                    | 481 ± 36               | -                            | 7.99x                          |
| MotionBlur           | **4385 ± 110**                   | 117 ± 6                | -                            | 37.55x                         |
| Normalize            | **1602 ± 9**                     | 1173 ± 39              | 947 ± 33                     | 1.37x                          |
| OpticalDistortion    | **801 ± 2**                      | 193 ± 4                | -                            | 4.14x                          |
| Pad                  | **47542 ± 820**                  | -                      | 4480 ± 129                   | 10.61x                         |
| Perspective          | **1173 ± 3**                     | 170 ± 5                | 217 ± 8                      | 5.40x                          |
| PhotoMetricDistort   | **943 ± 18**                     | -                      | 80 ± 3                       | 11.74x                         |
| PlankianJitter       | **3138 ± 69**                    | 1578 ± 100             | -                            | 1.99x                          |
| PlasmaBrightness     | **170 ± 8**                      | 76 ± 2                 | -                            | 2.24x                          |
| PlasmaContrast       | **156 ± 2**                      | 75 ± 6                 | -                            | 2.07x                          |
| PlasmaShadow         | 196 ± 2                          | **211 ± 5**            | -                            | 0.93x                          |
| Posterize            | 13203 ± 680                      | 709 ± 27               | **17723 ± 1380**             | 0.74x                          |
| RGBShift             | **2252 ± 23**                    | 1787 ± 71              | -                            | 1.26x                          |
| Rain                 | **2064 ± 15**                    | 1591 ± 61              | -                            | 1.30x                          |
| RandomCrop128        | **113953 ± 2731**                | 2802 ± 40              | 112838 ± 2384                | 1.01x                          |
| RandomGamma          | **13280 ± 1279**                 | 226 ± 5                | -                            | 58.64x                         |
| RandomResizedCrop    | **4322 ± 9**                     | 579 ± 6                | 789 ± 27                     | 5.48x                          |
| Resize               | **3502 ± 52**                    | 648 ± 15               | 271 ± 4                      | 5.40x                          |
| Rotate               | **2981 ± 11**                    | 330 ± 7                | 319 ± 8                      | 9.02x                          |
| SaltAndPepper        | **613 ± 4**                      | 450 ± 5                | -                            | 1.36x                          |
| Saturation           | **1328 ± 45**                    | 132 ± 4                | -                            | 10.09x                         |
| Sharpen              | **2251 ± 15**                    | 263 ± 14               | 274 ± 9                      | 8.20x                          |
| Shear                | **1290 ± 9**                     | 358 ± 11               | -                            | 3.60x                          |
| SmallestMaxSize      | **2621 ± 31**                    | 375 ± 10               | -                            | 6.99x                          |
| Snow                 | **723 ± 5**                      | 129 ± 4                | -                            | 5.60x                          |
| Solarize             | **12811 ± 785**                  | 262 ± 3                | 1117 ± 35                    | 11.47x                         |
| ThinPlateSpline      | **89 ± 2**                       | 61 ± 2                 | -                            | 1.45x                          |
| VerticalFlip         | **31055 ± 325**                  | 2387 ± 58              | 26928 ± 4799                 | 1.15x                          |
<!-- IMAGE_BENCHMARK_TABLE_END -->

### Multi-Channel Image Benchmarks (9ch)

Benchmarks on 9-channel images (3x stacked RGB) to test OpenCV chunking and library support for >4 channels.

<!-- MULTICHANNEL_BENCHMARK_TABLE_START -->
| Transform            | albumentationsx 2.0.18 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx/fastest other)   |
|:---------------------|:---------------------------------|:-----------------------|:-----------------------------|:-------------------------------|
| Affine               | **640 ± 7**                      | 228 ± 3                | 143 ± 3                      | 2.81x                          |
| AutoContrast         | **436 ± 4**                      | 374 ± 3                | -                            | 1.17x                          |
| Blur                 | **2307 ± 34**                    | 186 ± 3                | -                            | 12.37x                         |
| Brightness           | **3746 ± 22**                    | 1350 ± 40              | -                            | 2.77x                          |
| CenterCrop128        | 48885 ± 772                      | -                      | **223574 ± 5049**            | 0.22x                          |
| ChannelDropout       | **5853 ± 152**                   | 2179 ± 95              | -                            | 2.69x                          |
| ChannelShuffle       | **2282 ± 54**                    | 929 ± 25               | 1600 ± 41                    | 1.43x                          |
| Contrast             | **3756 ± 76**                    | 1346 ± 31              | -                            | 2.79x                          |
| CornerIllumination   | **209 ± 2**                      | 181 ± 3                | -                            | 1.16x                          |
| Erasing              | **9957 ± 240**                   | 426 ± 10               | 4321 ± 384                   | 2.30x                          |
| GaussianBlur         | **757 ± 4**                      | 188 ± 2                | 49 ± 6                       | 4.03x                          |
| GaussianIllumination | **251 ± 1**                      | 212 ± 15               | -                            | 1.18x                          |
| GaussianNoise        | **96 ± 2**                       | 65 ± 0                 | -                            | 1.47x                          |
| HorizontalFlip       | 2436 ± 204                       | 2286 ± 557             | **15102 ± 3640**             | 0.16x                          |
| Invert               | 9859 ± 1220                      | 2774 ± 169             | **15806 ± 3070**             | 0.62x                          |
| LinearIllumination   | 146 ± 2                          | **491 ± 12**           | -                            | 0.30x                          |
| LongestMaxSize       | **835 ± 17**                     | 376 ± 2                | -                            | 2.22x                          |
| MotionBlur           | **1489 ± 22**                    | 63 ± 1                 | -                            | 23.66x                         |
| Normalize            | 386 ± 4                          | **1402 ± 64**          | 795 ± 22                     | 0.28x                          |
| OpticalDistortion    | **466 ± 4**                      | 157 ± 4                | -                            | 2.97x                          |
| Pad                  | 8573 ± 797                       | -                      | **9112 ± 704**               | 0.94x                          |
| Perspective          | **581 ± 7**                      | 149 ± 1                | 129 ± 2                      | 3.91x                          |
| PlasmaBrightness     | **86 ± 0**                       | 24 ± 1                 | -                            | 3.55x                          |
| PlasmaContrast       | **69 ± 1**                       | 24 ± 1                 | -                            | 2.85x                          |
| PlasmaShadow         | 127 ± 1                          | **224 ± 2**            | -                            | 0.57x                          |
| Posterize            | 4088 ± 111                       | 317 ± 16               | **12018 ± 1989**             | 0.34x                          |
| RandomCrop128        | 47928 ± 912                      | 2566 ± 75              | **124539 ± 2345**            | 0.38x                          |
| RandomGamma          | **4161 ± 170**                   | 83 ± 0                 | -                            | 50.43x                         |
| RandomResizedCrop    | **970 ± 8**                      | 309 ± 2                | 297 ± 3                      | 3.14x                          |
| Resize               | **744 ± 6**                      | 297 ± 3                | 194 ± 1                      | 2.51x                          |
| Rotate               | **1729 ± 53**                    | 172 ± 1                | 152 ± 10                     | 10.06x                         |
| Sharpen              | **723 ± 3**                      | 140 ± 6                | -                            | 5.16x                          |
| Shear                | **658 ± 6**                      | 250 ± 2                | 163 ± 6                      | 2.63x                          |
| SmallestMaxSize      | **583 ± 7**                      | 187 ± 3                | -                            | 3.12x                          |
| Solarize             | **4048 ± 129**                   | 339 ± 4                | 456 ± 11                     | 8.88x                          |
| ThinPlateSpline      | **79 ± 2**                       | 62 ± 0                 | -                            | 1.27x                          |
| VerticalFlip         | 8577 ± 77                        | 2296 ± 118             | **15409 ± 890**              | 0.56x                          |
<!-- MULTICHANNEL_BENCHMARK_TABLE_END -->

### Video Benchmarks

The video benchmarks compare CPU-based processing (AlbumentationsX) with GPU-accelerated processing (Kornia) for video transformations. The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains realistic videos from 101 action categories.

<!-- VIDEO_BENCHMARK_TABLE_START -->
| Transform            | albumentationsx (video) 2.0.18 [vid/s]   | kornia (video) 0.8.0 [vid/s]   | torchvision (video) 0.21.0 [vid/s]   | Speedup (albx/fastest other)   |
|:---------------------|:-----------------------------------------|:-------------------------------|:-------------------------------------|:-------------------------------|
| Affine               | 16 ± 1                                   | 21 ± 0                         | **453 ± 0**                          | 0.04x                          |
| AutoContrast         | 16 ± 0                                   | 21 ± 0                         | **578 ± 17**                         | 0.03x                          |
| Blur                 | **49 ± 2**                               | 21 ± 0                         | -                                    | 2.37x                          |
| Brightness           | 55 ± 3                                   | 22 ± 0                         | **756 ± 435**                        | 0.07x                          |
| CenterCrop128        | 586 ± 8                                  | 70 ± 1                         | **1133 ± 235**                       | 0.52x                          |
| ChannelDropout       | **63 ± 3**                               | 22 ± 0                         | -                                    | 2.89x                          |
| ChannelShuffle       | 56 ± 2                                   | 20 ± 0                         | **958 ± 0**                          | 0.06x                          |
| ColorJitter          | 13 ± 0                                   | 19 ± 0                         | **69 ± 0**                           | 0.18x                          |
| Contrast             | 55 ± 4                                   | 22 ± 0                         | **547 ± 13**                         | 0.10x                          |
| CornerIllumination   | **5 ± 0**                                | 3 ± 0                          | -                                    | 1.88x                          |
| Elastic              | 6 ± 0                                    | -                              | **127 ± 1**                          | 0.05x                          |
| Equalize             | 11 ± 0                                   | 4 ± 0                          | **192 ± 1**                          | 0.06x                          |
| Erasing              | 70 ± 4                                   | -                              | **255 ± 7**                          | 0.27x                          |
| GaussianBlur         | 28 ± 0                                   | 22 ± 0                         | **543 ± 11**                         | 0.05x                          |
| GaussianIllumination | 6 ± 0                                    | **20 ± 0**                     | -                                    | 0.30x                          |
| GaussianNoise        | 4 ± 0                                    | **22 ± 0**                     | -                                    | 0.16x                          |
| Grayscale            | 73 ± 14                                  | 22 ± 0                         | **838 ± 467**                        | 0.09x                          |
| HorizontalFlip       | 55 ± 1                                   | 22 ± 0                         | **978 ± 49**                         | 0.06x                          |
| Hue                  | 12 ± 1                                   | **20 ± 0**                     | -                                    | 0.63x                          |
| Invert               | 82 ± 4                                   | 22 ± 0                         | **843 ± 176**                        | 0.10x                          |
| LinearIllumination   | **5 ± 0**                                | 4 ± 0                          | -                                    | 1.17x                          |
| MedianBlur           | **18 ± 0**                               | 8 ± 0                          | -                                    | 2.19x                          |
| Normalize            | 15 ± 1                                   | 22 ± 0                         | **461 ± 0**                          | 0.03x                          |
| Pad                  | 59 ± 3                                   | -                              | **760 ± 338**                        | 0.08x                          |
| Perspective          | 15 ± 1                                   | -                              | **435 ± 0**                          | 0.04x                          |
| PlankianJitter       | **26 ± 1**                               | 11 ± 0                         | -                                    | 2.37x                          |
| PlasmaBrightness     | 1 ± 0                                    | **17 ± 0**                     | -                                    | 0.07x                          |
| PlasmaContrast       | 1 ± 0                                    | **17 ± 0**                     | -                                    | 0.06x                          |
| PlasmaShadow         | 2 ± 0                                    | **19 ± 0**                     | -                                    | 0.08x                          |
| Posterize            | 69 ± 12                                  | -                              | **631 ± 15**                         | 0.11x                          |
| RGBShift             | **25 ± 1**                               | 22 ± 0                         | -                                    | 1.14x                          |
| Rain                 | **25 ± 1**                               | 4 ± 0                          | -                                    | 6.64x                          |
| RandomCrop128        | 529 ± 15                                 | 65 ± 0                         | **1133 ± 15**                        | 0.47x                          |
| RandomGamma          | **66 ± 3**                               | 22 ± 0                         | -                                    | 3.05x                          |
| RandomResizedCrop    | 15 ± 0                                   | 6 ± 0                          | **182 ± 16**                         | 0.08x                          |
| Resize               | 14 ± 1                                   | 6 ± 0                          | **140 ± 35**                         | 0.10x                          |
| Rotate               | 23 ± 1                                   | 22 ± 0                         | **534 ± 0**                          | 0.04x                          |
| SaltAndPepper        | **10 ± 0**                               | 9 ± 0                          | -                                    | 1.08x                          |
| Saturation           | 10 ± 1                                   | **37 ± 0**                     | -                                    | 0.27x                          |
| Sharpen              | 26 ± 0                                   | 18 ± 0                         | **420 ± 9**                          | 0.06x                          |
| Solarize             | 60 ± 1                                   | 21 ± 0                         | **628 ± 6**                          | 0.10x                          |
| ThinPlateSpline      | 1 ± 0                                    | **45 ± 1**                     | -                                    | 0.03x                          |
| VerticalFlip         | 66 ± 3                                   | 22 ± 0                         | **978 ± 5**                          | 0.07x                          |
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
