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
| Affine               | **1345 ± 39**                    | -                      | 264 ± 16                     | 5.09x                          |
| AutoContrast         | **1598 ± 19**                    | 576 ± 18               | 178 ± 2                      | 2.78x                          |
| Blur                 | **7411 ± 299**                   | 365 ± 8                | -                            | 20.31x                         |
| Brightness           | **11809 ± 1365**                 | 2276 ± 169             | 1681 ± 21                    | 5.19x                          |
| CLAHE                | **619 ± 8**                      | 109 ± 2                | -                            | 5.68x                          |
| CenterCrop128        | 107145 ± 5452                    | -                      | **203348 ± 7429**            | 0.53x                          |
| ChannelDropout       | **11815 ± 611**                  | 3065 ± 179             | -                            | 3.85x                          |
| ChannelShuffle       | **8085 ± 225**                   | 1446 ± 115             | 4290 ± 303                   | 1.88x                          |
| ColorJitter          | **1132 ± 10**                    | 100 ± 3                | 88 ± 3                       | 11.33x                         |
| Contrast             | **12680 ± 553**                  | 2159 ± 193             | 870 ± 26                     | 5.87x                          |
| CornerIllumination   | **447 ± 10**                     | 350 ± 4                | -                            | 1.28x                          |
| Equalize             | **1198 ± 11**                    | 310 ± 17               | 588 ± 17                     | 2.04x                          |
| Erasing              | **27290 ± 6206**                 | 776 ± 45               | 10421 ± 629                  | 2.62x                          |
| GaussianBlur         | **2509 ± 13**                    | 353 ± 13               | 124 ± 17                     | 7.12x                          |
| GaussianIllumination | **622 ± 54**                     | 428 ± 16               | -                            | 1.45x                          |
| GaussianNoise        | **320 ± 13**                     | 121 ± 2                | -                            | 2.64x                          |
| Grayscale            | **19884 ± 2088**                 | 1574 ± 77              | 2206 ± 179                   | 9.02x                          |
| HorizontalFlip       | **12947 ± 578**                  | 1128 ± 42              | 2234 ± 27                    | 5.79x                          |
| Hue                  | **1689 ± 58**                    | 123 ± 7                | -                            | 13.70x                         |
| Invert               | **33874 ± 5105**                 | 4412 ± 293             | 22891 ± 2484                 | 1.48x                          |
| JpegCompression      | **1277 ± 32**                    | 117 ± 5                | 826 ± 11                     | 1.55x                          |
| LinearIllumination   | 442 ± 9                          | **849 ± 22**           | -                            | 0.52x                          |
| LongestMaxSize       | **3544 ± 123**                   | 481 ± 36               | -                            | 7.38x                          |
| MotionBlur           | **4187 ± 90**                    | 117 ± 6                | -                            | 35.85x                         |
| Normalize            | **1510 ± 60**                    | 1173 ± 39              | 947 ± 33                     | 1.29x                          |
| OpticalDistortion    | **729 ± 15**                     | 193 ± 4                | -                            | 3.77x                          |
| Pad                  | **34558 ± 4740**                 | -                      | 4480 ± 129                   | 7.71x                          |
| Perspective          | **1153 ± 16**                    | 170 ± 5                | 217 ± 8                      | 5.30x                          |
| PhotoMetricDistort   | **859 ± 57**                     | -                      | 80 ± 3                       | 10.70x                         |
| PlankianJitter       | **2994 ± 124**                   | 1578 ± 100             | -                            | 1.90x                          |
| PlasmaBrightness     | **142 ± 4**                      | 76 ± 2                 | -                            | 1.87x                          |
| PlasmaContrast       | **136 ± 6**                      | 75 ± 6                 | -                            | 1.81x                          |
| PlasmaShadow         | 178 ± 7                          | **211 ± 5**            | -                            | 0.84x                          |
| Posterize            | 12776 ± 728                      | 709 ± 27               | **17723 ± 1380**             | 0.72x                          |
| RGBShift             | **2308 ± 9**                     | 1787 ± 71              | -                            | 1.29x                          |
| Rain                 | **1970 ± 82**                    | 1591 ± 61              | -                            | 1.24x                          |
| RandomCrop128        | 103591 ± 1101                    | 2802 ± 40              | **112838 ± 2384**            | 0.92x                          |
| RandomGamma          | **12351 ± 418**                  | 226 ± 5                | -                            | 54.54x                         |
| RandomResizedCrop    | **4292 ± 40**                    | 579 ± 6                | 789 ± 27                     | 5.44x                          |
| Resize               | **3365 ± 76**                    | 648 ± 15               | 271 ± 4                      | 5.19x                          |
| Rotate               | **2805 ± 119**                   | 330 ± 7                | 319 ± 8                      | 8.49x                          |
| SaltAndPepper        | **578 ± 7**                      | 450 ± 5                | -                            | 1.29x                          |
| Saturation           | **1179 ± 46**                    | 132 ± 4                | -                            | 8.95x                          |
| Sharpen              | **2205 ± 39**                    | 263 ± 14               | 274 ± 9                      | 8.04x                          |
| Shear                | **1235 ± 29**                    | 358 ± 11               | -                            | 3.45x                          |
| SmallestMaxSize      | **2447 ± 160**                   | 375 ± 10               | -                            | 6.53x                          |
| Snow                 | **667 ± 18**                     | 129 ± 4                | -                            | 5.16x                          |
| Solarize             | **11959 ± 707**                  | 262 ± 3                | 1117 ± 35                    | 10.71x                         |
| ThinPlateSpline      | **80 ± 1**                       | 61 ± 2                 | -                            | 1.31x                          |
| VerticalFlip         | **27128 ± 1197**                 | 2387 ± 58              | 26928 ± 4799                 | 1.01x                          |
<!-- IMAGE_BENCHMARK_TABLE_END -->

### Multi-Channel Image Benchmarks (9ch)

Benchmarks on 9-channel images (3x stacked RGB) to test OpenCV chunking and library support for >4 channels.

<!-- MULTICHANNEL_BENCHMARK_TABLE_START -->
| Transform            | albumentationsx 2.0.18 [img/s]   | kornia 0.8.2 [img/s]   | torchvision 0.25.0 [img/s]   | Speedup (albx/fastest other)   |
|:---------------------|:---------------------------------|:-----------------------|:-----------------------------|:-------------------------------|
| Affine               | **640 ± 6**                      | 228 ± 3                | 143 ± 3                      | 2.81x                          |
| AutoContrast         | **428 ± 8**                      | 374 ± 3                | -                            | 1.14x                          |
| Blur                 | **2418 ± 154**                   | 186 ± 3                | -                            | 12.97x                         |
| Brightness           | **3650 ± 162**                   | 1350 ± 40              | -                            | 2.70x                          |
| CenterCrop128        | 51247 ± 435                      | -                      | **223574 ± 5049**            | 0.23x                          |
| ChannelDropout       | **6387 ± 176**                   | 2179 ± 95              | -                            | 2.93x                          |
| ChannelShuffle       | **2427 ± 164**                   | 929 ± 25               | 1600 ± 41                    | 1.52x                          |
| Contrast             | **3837 ± 368**                   | 1346 ± 31              | -                            | 2.85x                          |
| CornerIllumination   | **203 ± 2**                      | 181 ± 3                | -                            | 1.12x                          |
| Erasing              | **9956 ± 359**                   | 426 ± 10               | 4321 ± 384                   | 2.30x                          |
| GaussianBlur         | **746 ± 9**                      | 188 ± 2                | 49 ± 6                       | 3.97x                          |
| GaussianIllumination | **240 ± 2**                      | 212 ± 15               | -                            | 1.13x                          |
| GaussianNoise        | **95 ± 3**                       | 65 ± 0                 | -                            | 1.45x                          |
| HorizontalFlip       | 2531 ± 234                       | 2286 ± 557             | **15102 ± 3640**             | 0.17x                          |
| Invert               | 10141 ± 2159                     | 2774 ± 169             | **15806 ± 3070**             | 0.64x                          |
| LinearIllumination   | 140 ± 1                          | **491 ± 12**           | -                            | 0.29x                          |
| LongestMaxSize       | 350 ± 1                          | **376 ± 2**            | -                            | 0.93x                          |
| MotionBlur           | **1443 ± 56**                    | 63 ± 1                 | -                            | 22.95x                         |
| Normalize            | 387 ± 5                          | **1402 ± 64**          | 795 ± 22                     | 0.28x                          |
| OpticalDistortion    | **457 ± 8**                      | 157 ± 4                | -                            | 2.91x                          |
| Pad                  | 8358 ± 319                       | -                      | **9112 ± 704**               | 0.92x                          |
| Perspective          | **579 ± 2**                      | 149 ± 1                | 129 ± 2                      | 3.90x                          |
| PlasmaBrightness     | **83 ± 1**                       | 24 ± 1                 | -                            | 3.44x                          |
| PlasmaContrast       | **68 ± 0**                       | 24 ± 1                 | -                            | 2.82x                          |
| PlasmaShadow         | 125 ± 1                          | **224 ± 2**            | -                            | 0.56x                          |
| Posterize            | 4375 ± 187                       | 317 ± 16               | **12018 ± 1989**             | 0.36x                          |
| RandomCrop128        | 49313 ± 715                      | 2566 ± 75              | **124539 ± 2345**            | 0.40x                          |
| RandomGamma          | **4383 ± 62**                    | 83 ± 0                 | -                            | 53.12x                         |
| RandomResizedCrop    | **369 ± 12**                     | 309 ± 2                | 297 ± 3                      | 1.19x                          |
| Resize               | 287 ± 4                          | **297 ± 3**            | 194 ± 1                      | 0.97x                          |
| Rotate               | **1713 ± 27**                    | 172 ± 1                | 152 ± 10                     | 9.96x                          |
| Sharpen              | **713 ± 9**                      | 140 ± 6                | -                            | 5.09x                          |
| Shear                | **642 ± 11**                     | 250 ± 2                | 163 ± 6                      | 2.57x                          |
| SmallestMaxSize      | **253 ± 4**                      | 187 ± 3                | -                            | 1.35x                          |
| Solarize             | **4220 ± 26**                    | 339 ± 4                | 456 ± 11                     | 9.26x                          |
| ThinPlateSpline      | **73 ± 2**                       | 62 ± 0                 | -                            | 1.17x                          |
| VerticalFlip         | 8698 ± 122                       | 2296 ± 118             | **15409 ± 890**              | 0.56x                          |
<!-- MULTICHANNEL_BENCHMARK_TABLE_END -->

### Video Benchmarks

The video benchmarks compare CPU-based processing (AlbumentationsX) with GPU-accelerated processing (Kornia) for video transformations. The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains realistic videos from 101 action categories.

<!-- VIDEO_BENCHMARK_TABLE_START -->
| Transform            | albumentationsx (video) 2.0.18 [vid/s]   | kornia (video) 0.8.0 [vid/s]   | torchvision (video) 0.21.0 [vid/s]   | Speedup (albx/fastest other)   |
|:---------------------|:-----------------------------------------|:-------------------------------|:-------------------------------------|:-------------------------------|
| Affine               | 18 ± 0                                   | 21 ± 0                         | **453 ± 0**                          | 0.04x                          |
| AutoContrast         | 17 ± 0                                   | 21 ± 0                         | **578 ± 17**                         | 0.03x                          |
| Blur                 | **60 ± 2**                               | 21 ± 0                         | -                                    | 2.92x                          |
| Brightness           | 70 ± 2                                   | 22 ± 0                         | **756 ± 435**                        | 0.09x                          |
| CenterCrop128        | 341 ± 25                                 | 70 ± 1                         | **1133 ± 235**                       | 0.30x                          |
| ChannelDropout       | **66 ± 2**                               | 22 ± 0                         | -                                    | 3.03x                          |
| ChannelShuffle       | 52 ± 4                                   | 20 ± 0                         | **958 ± 0**                          | 0.05x                          |
| ColorJitter          | 12 ± 1                                   | 19 ± 0                         | **69 ± 0**                           | 0.17x                          |
| Contrast             | 72 ± 4                                   | 22 ± 0                         | **547 ± 13**                         | 0.13x                          |
| CornerIllumination   | **6 ± 0**                                | 3 ± 0                          | -                                    | 2.20x                          |
| Elastic              | 6 ± 0                                    | -                              | **127 ± 1**                          | 0.05x                          |
| Equalize             | 12 ± 0                                   | 4 ± 0                          | **192 ± 1**                          | 0.06x                          |
| Erasing              | 85 ± 1                                   | -                              | **255 ± 7**                          | 0.33x                          |
| GaussianBlur         | 29 ± 1                                   | 22 ± 0                         | **543 ± 11**                         | 0.05x                          |
| GaussianIllumination | 7 ± 1                                    | **20 ± 0**                     | -                                    | 0.36x                          |
| GaussianNoise        | 4 ± 0                                    | **22 ± 0**                     | -                                    | 0.16x                          |
| Grayscale            | 69 ± 6                                   | 22 ± 0                         | **838 ± 467**                        | 0.08x                          |
| HorizontalFlip       | 68 ± 3                                   | 22 ± 0                         | **978 ± 49**                         | 0.07x                          |
| Hue                  | 16 ± 2                                   | **20 ± 0**                     | -                                    | 0.82x                          |
| Invert               | 92 ± 5                                   | 22 ± 0                         | **843 ± 176**                        | 0.11x                          |
| LinearIllumination   | **5 ± 0**                                | 4 ± 0                          | -                                    | 1.28x                          |
| MedianBlur           | **18 ± 0**                               | 8 ± 0                          | -                                    | 2.19x                          |
| Normalize            | 16 ± 0                                   | 22 ± 0                         | **461 ± 0**                          | 0.03x                          |
| Pad                  | 77 ± 4                                   | -                              | **760 ± 338**                        | 0.10x                          |
| Perspective          | 16 ± 0                                   | -                              | **435 ± 0**                          | 0.04x                          |
| PlankianJitter       | **27 ± 1**                               | 11 ± 0                         | -                                    | 2.52x                          |
| PlasmaBrightness     | 2 ± 0                                    | **17 ± 0**                     | -                                    | 0.09x                          |
| PlasmaContrast       | 1 ± 0                                    | **17 ± 0**                     | -                                    | 0.08x                          |
| PlasmaShadow         | 2 ± 0                                    | **19 ± 0**                     | -                                    | 0.09x                          |
| Posterize            | 75 ± 5                                   | -                              | **631 ± 15**                         | 0.12x                          |
| RGBShift             | **27 ± 0**                               | 22 ± 0                         | -                                    | 1.22x                          |
| Rain                 | **26 ± 1**                               | 4 ± 0                          | -                                    | 7.01x                          |
| RandomCrop128        | 330 ± 24                                 | 65 ± 0                         | **1133 ± 15**                        | 0.29x                          |
| RandomGamma          | **79 ± 3**                               | 22 ± 0                         | -                                    | 3.66x                          |
| RandomResizedCrop    | 18 ± 0                                   | 6 ± 0                          | **182 ± 16**                         | 0.10x                          |
| Resize               | 17 ± 1                                   | 6 ± 0                          | **140 ± 35**                         | 0.12x                          |
| Rotate               | 29 ± 1                                   | 22 ± 0                         | **534 ± 0**                          | 0.05x                          |
| SaltAndPepper        | **10 ± 0**                               | 9 ± 0                          | -                                    | 1.09x                          |
| Saturation           | 11 ± 1                                   | **37 ± 0**                     | -                                    | 0.30x                          |
| Sharpen              | 28 ± 1                                   | 18 ± 0                         | **420 ± 9**                          | 0.07x                          |
| Solarize             | 66 ± 2                                   | 21 ± 0                         | **628 ± 6**                          | 0.11x                          |
| ThinPlateSpline      | 1 ± 0                                    | **45 ± 1**                     | -                                    | 0.03x                          |
| VerticalFlip         | 76 ± 2                                   | 22 ± 0                         | **978 ± 5**                          | 0.08x                          |
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

### Running Image Benchmarks

To benchmark a single library:

```bash
./run_single.sh -l albumentationsx -d /path/to/images -o /path/to/output
```

To run benchmarks for all supported libraries and generate a comparison:

```bash
./run_all.sh -d /path/to/images -o /path/to/output --update-docs
```

### Running Video Benchmarks

To benchmark a single library:

```bash
./run_video_single.sh -l albumentationsx -d /path/to/videos -o /path/to/output
```

To run benchmarks for all supported libraries and generate a comparison:

```bash
./run_video_all.sh -d /path/to/videos -o /path/to/output --update-docs
```

#### Using Custom Transforms

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
# Using the shell script (recommended)
./run_video_single.sh -d /path/to/videos -o output/ -s my_transforms.py

# Or Python directly
python -m benchmark.video_runner -d /path/to/videos -o output.json -s my_transforms.py
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
