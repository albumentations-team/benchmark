# Image Augmentation Library Benchmarks

A comprehensive benchmarking suite for comparing the performance of popular image augmentation libraries including [Albumentations](https://albumentations.ai/), [imgaug](https://imgaug.readthedocs.io/en/latest/), [torchvision](https://pytorch.org/vision/stable/index.html), [Kornia](https://kornia.readthedocs.io/en/latest/), and [Augly](https://github.com/facebookresearch/AugLy).

<details>
<summary>Table of Contents</summary>

- [Image Augmentation Library Benchmarks](#image-augmentation-library-benchmarks)
  - [Overview](#overview)
  - [Benchmark Results](#benchmark-results)
    - [System Information](#system-information)
    - [Benchmark Parameters](#benchmark-parameters)
    - [Library Versions](#library-versions)
  - [Performance Comparison](#performance-comparison)
  - [Requirements](#requirements)
  - [Supported Libraries](#supported-libraries)
  - [Notes](#notes)
  - [Setup](#setup)
    - [Getting Started](#getting-started)
    - [Using Your Own Images](#using-your-own-images)
  - [Methodology](#methodology)
    - [Benchmark Process](#benchmark-process)
  - [Running Benchmarks](#running-benchmarks)
    - [Single Library](#single-library)
      - [Command Line Arguments](#command-line-arguments)
    - [All Libraries](#all-libraries)
      - [All Libraries Options](#all-libraries-options)
    - [Output Structure](#output-structure)
    - [Output Format](#output-format)

</details>

## Overview

This benchmark suite measures the throughput and performance characteristics of common image augmentation operations across different libraries. It features:

- Adaptive warmup to ensure stable measurements
- Multiple runs for statistical significance
- Detailed performance metrics and system information
- Thread control settings for consistent single-threaded performance
- Support for multiple image formats and loading methods

## Benchmark Results

### System Information

- Platform: macOS-15.0.1-arm64-arm-64bit
- Processor: arm
- CPU Count: 10
- Python Version: 3.12.7

### Benchmark Parameters

- Number of images: 1000
- Runs per transform: 20
- Max warmup iterations: 1000


### Library Versions

- albumentations: 1.4.21
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.7.3
- torchvision: 0.20.1
- ultralytics: 8.3.27

## Performance Comparison

| Transform         | albumentations<br>1.4.21   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.7.3   | torchvision<br>0.20.1   | ultralytics<br>8.3.27   |
|:------------------|:---------------------------|:-----------------|:------------------|:------------------|:------------------------|:------------------------|
| HorizontalFlip    | **8622 ± 891**             | 4121 ± 1001      | 6162 ± 393        | 571 ± 84          | 861 ± 38                | 712 ± 170               |
| VerticalFlip      | 23951 ± 5013               | 7775 ± 1376      | 11663 ± 2258      | 1590 ± 100        | 3156 ± 402              | **24761 ± 2631**        |
| Rotate            | 1163 ± 84                  | 1095 ± 82        | **1224 ± 75**     | 167 ± 11          | 160 ± 11                | -                       |
| Affine            | **907 ± 49**               | -                | 890 ± 34          | 181 ± 6           | 129 ± 16                | -                       |
| Equalize          | **852 ± 90**               | -                | 610 ± 35          | 184 ± 9           | 416 ± 44                | -                       |
| RandomCrop80      | **107764 ± 3630**          | 25192 ± 5964     | 12343 ± 2013      | 1492 ± 22         | 28767 ± 858             | -                       |
| ShiftRGB          | **2351 ± 276**             | -                | 1674 ± 63         | -                 | -                       | -                       |
| Resize            | **2372 ± 156**             | 632 ± 38         | 2025 ± 74         | 332 ± 18          | 180 ± 11                | -                       |
| RandomGamma       | **9014 ± 371**             | -                | 2592 ± 143        | 128 ± 10          | -                       | -                       |
| Grayscale         | **11373 ± 923**            | 3359 ± 65        | 1849 ± 75         | 628 ± 75          | 1497 ± 317              | -                       |
| RandomPerspective | 401 ± 24                   | -                | 596 ± 38          | 98 ± 7            | 106 ± 4                 | **911 ± 57**            |
| GaussianBlur      | **1664 ± 144**             | 235 ± 18         | 1043 ± 142        | 165 ± 12          | 82 ± 3                  | -                       |
| MedianBlur        | 847 ± 36                   | -                | **849 ± 32**      | 4 ± 0             | -                       | -                       |
| MotionBlur        | **3928 ± 742**             | -                | 663 ± 36          | 75 ± 6            | -                       | -                       |
| Posterize         | **9034 ± 331**             | -                | 2400 ± 142        | 363 ± 69          | 3052 ± 380              | -                       |
| JpegCompression   | **906 ± 40**               | 754 ± 14         | 443 ± 79          | 69 ± 3            | 606 ± 42                | -                       |
| GaussianNoise     | 183 ± 5                    | 70 ± 1           | **204 ± 18**      | 65 ± 2            | -                       | -                       |
| Elastic           | 229 ± 17                   | -                | **251 ± 22**      | 1 ± 0             | 3 ± 0                   | -                       |
| Clahe             | **471 ± 18**               | -                | 422 ± 12          | 90 ± 2            | -                       | -                       |
| Brightness        | **9251 ± 709**             | 1297 ± 42        | 742 ± 39          | 519 ± 15          | 449 ± 14                | -                       |
| Contrast          | **9146 ± 1034**            | 880 ± 9          | 510 ± 9           | 476 ± 116         | 358 ± 4                 | -                       |
| CoarseDropout     | **14488 ± 2108**           | -                | 653 ± 85          | 526 ± 86          | -                       | -                       |
| Blur              | **5804 ± 305**             | 243 ± 9          | 3857 ± 385        | -                 | -                       | -                       |
| ColorJitter       | **700 ± 31**               | 252 ± 15         | -                 | 50 ± 4            | 47 ± 2                  | -                       |
| RandomResizedCrop | **2879 ± 158**             | -                | -                 | 321 ± 10          | 462 ± 47                | -                       |
| Normalize         | **1349 ± 65**              | -                | -                 | 645 ± 40          | 528 ± 20                | -                       |
| PlankianJitter    | **2155 ± 340**             | -                | -                 | 1023 ± 114        | -                       | -                       |
| HSV               | **1243 ± 44**              | -                | -                 | -                 | -                       | 991 ± 38                |

## Requirements

The benchmark automatically creates isolated virtual environments for each library and installs the necessary dependencies. Base requirements:

- Python 3.10+
- uv (for fast package installation)
- Disk space for virtual environments
- Image dataset in a supported format (JPEG, PNG)

## Supported Libraries

- [Albumentations](https://albumentations.ai/)
- [imgaug](https://imgaug.readthedocs.io/en/latest/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [Kornia](https://kornia.readthedocs.io/en/latest/)
- [Augly](https://github.com/facebookresearch/AugLy)

Each library's specific dependencies are managed through separate requirements files in the `requirements/` directory.

## Notes

- The benchmark prioritizes consistent measurement over raw speed by enforcing single-threaded execution
- Early stopping mechanisms prevent excessive time spent on slow transforms
- Variance stability checks ensure meaningful measurements
- System information and thread settings are captured to aid in reproducibility

## Setup

### Getting Started

For testing and comparison purposes, you can use the ImageNet validation set:

```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xf ILSVRC2012_img_val.tar -C /path/to/your/target/directory
```

### Using Your Own Images

While the ImageNet validation set provides a standardized benchmark, we strongly recommend running the benchmarks on your own dataset that matches your use case:

- Use images that are representative of your actual workload
- Consider image sizes and formats you typically work with
- Include edge cases specific to your application

This will give you more relevant performance metrics for your specific use case, as:

- Different image sizes can significantly impact performance
- Some transforms may perform differently on different types of images
- Your specific image characteristics might favor certain libraries over others

## Methodology

### Benchmark Process

1. **Image Loading**: Images are loaded using library-specific loaders to ensure optimal format compatibility:
   - OpenCV (BGR → RGB) for Albumentations and imgaug
   - torchvision for PyTorch-based operations
   - PIL for augly
   - Normalized tensors for Kornia

2. **Warmup Phase**:
   - Performs adaptive warmup until performance variance stabilizes
   - Uses configurable parameters for stability detection
   - Implements early stopping for slow transforms
   - Maximum time limits prevent hanging on problematic transforms

3. **Measurement Phase**:
   - Multiple runs of each transform
   - Measures throughput (images/second)
   - Calculates statistical metrics (median, standard deviation)

4. **Environment Control**:
   - Forces single-threaded execution across libraries
   - Captures detailed system information and library versions
   - Monitors thread settings for various numerical libraries

## Running Benchmarks

### Single Library

To benchmark a single library:

```bash
./benchmark/run_single.sh -l albumentations -d /path/to/images -o /path/to/output
```

#### Command Line Arguments

```bash
Usage: run_single.sh -l LIBRARY -d DATA_DIR -o OUTPUT_DIR [-n NUM_IMAGES] [-r NUM_RUNS]
[--max-warmup MAX_WARMUP] [--warmup-window WINDOW]
[--warmup-threshold THRESHOLD] [--min-warmup-windows MIN_WINDOWS]
Required arguments:
-l LIBRARY Library to benchmark (albumentations, imgaug, torchvision, kornia, augly)
-d DATA_DIR Directory containing images
-o OUTPUT_DIR Directory for output files
Optional arguments:
-n NUM_IMAGES Number of images to process (default: 1000)
-r NUM_RUNS Number of benchmark runs (default: 5)
--max-warmup Maximum warmup iterations (default: 5000)
--warmup-window Window size for variance check (default: 5)
--warmup-threshold Variance stability threshold (default: 0.05)
--min-warmup-windows Minimum windows to check (default: 3)
```

### All Libraries

To run benchmarks for all supported libraries and generate a comparison:

```bash
./run_all.sh -d /path/to/images -o /path/to/output
```

#### All Libraries Options

```bash
Usage: run_all.sh -d DATA_DIR -o OUTPUT_DIR [-n NUM_IMAGES] [-r NUM_RUNS]
[--max-warmup MAX_WARMUP] [--warmup-window WINDOW]
[--warmup-threshold THRESHOLD] [--min-warmup-windows MIN_WINDOWS]
Required arguments:
-d DATA_DIR Directory containing images
-o OUTPUT_DIR Directory for output files
Optional arguments:
-n NUM_IMAGES Number of images to process (default: 2000)
-r NUM_RUNS Number of benchmark runs (default: 5)
--max-warmup Maximum warmup iterations (default: 1000)
--warmup-window Window size for variance check (default: 5)
--warmup-threshold Variance stability threshold (default: 0.05)
--min-warmup-windows Minimum windows to check (default: 3)
```

The `run_all.sh` script will:

1. Run benchmarks for each library ([albumentations](https://albumentations.ai/), [imgaug](https://imgaug.readthedocs.io/en/latest/), [torchvision](https://pytorch.org/vision/stable/index.html), [kornia](https://kornia.readthedocs.io/en/latest/), [augly](https://github.com/facebookresearch/AugLy))
2. Save individual results as JSON files in the output directory
3. Generate a comparison CSV file combining results from all libraries

### Output Structure

```tree
output_directory/
├── albumentations_results.json
├── imgaug_results.json
├── torchvision_results.json
├── kornia_results.json
└── augly_results.json
```

When running all benchmarks, the output directory will contain:

### Output Format

The benchmark produces a JSON file containing:

```json
{
    "metadata": {
        "system_info": {
            "python_version": "...",
            "platform": "...",
            "processor": "...",
            "cpu_count": "...",
            "timestamp": "..."
        },
        "library_versions": {...},
        "thread_settings": {...},
        "benchmark_params": {...}
    },
    "results": {
        "transform_name": {
            "supported": true,
            "warmup_iterations": 100,
            "throughputs": [...],
            "median_throughput": 123.45,
            "std_throughput": 1.23,
            "times": [...],
            "mean_time": 0.123,
            "std_time": 0.001,
            "variance_stable": true,
            "early_stopped": false,
            "early_stop_reason": null
        }
        // ... results for other transforms
    }
}
```
