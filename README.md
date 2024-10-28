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
- Runs per transform: 10
- Max warmup iterations: 1000

### Library Versions

- albumentations: 1.4.20
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.7.3
- torchvision: 0.20.0

### Performance Comparison

| Transform         | albumentations<br>1.4.20   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.7.3   | torchvision<br>0.20.0   |
|:------------------|:---------------------------|:-----------------|:------------------|:------------------|:------------------------|
| HorizontalFlip    | **8325 ± 955**             | 4807 ± 818       | 5585 ± 1146       | 390 ± 106         | 875 ± 69                |
| VerticalFlip      | **20493 ± 1134**           | 9153 ± 1291      | 10390 ± 290       | 1212 ± 402        | 3131 ± 61               |
| Rotate            | **1272 ± 12**              | 1119 ± 41        | 1054 ± 96         | 143 ± 11          | 147 ± 6                 |
| Affine            | **967 ± 3**                | -                | 802 ± 55          | 147 ± 9           | 128 ± 6                 |
| Equalize          | **961 ± 4**                | -                | 540 ± 39          | 152 ± 19          | 414 ± 64                |
| RandomCrop80      | **118946 ± 741**           | 25272 ± 1822     | 11009 ± 404       | 1510 ± 230        | 22499 ± 5532            |
| ShiftRGB          | **1873 ± 252**             | -                | 1563 ± 195        | -                 | -                       |
| Resize            | **2365 ± 153**             | 611 ± 78         | 1699 ± 105        | 232 ± 24          | 166 ± 9                 |
| RandomGamma       | **8608 ± 220**             | -                | 2315 ± 151        | 108 ± 13          | -                       |
| Grayscale         | **3050 ± 597**             | 2720 ± 932       | 1670 ± 56         | 289 ± 75          | 1626 ± 156              |
| RandomPerspective | 410 ± 20                   | -                | **537 ± 41**      | 86 ± 11           | 96 ± 4                  |
| GaussianBlur      | **1734 ± 204**             | 242 ± 4          | 1047 ± 122        | 176 ± 18          | 73 ± 6                  |
| MedianBlur        | **862 ± 30**               | -                | 814 ± 71          | 5 ± 0             | -                       |
| MotionBlur        | **2975 ± 52**              | -                | 583 ± 52          | 73 ± 2            | -                       |
| Posterize         | **5214 ± 101**             | -                | 2112 ± 281        | 430 ± 49          | 3063 ± 116              |
| JpegCompression   | **845 ± 61**               | 778 ± 5          | 413 ± 29          | 71 ± 3            | 617 ± 24                |
| GaussianNoise     | 147 ± 10                   | 67 ± 2           | **203 ± 10**      | 75 ± 1            | -                       |
| Elastic           | 171 ± 15                   | -                | **227 ± 17**      | 1 ± 0             | 2 ± 0                   |
| ColorJitter       | **536 ± 41**               | 255 ± 13         | -                 | 55 ± 18           | 45 ± 2                  |
| Brightness        | **4443 ± 84**              | 1163 ± 86        | -                 | 472 ± 101         | -                       |
| Contrast          | **4398 ± 143**             | 736 ± 79         | -                 | 425 ± 52          | -                       |
| Blur              | **4816 ± 59**              | 246 ± 3          | -                 | -                 | -                       |
| RandomResizedCrop | **2952 ± 24**              | -                | -                 | 287 ± 58          | 466 ± 30                |
| Normalize         | **1016 ± 84**              | -                | -                 | 626 ± 40          | 422 ± 64                |
| PlankianJitter    | **1844 ± 208**             | -                | -                 | 813 ± 211         | -                       |
| Clahe             | **423 ± 10**               | -                | -                 | 94 ± 9            | -                       |
| CoarseDropout     | **11288 ± 609**            | -                | -                 | 536 ± 87          | -                       |


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
