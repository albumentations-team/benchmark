<!-- This file is auto-generated. Do not edit directly. -->

# Video Augmentation Benchmarks

This directory contains benchmark results for video augmentation libraries.

## Overview

The video benchmarks measure the performance of various augmentation libraries on video transformations. The benchmarks compare CPU-based processing (Albumentations) with GPU-accelerated processing (Kornia).

## Dataset

The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains 13,320 videos from 101 action categories. The videos are realistic, collected from YouTube, and include a wide variety of camera motion, object appearance, pose, scale, viewpoint, and background. This makes it an excellent dataset for benchmarking video augmentation performance across diverse real-world scenarios.

You can download the dataset from: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

## Methodology

1. **Video Loading**: Videos are loaded using library-specific loaders:
   - OpenCV for Albumentations
   - PyTorch tensors for Kornia

2. **Warmup Phase**:
   - Performs adaptive warmup until performance variance stabilizes
   - Uses configurable parameters for stability detection
   - Implements early stopping for slow transforms

3. **Measurement Phase**:
   - Multiple runs of each transform
   - Measures throughput (videos/second)
   - Calculates statistical metrics (median, standard deviation)

4. **Environment Control**:
   - CPU benchmarks are run single-threaded
   - GPU benchmarks utilize the specified GPU device
   - Thread settings are controlled for consistent results

## Hardware Comparison

The benchmarks compare:
- Albumentations: CPU-based processing (single thread)
- Kornia: GPU-accelerated processing (NVIDIA GPUs)

This provides insights into the trade-offs between CPU and GPU processing for video augmentation.

## Running the Benchmarks

To run the video benchmarks:

```bash
./run_video_single.sh -l albumentations -d /path/to/videos -o /path/to/output
```

To run all libraries and generate a comparison:

```bash
./run_video_all.sh -d /path/to/videos -o /path/to/output
```

## Benchmark Results

<!-- BENCHMARK_RESULTS_START -->
<!-- This file is auto-generated. Do not edit directly. -->

# Video Benchmark Results

Number shows how many videos per second can be processed. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

| Transform | albumentations (videos per second)<br>arm (1 core) | kornia (videos per second)<br>NVIDIA GeForce RTX 4090 | torchvision (videos per second)<br>NVIDIA GeForce RTX 4090 | Speedup<br>(Alb/fastest other) |
|---|---|---|---|---|
| Affine | 4.29 ± 0.07 | 21.39 ± 0.05 | **452.58 ± 0.14** | 0.01x |
| AutoContrast | 20.51 ± 0.08 | 21.41 ± 0.02 | **577.72 ± 16.86** | 0.04x |
| Blur | **50.11 ± 2.71** | 20.61 ± 0.06 | N/A | 2.43x |
| Brightness | 177.33 ± 7.33 | 21.85 ± 0.02 | **755.52 ± 435.17** | 0.23x |
| CLAHE | **8.79 ± 0.09** | N/A | N/A | N/A |
| CenterCrop128 | 704.96 ± 15.03 | 70.12 ± 1.29 | **1133.39 ± 234.60** | 0.62x |
| ChannelDropout | **58.73 ± 3.46** | 21.81 ± 0.03 | N/A | 2.69x |
| ChannelShuffle | 49.64 ± 1.43 | 19.99 ± 0.03 | **958.35 ± 0.20** | 0.05x |
| CoarseDropout | **66.85 ± 4.42** | N/A | N/A | N/A |
| ColorJitter | 10.85 ± 0.53 | 18.79 ± 0.03 | **68.75 ± 0.13** | 0.16x |
| Contrast | 188.99 ± 6.48 | 21.69 ± 0.04 | **546.55 ± 13.23** | 0.35x |
| CornerIllumination | **5.43 ± 0.23** | 2.60 ± 0.07 | N/A | 2.09x |
| Elastic | 4.28 ± 0.15 | N/A | **126.83 ± 1.28** | 0.03x |
| Equalize | 13.23 ± 0.14 | 4.21 ± 0.00 | **191.55 ± 1.25** | 0.07x |
| Erasing | 71.48 ± 2.52 | N/A | **254.59 ± 6.57** | 0.28x |
| GaussianBlur | 26.76 ± 0.90 | 21.61 ± 0.05 | **543.44 ± 11.50** | 0.05x |
| GaussianIllumination | 6.89 ± 0.53 | **20.33 ± 0.08** | N/A | 0.34x |
| GaussianNoise | 8.82 ± 0.36 | **22.38 ± 0.08** | N/A | 0.39x |
| Grayscale | 158.36 ± 2.82 | 22.24 ± 0.04 | **838.40 ± 466.76** | 0.19x |
| HSV | **6.71 ± 0.33** | N/A | N/A | N/A |
| HorizontalFlip | 27.26 ± 0.14 | 21.86 ± 0.07 | **977.87 ± 49.03** | 0.03x |
| Hue | 14.22 ± 0.38 | **19.53 ± 0.02** | N/A | 0.73x |
| Invert | 355.23 ± 2.36 | 21.91 ± 0.23 | **843.27 ± 176.00** | 0.42x |
| JpegCompression | **19.69 ± 0.36** | N/A | N/A | N/A |
| LinearIllumination | **4.90 ± 0.28** | 4.29 ± 0.19 | N/A | 1.14x |
| MedianBlur | **13.36 ± 0.22** | 8.39 ± 0.09 | N/A | 1.59x |
| MotionBlur | **35.66 ± 0.56** | N/A | N/A | N/A |
| Normalize | 21.44 ± 0.34 | 21.82 ± 0.02 | **460.80 ± 0.18** | 0.05x |
| OpticalDistortion | **4.50 ± 0.10** | N/A | N/A | N/A |
| Pad | 66.10 ± 0.43 | N/A | **759.68 ± 337.78** | 0.09x |
| Perspective | 4.21 ± 0.19 | N/A | **434.75 ± 0.14** | 0.01x |
| PlankianJitter | **22.01 ± 1.27** | 10.85 ± 0.01 | N/A | 2.03x |
| PlasmaBrightness | 3.31 ± 0.05 | **16.94 ± 0.36** | N/A | 0.20x |
| PlasmaContrast | 2.66 ± 0.08 | **16.97 ± 0.03** | N/A | 0.16x |
| PlasmaShadow | 5.97 ± 0.18 | **19.03 ± 0.50** | N/A | 0.31x |
| Posterize | 65.56 ± 1.27 | N/A | **631.46 ± 14.74** | 0.10x |
| RGBShift | **32.34 ± 0.70** | 22.27 ± 0.04 | N/A | 1.45x |
| Rain | **23.19 ± 0.50** | 3.77 ± 0.00 | N/A | 6.15x |
| RandomCrop128 | 604.41 ± 12.48 | 65.33 ± 0.35 | **1132.79 ± 15.23** | 0.53x |
| RandomGamma | **190.73 ± 3.99** | 21.63 ± 0.02 | N/A | 8.82x |
| RandomResizedCrop | 15.64 ± 0.33 | 6.29 ± 0.03 | **182.09 ± 15.75** | 0.09x |
| Resize | 14.19 ± 0.88 | 5.87 ± 0.03 | **139.96 ± 35.04** | 0.10x |
| Rotate | 26.87 ± 2.83 | 21.53 ± 0.05 | **534.18 ± 0.16** | 0.05x |
| SaltAndPepper | **10.10 ± 0.04** | 8.82 ± 0.12 | N/A | 1.15x |
| Saturation | 8.82 ± 0.15 | **36.56 ± 0.12** | N/A | 0.24x |
| Sharpen | 25.33 ± 1.19 | 17.86 ± 0.03 | **420.09 ± 8.99** | 0.06x |
| Shear | **4.44 ± 0.06** | N/A | N/A | N/A |
| Snow | **12.58 ± 0.09** | N/A | N/A | N/A |
| Solarize | 50.74 ± 4.14 | 20.73 ± 0.02 | **628.42 ± 5.91** | 0.08x |
| ThinPlateSpline | 4.42 ± 0.14 | **44.90 ± 0.67** | N/A | 0.10x |
| VerticalFlip | 381.54 ± 3.99 | 21.96 ± 0.24 | **977.92 ± 5.22** | 0.39x |

## Torchvision Metadata

```yaml
system_info:
  python_version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27)
    [GCC 11.2.0]
  platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.31
  processor: x86_64
  cpu_count: '64'
  timestamp: '2025-03-11T11:14:57.765540+00:00'
library_versions:
  torchvision: 0.21.0
  numpy: 2.2.3
  pillow: 11.1.0
  opencv-python-headless: not installed
  torch: 2.6.0
  opencv-python: not installed
thread_settings:
  environment:
    OMP_NUM_THREADS: '1'
    OPENBLAS_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    VECLIB_MAXIMUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
  opencv: not installed
  pytorch:
    threads: 32
    gpu_available: true
    gpu_device: 0
    gpu_name: NVIDIA GeForce RTX 4090
    gpu_memory_total: 23.55084228515625
    gpu_memory_allocated: 15.05643081665039
  pillow:
    threads: unknown
    simd: false
benchmark_params:
  num_videos: 200
  num_runs: 10
  max_warmup_iterations: 100
  warmup_window: 5
  warmup_threshold: 0.05
  min_warmup_windows: 3
precision: torch.float16

```

## Kornia Metadata

```yaml
system_info:
  python_version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27)
    [GCC 11.2.0]
  platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.31
  processor: x86_64
  cpu_count: '64'
  timestamp: '2025-03-11T00:46:14.791885+00:00'
library_versions:
  kornia: 0.8.0
  numpy: 2.2.3
  pillow: 11.1.0
  opencv-python-headless: not installed
  torch: 2.6.0
  opencv-python: not installed
thread_settings:
  environment:
    OMP_NUM_THREADS: '1'
    OPENBLAS_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    VECLIB_MAXIMUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
  opencv: not installed
  pytorch:
    threads: 32
    gpu_available: true
    gpu_device: 0
    gpu_name: NVIDIA GeForce RTX 4090
    gpu_memory_total: 23.55084228515625
    gpu_memory_allocated: 15.05643081665039
  pillow:
    threads: unknown
    simd: false
benchmark_params:
  num_videos: 200
  num_runs: 5
  max_warmup_iterations: 100
  warmup_window: 5
  warmup_threshold: 0.05
  min_warmup_windows: 3
precision: torch.float16

```

## Albumentations Metadata

```yaml
system_info:
  python_version: 3.12.8 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 10:37:40)
    [Clang 14.0.6 ]
  platform: macOS-15.1-arm64-arm-64bit
  processor: arm
  cpu_count: '16'
  timestamp: '2025-04-16T17:31:48.175211+00:00'
library_versions:
  albumentations: 2.0.5
  numpy: 2.2.4
  pillow: 11.2.1
  opencv-python-headless: 4.11.0.86
  torch: 2.6.0
  opencv-python: not installed
thread_settings:
  environment:
    OMP_NUM_THREADS: '1'
    OPENBLAS_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    VECLIB_MAXIMUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
  opencv:
    threads: 1
    opencl: false
  pytorch:
    threads: 1
    gpu_available: false
    gpu_device: null
  pillow:
    threads: unknown
    simd: false
benchmark_params:
  num_videos: 200
  num_runs: 5
  max_warmup_iterations: 100
  warmup_window: 5
  warmup_threshold: 0.05
  min_warmup_windows: 3

```


<!-- BENCHMARK_RESULTS_END -->

## Analysis

The benchmark results show interesting trade-offs between CPU and GPU processing:

- **CPU Advantages**:
  - Better for simple transformations with low computational complexity
  - No data transfer overhead between CPU and GPU
  - More consistent performance across different transform types

- **GPU Advantages**:
  - Significantly faster for complex transformations
  - Better scaling with video resolution
  - More efficient for batch processing

## Recommendations

Based on the benchmark results, we recommend:

1. For simple transformations on a small number of videos, CPU processing may be sufficient
2. For complex transformations or batch processing, GPU acceleration provides significant benefits
3. Consider the specific transformations you need and their relative performance on CPU vs GPU
