<!-- This file is auto-generated. Do not edit directly. -->

# Video Augmentation Benchmarks

This directory contains benchmark results for video augmentation libraries.

## Overview

The video benchmarks measure the performance of various augmentation libraries on video transformations.
The benchmarks compare CPU-based processing (Albumentations) with GPU-accelerated processing (Kornia).

## Dataset

The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains 13,320 videos from
101 action categories. The videos are realistic, collected from YouTube, and include a wide variety of camera
motion, object appearance, pose, scale, viewpoint, and background. This makes it an excellent dataset for
benchmarking video augmentation performance across diverse real-world scenarios.

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

### Video Benchmark Results

Number shows how many videos per second can be processed. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

| Transform | albumentations (videos per second)<br>arm (1 core) | kornia (videos per second)<br>NVIDIA GeForce RTX 4090 | torchvision (videos per second)<br>NVIDIA GeForce RTX 4090 | Speedup<br>(Alb/fastest other) |
|---|---|---|---|---|
| Affine | 4.61 ± 0.19 | 21.39 ± 0.05 | **452.58 ± 0.14** | 0.01x |
| AutoContrast | 20.44 ± 0.39 | 21.41 ± 0.02 | **577.72 ± 16.86** | 0.04x |
| Blur | **54.30 ± 0.54** | 20.61 ± 0.06 | N/A | 2.63x |
| Brightness | 204.47 ± 10.47 | 21.85 ± 0.02 | **755.52 ± 435.17** | 0.27x |
| CLAHE | **9.57 ± 0.02** | N/A | N/A | N/A |
| CenterCrop128 | 699.10 ± 19.80 | 70.12 ± 1.29 | **1133.39 ± 234.60** | 0.62x |
| ChannelDropout | **55.79 ± 1.00** | 21.81 ± 0.03 | N/A | 2.56x |
| ChannelShuffle | 50.13 ± 0.72 | 19.99 ± 0.03 | **958.35 ± 0.20** | 0.05x |
| CoarseDropout | **75.44 ± 1.25** | N/A | N/A | N/A |
| ColorJitter | 10.53 ± 0.13 | 18.79 ± 0.03 | **68.75 ± 0.13** | 0.15x |
| Contrast | 202.39 ± 12.96 | 21.69 ± 0.04 | **546.55 ± 13.23** | 0.37x |
| CornerIllumination | **5.35 ± 0.10** | 2.60 ± 0.07 | N/A | 2.05x |
| Elastic | 4.24 ± 0.03 | N/A | **126.83 ± 1.28** | 0.03x |
| Equalize | 14.34 ± 0.19 | 4.21 ± 0.00 | **191.55 ± 1.25** | 0.07x |
| Erasing | 77.47 ± 2.67 | N/A | **254.59 ± 6.57** | 0.30x |
| GaussianBlur | 26.70 ± 0.18 | 21.61 ± 0.05 | **543.44 ± 11.50** | 0.05x |
| GaussianIllumination | 7.36 ± 0.07 | **20.33 ± 0.08** | N/A | 0.36x |
| GaussianNoise | 8.65 ± 0.46 | **22.38 ± 0.08** | N/A | 0.39x |
| Grayscale | 151.22 ± 3.11 | 22.24 ± 0.04 | **838.40 ± 466.76** | 0.18x |
| HSV | **7.46 ± 0.14** | N/A | N/A | N/A |
| HorizontalFlip | 8.68 ± 0.10 | 21.86 ± 0.07 | **977.87 ± 49.03** | 0.01x |
| Hue | 14.71 ± 0.45 | **19.53 ± 0.02** | N/A | 0.75x |
| Invert | 300.44 ± 15.74 | 21.91 ± 0.23 | **843.27 ± 176.00** | 0.36x |
| JpegCompression | **21.62 ± 0.21** | N/A | N/A | N/A |
| LinearIllumination | **4.74 ± 0.20** | 4.29 ± 0.19 | N/A | 1.10x |
| MedianBlur | **14.51 ± 0.13** | 8.39 ± 0.09 | N/A | 1.73x |
| MotionBlur | **39.38 ± 1.40** | N/A | N/A | N/A |
| Normalize | 23.18 ± 0.27 | 21.82 ± 0.02 | **460.80 ± 0.18** | 0.05x |
| OpticalDistortion | **4.43 ± 0.07** | N/A | N/A | N/A |
| Pad | 63.53 ± 3.99 | N/A | **759.68 ± 337.78** | 0.08x |
| Perspective | 4.20 ± 0.18 | N/A | **434.75 ± 0.14** | 0.01x |
| PlankianJitter | **23.43 ± 0.74** | 10.85 ± 0.01 | N/A | 2.16x |
| PlasmaBrightness | 3.47 ± 0.01 | **16.94 ± 0.36** | N/A | 0.20x |
| PlasmaContrast | 2.62 ± 0.04 | **16.97 ± 0.03** | N/A | 0.15x |
| PlasmaShadow | 5.92 ± 0.06 | **19.03 ± 0.50** | N/A | 0.31x |
| Posterize | 54.92 ± 1.15 | N/A | **631.46 ± 14.74** | 0.09x |
| RGBShift | **32.18 ± 0.31** | 22.27 ± 0.04 | N/A | 1.44x |
| Rain | **24.26 ± 1.55** | 3.77 ± 0.00 | N/A | 6.43x |
| RandomCrop128 | 693.41 ± 6.85 | 65.33 ± 0.35 | **1132.79 ± 15.23** | 0.61x |
| RandomGamma | **207.41 ± 1.62** | 21.63 ± 0.02 | N/A | 9.59x |
| RandomResizedCrop | 16.51 ± 1.12 | 6.29 ± 0.03 | **182.09 ± 15.75** | 0.09x |
| Resize | 16.08 ± 0.08 | 5.87 ± 0.03 | **139.96 ± 35.04** | 0.11x |
| Rotate | 28.61 ± 0.94 | 21.53 ± 0.05 | **534.18 ± 0.16** | 0.05x |
| SaltAndPepper | **10.09 ± 0.08** | 8.82 ± 0.12 | N/A | 1.14x |
| Saturation | 8.70 ± 0.13 | **36.56 ± 0.12** | N/A | 0.24x |
| Sharpen | 25.65 ± 0.14 | 17.86 ± 0.03 | **420.09 ± 8.99** | 0.06x |
| Shear | **4.47 ± 0.08** | N/A | N/A | N/A |
| Snow | **12.52 ± 0.46** | N/A | N/A | N/A |
| Solarize | 50.93 ± 1.45 | 20.73 ± 0.02 | **628.42 ± 5.91** | 0.08x |
| ThinPlateSpline | 4.34 ± 0.06 | **44.90 ± 0.67** | N/A | 0.10x |
| VerticalFlip | 9.22 ± 0.31 | 21.96 ± 0.24 | **977.92 ± 5.22** | 0.01x |

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
  timestamp: '2025-03-17T21:40:35.132413+00:00'
library_versions:
  albumentations: 2.0.5
  numpy: 2.2.4
  pillow: 11.1.0
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
