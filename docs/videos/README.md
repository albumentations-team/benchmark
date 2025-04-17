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
| Affine | 4.51 ± 0.03 | 21.39 ± 0.05 | **452.58 ± 0.14** | 0.01x |
| AutoContrast | 20.56 ± 0.19 | 21.41 ± 0.02 | **577.72 ± 16.86** | 0.04x |
| Blur | **52.40 ± 1.82** | 20.61 ± 0.06 | N/A | 2.54x |
| Brightness | 187.30 ± 5.12 | 21.85 ± 0.02 | **755.52 ± 435.17** | 0.25x |
| CLAHE | **9.29 ± 0.05** | N/A | N/A | N/A |
| CenterCrop128 | 823.73 ± 20.90 | 70.12 ± 1.29 | **1133.39 ± 234.60** | 0.73x |
| ChannelDropout | **62.00 ± 2.35** | 21.81 ± 0.03 | N/A | 2.84x |
| ChannelShuffle | 23.33 ± 0.10 | 19.99 ± 0.03 | **958.35 ± 0.20** | 0.02x |
| CoarseDropout | **68.63 ± 4.09** | N/A | N/A | N/A |
| ColorJitter | 10.89 ± 0.56 | 18.79 ± 0.03 | **68.75 ± 0.13** | 0.16x |
| Contrast | 189.97 ± 0.73 | 21.69 ± 0.04 | **546.55 ± 13.23** | 0.35x |
| CornerIllumination | **5.92 ± 0.06** | 2.60 ± 0.07 | N/A | 2.27x |
| Elastic | 4.19 ± 0.12 | N/A | **126.83 ± 1.28** | 0.03x |
| Equalize | 13.85 ± 0.09 | 4.21 ± 0.00 | **191.55 ± 1.25** | 0.07x |
| Erasing | 70.64 ± 0.63 | N/A | **254.59 ± 6.57** | 0.28x |
| GaussianBlur | 28.56 ± 0.25 | 21.61 ± 0.05 | **543.44 ± 11.50** | 0.05x |
| GaussianIllumination | 7.98 ± 0.11 | **20.33 ± 0.08** | N/A | 0.39x |
| GaussianNoise | 9.82 ± 0.31 | **22.38 ± 0.08** | N/A | 0.44x |
| Grayscale | 157.71 ± 3.00 | 22.24 ± 0.04 | **838.40 ± 466.76** | 0.19x |
| HSV | **7.58 ± 0.06** | N/A | N/A | N/A |
| HorizontalFlip | 27.59 ± 0.05 | 21.86 ± 0.07 | **977.87 ± 49.03** | 0.03x |
| Hue | 15.57 ± 0.48 | **19.53 ± 0.02** | N/A | 0.80x |
| Invert | 362.78 ± 10.24 | 21.91 ± 0.23 | **843.27 ± 176.00** | 0.43x |
| JpegCompression | **20.67 ± 0.34** | N/A | N/A | N/A |
| LinearIllumination | **5.40 ± 0.03** | 4.29 ± 0.19 | N/A | 1.26x |
| MedianBlur | **13.97 ± 0.12** | 8.39 ± 0.09 | N/A | 1.66x |
| MotionBlur | **36.79 ± 0.76** | N/A | N/A | N/A |
| Normalize | 21.48 ± 0.09 | 21.82 ± 0.02 | **460.80 ± 0.18** | 0.05x |
| OpticalDistortion | **4.71 ± 0.02** | N/A | N/A | N/A |
| Pad | 70.98 ± 0.86 | N/A | **759.68 ± 337.78** | 0.09x |
| Perspective | 4.40 ± 0.06 | N/A | **434.75 ± 0.14** | 0.01x |
| PlankianJitter | **22.56 ± 2.75** | 10.85 ± 0.01 | N/A | 2.08x |
| PlasmaBrightness | 3.42 ± 0.01 | **16.94 ± 0.36** | N/A | 0.20x |
| PlasmaContrast | 2.69 ± 0.01 | **16.97 ± 0.03** | N/A | 0.16x |
| PlasmaShadow | 6.14 ± 0.02 | **19.03 ± 0.50** | N/A | 0.32x |
| Posterize | 65.43 ± 0.61 | N/A | **631.46 ± 14.74** | 0.10x |
| RGBShift | **34.41 ± 0.18** | 22.27 ± 0.04 | N/A | 1.55x |
| Rain | **24.70 ± 0.49** | 3.77 ± 0.00 | N/A | 6.55x |
| RandomCrop128 | 766.40 ± 17.69 | 65.33 ± 0.35 | **1132.79 ± 15.23** | 0.68x |
| RandomGamma | **189.61 ± 3.37** | 21.63 ± 0.02 | N/A | 8.77x |
| RandomResizedCrop | 16.63 ± 1.06 | 6.29 ± 0.03 | **182.09 ± 15.75** | 0.09x |
| Resize | 15.60 ± 0.22 | 5.87 ± 0.03 | **139.96 ± 35.04** | 0.11x |
| Rotate | 28.29 ± 0.32 | 21.53 ± 0.05 | **534.18 ± 0.16** | 0.05x |
| SaltAndPepper | **10.22 ± 0.07** | 8.82 ± 0.12 | N/A | 1.16x |
| Saturation | 9.10 ± 0.10 | **36.56 ± 0.12** | N/A | 0.25x |
| Sharpen | 26.69 ± 0.58 | 17.86 ± 0.03 | **420.09 ± 8.99** | 0.06x |
| Shear | **4.72 ± 0.02** | N/A | N/A | N/A |
| Snow | **13.26 ± 0.10** | N/A | N/A | N/A |
| Solarize | 58.35 ± 0.60 | 20.73 ± 0.02 | **628.42 ± 5.91** | 0.09x |
| ThinPlateSpline | 4.60 ± 0.03 | **44.90 ± 0.67** | N/A | 0.10x |
| VerticalFlip | 391.52 ± 12.26 | 21.96 ± 0.24 | **977.92 ± 5.22** | 0.40x |

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
  timestamp: '2025-04-17T02:12:28.247902+00:00'
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
