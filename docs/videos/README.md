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
| Affine | 4.05 ± 0.16 | 21.39 ± 0.05 | **452.58 ± 0.14** | 0.01x |
| AutoContrast | 19.51 ± 0.21 | 21.41 ± 0.02 | **577.72 ± 16.86** | 0.03x |
| Blur | **43.58 ± 1.95** | 20.61 ± 0.06 | N/A | 2.11x |
| Brightness | 177.48 ± 8.78 | 21.85 ± 0.02 | **755.52 ± 435.17** | 0.23x |
| CLAHE | **8.49 ± 0.24** | N/A | N/A | N/A |
| CenterCrop128 | 781.08 ± 31.02 | 70.12 ± 1.29 | **1133.39 ± 234.60** | 0.69x |
| ChannelDropout | **59.40 ± 1.73** | 21.81 ± 0.03 | N/A | 2.72x |
| ChannelShuffle | 22.11 ± 0.13 | 19.99 ± 0.03 | **958.35 ± 0.20** | 0.02x |
| CoarseDropout | **299.85 ± 7.78** | N/A | N/A | N/A |
| ColorJitter | 10.38 ± 0.39 | 18.79 ± 0.03 | **68.75 ± 0.13** | 0.15x |
| Contrast | 175.39 ± 6.60 | 21.69 ± 0.04 | **546.55 ± 13.23** | 0.32x |
| CornerIllumination | **4.86 ± 0.15** | 2.60 ± 0.07 | N/A | 1.87x |
| Elastic | 4.23 ± 0.02 | N/A | **126.83 ± 1.28** | 0.03x |
| Equalize | 12.90 ± 0.37 | 4.21 ± 0.00 | **191.55 ± 1.25** | 0.07x |
| Erasing | **365.07 ± 7.73** | N/A | 254.59 ± 6.57 | 1.43x |
| GaussianBlur | 25.92 ± 0.26 | 21.61 ± 0.05 | **543.44 ± 11.50** | 0.05x |
| GaussianIllumination | 6.47 ± 0.38 | **20.33 ± 0.08** | N/A | 0.32x |
| GaussianNoise | 9.00 ± 0.28 | **22.38 ± 0.08** | N/A | 0.40x |
| Grayscale | 147.73 ± 2.65 | 22.24 ± 0.04 | **838.40 ± 466.76** | 0.18x |
| HSV | **6.62 ± 0.07** | N/A | N/A | N/A |
| HorizontalFlip | 26.98 ± 0.18 | 21.86 ± 0.07 | **977.87 ± 49.03** | 0.03x |
| Hue | 13.68 ± 0.50 | **19.53 ± 0.02** | N/A | 0.70x |
| Invert | 306.14 ± 23.13 | 21.91 ± 0.23 | **843.27 ± 176.00** | 0.36x |
| JpegCompression | **19.85 ± 0.25** | N/A | N/A | N/A |
| LinearIllumination | **4.74 ± 0.17** | 4.29 ± 0.19 | N/A | 1.10x |
| MedianBlur | **13.15 ± 0.15** | 8.39 ± 0.09 | N/A | 1.57x |
| MotionBlur | **40.11 ± 0.83** | N/A | N/A | N/A |
| Normalize | 21.13 ± 0.31 | 21.82 ± 0.02 | **460.80 ± 0.18** | 0.05x |
| OpticalDistortion | **4.62 ± 0.02** | N/A | N/A | N/A |
| Pad | 217.81 ± 1.31 | N/A | **759.68 ± 337.78** | 0.29x |
| Perspective | 4.14 ± 0.13 | N/A | **434.75 ± 0.14** | 0.01x |
| PlankianJitter | **24.86 ± 1.67** | 10.85 ± 0.01 | N/A | 2.29x |
| PlasmaBrightness | 3.32 ± 0.05 | **16.94 ± 0.36** | N/A | 0.20x |
| PlasmaContrast | 2.50 ± 0.02 | **16.97 ± 0.03** | N/A | 0.15x |
| PlasmaShadow | 5.74 ± 0.24 | **19.03 ± 0.50** | N/A | 0.30x |
| Posterize | 58.97 ± 1.09 | N/A | **631.46 ± 14.74** | 0.09x |
| RGBShift | **31.06 ± 0.72** | 22.27 ± 0.04 | N/A | 1.39x |
| Rain | **24.82 ± 0.29** | 3.77 ± 0.00 | N/A | 6.58x |
| RandomCrop128 | 755.55 ± 24.20 | 65.33 ± 0.35 | **1132.79 ± 15.23** | 0.67x |
| RandomGamma | **184.46 ± 2.85** | 21.63 ± 0.02 | N/A | 8.53x |
| RandomResizedCrop | 15.95 ± 0.92 | 6.29 ± 0.03 | **182.09 ± 15.75** | 0.09x |
| Resize | 14.08 ± 0.61 | 5.87 ± 0.03 | **139.96 ± 35.04** | 0.10x |
| Rotate | 28.29 ± 1.82 | 21.53 ± 0.05 | **534.18 ± 0.16** | 0.05x |
| SaltAndPepper | **10.00 ± 0.06** | 8.82 ± 0.12 | N/A | 1.13x |
| Saturation | 9.01 ± 0.05 | **36.56 ± 0.12** | N/A | 0.25x |
| Sharpen | 24.77 ± 0.47 | 17.86 ± 0.03 | **420.09 ± 8.99** | 0.06x |
| Shear | **4.47 ± 0.02** | N/A | N/A | N/A |
| Snow | **12.62 ± 0.28** | N/A | N/A | N/A |
| Solarize | 51.74 ± 1.30 | 20.73 ± 0.02 | **628.42 ± 5.91** | 0.08x |
| ThinPlateSpline | 4.33 ± 0.02 | **44.90 ± 0.67** | N/A | 0.10x |
| VerticalFlip | 394.56 ± 5.96 | 21.96 ± 0.24 | **977.92 ± 5.22** | 0.40x |

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
  platform: macOS-15.5-arm64-arm-64bit
  processor: arm
  cpu_count: '16'
  timestamp: '2025-05-26T23:29:19.852599+00:00'
library_versions:
  albumentations: 2.0.7
  numpy: 2.2.6
  pillow: 11.2.1
  opencv-python-headless: 4.11.0.86
  torch: 2.7.0
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
