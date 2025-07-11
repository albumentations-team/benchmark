<!-- This file is auto-generated. Do not edit directly. -->

# Video Augmentation Benchmarks

This directory contains benchmark results for video augmentation libraries.

## Overview

The video benchmarks measure the performance of various augmentation libraries on video transformations.
The benchmarks compare CPU-based processing (Albumentationsx) with GPU-accelerated processing (Kornia).

## Dataset

The benchmarks use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains 13,320 videos from
101 action categories. The videos are realistic, collected from YouTube, and include a wide variety of camera
motion, object appearance, pose, scale, viewpoint, and background. This makes it an excellent dataset for
benchmarking video augmentation performance across diverse real-world scenarios.

You can download the dataset from: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

## Methodology

1. **Video Loading**: Videos are loaded using library-specific loaders:
   - OpenCV for Albumentationsx
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
- Albumentationsx: CPU-based processing (single thread)
- Kornia: GPU-accelerated processing (NVIDIA GPUs)

This provides insights into the trade-offs between CPU and GPU processing for video augmentation.

## Running the Benchmarks

To run the video benchmarks:

```bash
./run_video_single.sh -l albumentationsx -d /path/to/videos -o /path/to/output
```

To run all libraries and generate a comparison:

```bash
./run_video_all.sh -d /path/to/videos -o /path/to/output
```

## Benchmark Results

### Video Benchmark Results

Number shows how many videos per second can be processed. Larger is better.
The Speedup column shows how many times faster Albumentationsx is compared to the fastest other
library for each transform.

| Transform            | albumentationsx<br>arm (1 core)   | kornia<br>NVIDIA GeForce RTX 4090   | torchvision<br>NVIDIA GeForce RTX 4090   | Speedup<br>(Albx/fastest other)   |
|:---------------------|:----------------------------------|:------------------------------------|:-----------------------------------------|:----------------------------------|
| Affine               | 4.15 ± 0.07                       | 21.39 ± 0.05                        | **452.58 ± 0.14**                        | 0.01x                             |
| AutoContrast         | 19.71 ± 0.02                      | 21.41 ± 0.02                        | **577.72 ± 16.86**                       | 0.03x                             |
| Blur                 | **46.78 ± 1.07**                  | 20.61 ± 0.06                        | -                                        | 2.27x                             |
| Brightness           | 174.75 ± 4.55                     | 21.85 ± 0.02                        | **755.52 ± 435.17**                      | 0.23x                             |
| CenterCrop128        | 729.91 ± 31.26                    | 70.12 ± 1.29                        | **1133.39 ± 234.60**                     | 0.64x                             |
| ChannelDropout       | **60.88 ± 2.10**                  | 21.81 ± 0.03                        | -                                        | 2.79x                             |
| ChannelShuffle       | 22.01 ± 0.13                      | 19.99 ± 0.03                        | **958.35 ± 0.20**                        | 0.02x                             |
| ColorJitter          | 10.51 ± 0.48                      | 18.79 ± 0.03                        | **68.75 ± 0.13**                         | 0.15x                             |
| Contrast             | 174.89 ± 3.36                     | 21.69 ± 0.04                        | **546.55 ± 13.23**                       | 0.32x                             |
| CornerIllumination   | **5.55 ± 0.05**                   | 2.60 ± 0.07                         | -                                        | 2.13x                             |
| Elastic              | 4.03 ± 0.18                       | -                                   | **126.83 ± 1.28**                        | 0.03x                             |
| Equalize             | 11.87 ± 0.55                      | 4.21 ± 0.00                         | **191.55 ± 1.25**                        | 0.06x                             |
| Erasing              | **378.69 ± 37.79**                | -                                   | 254.59 ± 6.57                            | 1.49x                             |
| GaussianBlur         | 26.57 ± 0.66                      | 21.61 ± 0.05                        | **543.44 ± 11.50**                       | 0.05x                             |
| GaussianIllumination | 7.35 ± 0.11                       | **20.33 ± 0.08**                    | -                                        | 0.36x                             |
| GaussianNoise        | 9.06 ± 0.07                       | **22.38 ± 0.08**                    | -                                        | 0.40x                             |
| Grayscale            | 70.04 ± 0.68                      | 22.24 ± 0.04                        | **838.40 ± 466.76**                      | 0.08x                             |
| HorizontalFlip       | 26.79 ± 0.38                      | 21.86 ± 0.07                        | **977.87 ± 49.03**                       | 0.03x                             |
| Hue                  | 14.30 ± 0.40                      | **19.53 ± 0.02**                    | -                                        | 0.73x                             |
| Invert               | 315.23 ± 5.42                     | 21.91 ± 0.23                        | **843.27 ± 176.00**                      | 0.37x                             |
| LinearIllumination   | **5.21 ± 0.04**                   | 4.29 ± 0.19                         | -                                        | 1.21x                             |
| MedianBlur           | **13.20 ± 0.22**                  | 8.39 ± 0.09                         | -                                        | 1.57x                             |
| Normalize            | 19.32 ± 0.33                      | 21.82 ± 0.02                        | **460.80 ± 0.18**                        | 0.04x                             |
| Pad                  | 184.35 ± 7.14                     | -                                   | **759.68 ± 337.78**                      | 0.24x                             |
| Perspective          | 4.18 ± 0.23                       | -                                   | **434.75 ± 0.14**                        | 0.01x                             |
| PlankianJitter       | **50.65 ± 0.27**                  | 10.85 ± 0.01                        | -                                        | 4.67x                             |
| PlasmaBrightness     | 3.51 ± 0.01                       | **16.94 ± 0.36**                    | -                                        | 0.21x                             |
| PlasmaContrast       | 2.76 ± 0.01                       | **16.97 ± 0.03**                    | -                                        | 0.16x                             |
| PlasmaShadow         | 6.11 ± 0.03                       | **19.03 ± 0.50**                    | -                                        | 0.32x                             |
| Posterize            | 184.80 ± 4.42                     | -                                   | **631.46 ± 14.74**                       | 0.29x                             |
| RGBShift             | 7.41 ± 0.10                       | **22.27 ± 0.04**                    | -                                        | 0.33x                             |
| Rain                 | **24.05 ± 0.17**                  | 3.77 ± 0.00                         | -                                        | 6.38x                             |
| RandomCrop128        | 667.36 ± 15.17                    | 65.33 ± 0.35                        | **1132.79 ± 15.23**                      | 0.59x                             |
| RandomGamma          | **198.35 ± 4.67**                 | 21.63 ± 0.02                        | -                                        | 9.17x                             |
| RandomResizedCrop    | 13.98 ± 0.93                      | 6.29 ± 0.03                         | **182.09 ± 15.75**                       | 0.08x                             |
| Resize               | 13.85 ± 0.40                      | 5.87 ± 0.03                         | **139.96 ± 35.04**                       | 0.10x                             |
| Rotate               | 22.52 ± 2.14                      | 21.53 ± 0.05                        | **534.18 ± 0.16**                        | 0.04x                             |
| SaltAndPepper        | **11.24 ± 0.02**                  | 8.82 ± 0.12                         | -                                        | 1.28x                             |
| Saturation           | 8.74 ± 0.12                       | **36.56 ± 0.12**                    | -                                        | 0.24x                             |
| Sharpen              | 24.35 ± 1.10                      | 17.86 ± 0.03                        | **420.09 ± 8.99**                        | 0.06x                             |
| Solarize             | 50.93 ± 1.58                      | 20.73 ± 0.02                        | **628.42 ± 5.91**                        | 0.08x                             |
| ThinPlateSpline      | 4.51 ± 0.02                       | **44.90 ± 0.67**                    | -                                        | 0.10x                             |
| VerticalFlip         | 336.37 ± 35.94                    | 21.96 ± 0.24                        | **977.92 ± 5.22**                        | 0.34x                             |

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

## Albumentationsx Metadata

```yaml
system_info:
  python_version: 3.12.8 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 10:37:40)
    [Clang 14.0.6 ]
  platform: macOS-15.5-arm64-arm-64bit
  processor: arm
  cpu_count: '16'
  timestamp: '2025-07-05T14:41:11.448369+00:00'
library_versions:
  albumentationsx: not installed
  numpy: 2.3.1
  pillow: not installed
  opencv-python-headless: 4.11.0.86
  torch: not installed
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
  pytorch: not installed
  pillow: not installed
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
