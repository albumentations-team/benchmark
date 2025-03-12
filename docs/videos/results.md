<!-- This file is auto-generated. Do not edit directly. -->

# Video Benchmark Results

Number shows how many videos per second can be processed. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other library for each transform.

| Transform | albumentations (videos per second)<br>arm (16 cores) | kornia (videos per second)<br>NVIDIA GeForce RTX 4090 | torchvision (videos per second)<br>NVIDIA GeForce RTX 4090 | Speedup<br>(Alb/fastest other) |
|---|---|---|---|---|
| Affine | 4.45 ± 0.06 | 21.39 ± 0.05 | **452.58 ± 0.14** | 0.01× |
| AutoContrast | 20.85 ± 0.10 | 21.41 ± 0.02 | **577.72 ± 16.86** | 0.04× |
| Blur | **49.61 ± 1.95** | 20.61 ± 0.06 | N/A | 2.41× |
| Brightness | 56.84 ± 1.94 | 21.85 ± 0.02 | **755.52 ± 435.17** | 0.08× |
| CLAHE | **8.89 ± 0.09** | N/A | N/A | N/A |
| CenterCrop128 | 733.66 ± 4.03 | 70.12 ± 1.29 | **1133.39 ± 234.60** | 0.65× |
| ChannelDropout | **58.28 ± 2.96** | 21.81 ± 0.03 | N/A | 2.67× |
| ChannelShuffle | 46.92 ± 2.29 | 19.99 ± 0.03 | **958.35 ± 0.20** | 0.05× |
| CoarseDropout | **65.62 ± 1.82** | N/A | N/A | N/A |
| ColorJitter | 10.67 ± 0.23 | 18.79 ± 0.03 | **68.75 ± 0.13** | 0.16× |
| Contrast | 58.81 ± 1.10 | 21.69 ± 0.04 | **546.55 ± 13.23** | 0.11× |
| CornerIllumination | **4.80 ± 0.47** | 2.60 ± 0.07 | N/A | 1.84× |
| Elastic | 4.31 ± 0.07 | N/A | **126.83 ± 1.28** | 0.03× |
| Equalize | 13.09 ± 0.22 | 4.21 ± 0.00 | **191.55 ± 1.25** | 0.07× |
| Erasing | 69.44 ± 3.31 | N/A | **254.59 ± 6.57** | 0.27× |
| GaussianBlur | 25.63 ± 0.42 | 21.61 ± 0.05 | **543.44 ± 11.50** | 0.05× |
| GaussianIllumination | 7.10 ± 0.15 | **20.33 ± 0.08** | N/A | 0.35× |
| GaussianNoise | 8.40 ± 0.19 | **22.38 ± 0.08** | N/A | 0.38× |
| Grayscale | 152.01 ± 11.18 | 22.24 ± 0.04 | **838.40 ± 466.76** | 0.18× |
| HSV | **6.48 ± 0.35** | N/A | N/A | N/A |
| HorizontalFlip | 8.69 ± 0.21 | 21.86 ± 0.07 | **977.87 ± 49.03** | 0.01× |
| Hue | 14.47 ± 0.33 | **19.53 ± 0.02** | N/A | 0.74× |
| Invert | 67.77 ± 2.60 | 21.91 ± 0.23 | **843.27 ± 176.00** | 0.08× |
| JpegCompression | **19.62 ± 0.20** | N/A | N/A | N/A |
| LinearIllumination | **4.81 ± 0.25** | 4.29 ± 0.19 | N/A | 1.12× |
| MedianBlur | **13.87 ± 0.33** | 8.39 ± 0.09 | N/A | 1.65× |
| MotionBlur | **33.49 ± 0.66** | N/A | N/A | N/A |
| Normalize | 21.70 ± 0.18 | 21.82 ± 0.02 | **460.80 ± 0.18** | 0.05× |
| OpticalDistortion | **4.29 ± 0.10** | N/A | N/A | N/A |
| Pad | 68.10 ± 0.91 | N/A | **759.68 ± 337.78** | 0.09× |
| Perspective | 4.37 ± 0.08 | N/A | **434.75 ± 0.14** | 0.01× |
| PlankianJitter | **21.29 ± 0.67** | 10.85 ± 0.01 | N/A | 1.96× |
| PlasmaBrightness | 3.37 ± 0.03 | **16.94 ± 0.36** | N/A | 0.20× |
| PlasmaContrast | 2.64 ± 0.01 | **16.97 ± 0.03** | N/A | 0.16× |
| PlasmaShadow | 6.08 ± 0.05 | **19.03 ± 0.50** | N/A | 0.32× |
| Posterize | 56.50 ± 2.44 | N/A | **631.46 ± 14.74** | 0.09× |
| RGBShift | **31.73 ± 0.71** | 22.27 ± 0.04 | N/A | 1.42× |
| Rain | **23.09 ± 1.52** | 3.77 ± 0.00 | N/A | 6.12× |
| RandomCrop128 | 695.33 ± 29.37 | 65.33 ± 0.35 | **1132.79 ± 15.23** | 0.61× |
| RandomGamma | **183.49 ± 6.45** | 21.63 ± 0.02 | N/A | 8.48× |
| RandomResizedCrop | 15.48 ± 1.12 | 6.29 ± 0.03 | **182.09 ± 15.75** | 0.09× |
| Resize | 15.67 ± 0.49 | 5.87 ± 0.03 | **139.96 ± 35.04** | 0.11× |
| Rotate | 28.62 ± 0.76 | 21.53 ± 0.05 | **534.18 ± 0.16** | 0.05× |
| SaltAndPepper | **9.88 ± 0.19** | 8.82 ± 0.12 | N/A | 1.12× |
| Saturation | 8.42 ± 0.14 | **36.56 ± 0.12** | N/A | 0.23× |
| Sharpen | 25.02 ± 0.30 | 17.86 ± 0.03 | **420.09 ± 8.99** | 0.06× |
| Shear | **4.41 ± 0.08** | N/A | N/A | N/A |
| Snow | **12.72 ± 0.21** | N/A | N/A | N/A |
| Solarize | 52.02 ± 1.45 | 20.73 ± 0.02 | **628.42 ± 5.91** | 0.08× |
| ThinPlateSpline | 4.30 ± 0.14 | **44.90 ± 0.67** | N/A | 0.10× |
| VerticalFlip | 9.57 ± 0.27 | 21.96 ± 0.24 | **977.92 ± 5.22** | 0.01× |

## Torchvision Metadata

```
system_info:
  python_version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0]
  platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.31
  processor: x86_64
  cpu_count: 64
  timestamp: 2025-03-11T11:14:57.765540+00:00
library_versions:
  torchvision: 0.21.0
  numpy: 2.2.3
  pillow: 11.1.0
  opencv-python-headless: not installed
  torch: 2.6.0
  opencv-python: not installed
thread_settings:
  environment: {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
  opencv: not installed
  pytorch: {'threads': 32, 'gpu_available': True, 'gpu_device': 0, 'gpu_name': 'NVIDIA GeForce RTX 4090', 'gpu_memory_total': 23.55084228515625, 'gpu_memory_allocated': 15.05643081665039}
  pillow: {'threads': 'unknown', 'simd': False}
benchmark_params:
  num_videos: 200
  num_runs: 10
  max_warmup_iterations: 100
  warmup_window: 5
  warmup_threshold: 0.05
  min_warmup_windows: 3
precision: torch.float16
```

## Albumentations Metadata

```
system_info:
  python_version: 3.12.8 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 10:37:40) [Clang 14.0.6 ]
  platform: macOS-15.1-arm64-arm-64bit
  processor: arm
  cpu_count: 16
  timestamp: 2025-03-11T01:57:36.320659+00:00
library_versions:
  albumentations: 2.0.5
  numpy: 2.2.3
  pillow: 11.1.0
  opencv-python-headless: 4.11.0.86
  torch: 2.6.0
  opencv-python: not installed
thread_settings:
  environment: {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
  opencv: {'threads': 1, 'opencl': False}
  pytorch: {'threads': 1, 'gpu_available': False, 'gpu_device': None}
  pillow: {'threads': 'unknown', 'simd': False}
benchmark_params:
  num_videos: 200
  num_runs: 5
  max_warmup_iterations: 100
  warmup_window: 5
  warmup_threshold: 0.05
  min_warmup_windows: 3
```

## Kornia Metadata

```
system_info:
  python_version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0]
  platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.31
  processor: x86_64
  cpu_count: 64
  timestamp: 2025-03-11T00:46:14.791885+00:00
library_versions:
  kornia: 0.8.0
  numpy: 2.2.3
  pillow: 11.1.0
  opencv-python-headless: not installed
  torch: 2.6.0
  opencv-python: not installed
thread_settings:
  environment: {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
  opencv: not installed
  pytorch: {'threads': 32, 'gpu_available': True, 'gpu_device': 0, 'gpu_name': 'NVIDIA GeForce RTX 4090', 'gpu_memory_total': 23.55084228515625, 'gpu_memory_allocated': 15.05643081665039}
  pillow: {'threads': 'unknown', 'simd': False}
benchmark_params:
  num_videos: 200
  num_runs: 5
  max_warmup_iterations: 100
  warmup_window: 5
  warmup_threshold: 0.05
  min_warmup_windows: 3
precision: torch.float16
```
