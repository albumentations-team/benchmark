<!-- This file is auto-generated. Do not edit directly. -->

# Image Augmentation Benchmarks

This directory contains benchmark results for image augmentation libraries.

## Overview

The image benchmarks measure the performance of various image augmentation libraries on standard image transformations. The benchmarks are run on a single CPU thread to ensure consistent and comparable results.

## Methodology

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

## Running the Benchmarks

To run the image benchmarks:

```bash
./run_single.sh -l albumentations -d /path/to/images -o /path/to/output
```

To run all libraries and generate a comparison:

```bash
./run_all.sh -d /path/to/images -o /path/to/output
```

## Benchmark Results

<!-- BENCHMARK_RESULTS_START -->
<!-- This file is auto-generated. Do not edit directly. -->

# Image Benchmark Results

### System Information

- Platform: macOS-15.1-arm64-arm-64bit
- Processor: arm
- CPU Count: 16
- Python Version: 3.12.8

### Benchmark Parameters

- Number of images: 2000
- Runs per transform: 5
- Max warmup iterations: 1000


### Library Versions

- albumentations: 2.0.4
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.8.0
- torchvision: 0.20.1

## Performance Comparison

Number shows how many uint8 images per second can be processed on one CPU thread. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

| Transform            | albumentations<br>2.0.4   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.8.0   | torchvision<br>0.20.1   | Speedup<br>(Alb/fastest other)   |
|:---------------------|:--------------------------|:-----------------|:------------------|:------------------|:------------------------|:---------------------------------|
| Affine               | **1445 ± 9**              | -                | 1328 ± 16         | 248 ± 6           | 188 ± 2                 | 1.09x                            |
| AutoContrast         | **1657 ± 13**             | -                | -                 | 541 ± 8           | 344 ± 1                 | 3.06x                            |
| Blur                 | **7657 ± 114**            | 386 ± 4          | 5381 ± 125        | 265 ± 11          | -                       | 1.42x                            |
| Brightness           | **11985 ± 455**           | 2108 ± 32        | 1076 ± 32         | 1127 ± 27         | 854 ± 13                | 5.68x                            |
| CLAHE                | **647 ± 4**               | -                | 555 ± 14          | 165 ± 3           | -                       | 1.17x                            |
| CenterCrop128        | **119293 ± 2164**         | -                | -                 | -                 | -                       | N/A                              |
| ChannelDropout       | **11534 ± 306**           | -                | -                 | 2283 ± 24         | -                       | 5.05x                            |
| ChannelShuffle       | **6772 ± 109**            | -                | 1252 ± 26         | 1328 ± 44         | 4417 ± 234              | 1.53x                            |
| CoarseDropout        | **18962 ± 1346**          | -                | 1190 ± 22         | -                 | -                       | 15.93x                           |
| ColorJitter          | **1020 ± 91**             | 418 ± 5          | -                 | 104 ± 4           | 87 ± 1                  | 2.44x                            |
| Contrast             | **12394 ± 363**           | 1379 ± 25        | 717 ± 5           | 1109 ± 41         | 602 ± 13                | 8.99x                            |
| CornerIllumination   | **484 ± 7**               | -                | -                 | 452 ± 3           | -                       | 1.07x                            |
| Elastic              | 374 ± 2                   | -                | **395 ± 14**      | 1 ± 0             | 3 ± 0                   | 0.95x                            |
| Equalize             | **1236 ± 21**             | -                | 814 ± 11          | 306 ± 1           | 795 ± 3                 | 1.52x                            |
| Erasing              | **27451 ± 2794**          | -                | -                 | 1210 ± 27         | 3577 ± 49               | 7.67x                            |
| GaussianBlur         | **2350 ± 118**            | 387 ± 4          | 1460 ± 23         | 254 ± 5           | 127 ± 4                 | 1.61x                            |
| GaussianIllumination | **720 ± 7**               | -                | -                 | 436 ± 13          | -                       | 1.65x                            |
| GaussianNoise        | **315 ± 4**               | -                | 263 ± 9           | 125 ± 1           | -                       | 1.20x                            |
| Grayscale            | **32284 ± 1130**          | 6088 ± 107       | 3100 ± 24         | 1201 ± 52         | 2600 ± 23               | 5.30x                            |
| HSV                  | **1197 ± 23**             | -                | -                 | -                 | -                       | N/A                              |
| HorizontalFlip       | **14460 ± 368**           | 8808 ± 1012      | 9599 ± 495        | 1297 ± 13         | 2486 ± 107              | 1.51x                            |
| Hue                  | **1944 ± 64**             | -                | -                 | 150 ± 1           | -                       | 12.98x                           |
| Invert               | **27665 ± 3803**          | -                | 3682 ± 79         | 2881 ± 43         | 4244 ± 30               | 6.52x                            |
| JpegCompression      | **1321 ± 33**             | 1202 ± 19        | 687 ± 26          | 120 ± 1           | 889 ± 7                 | 1.10x                            |
| LinearIllumination   | 479 ± 5                   | -                | -                 | **708 ± 6**       | -                       | 0.68x                            |
| MedianBlur           | **1229 ± 9**              | -                | 1152 ± 14         | 6 ± 0             | -                       | 1.07x                            |
| MotionBlur           | **3521 ± 25**             | -                | 928 ± 37          | 159 ± 1           | -                       | 3.79x                            |
| Normalize            | **1819 ± 49**             | -                | -                 | 1251 ± 14         | 1018 ± 7                | 1.45x                            |
| OpticalDistortion    | **661 ± 7**               | -                | -                 | 174 ± 0           | -                       | 3.80x                            |
| Pad                  | **48589 ± 2059**          | -                | -                 | -                 | 4889 ± 183              | 9.94x                            |
| Perspective          | **1206 ± 3**              | -                | 908 ± 8           | 154 ± 3           | 147 ± 5                 | 1.33x                            |
| PlankianJitter       | **3221 ± 63**             | -                | -                 | 2150 ± 52         | -                       | 1.50x                            |
| PlasmaBrightness     | **168 ± 2**               | -                | -                 | 85 ± 1            | -                       | 1.98x                            |
| PlasmaContrast       | **145 ± 3**               | -                | -                 | 84 ± 0            | -                       | 1.71x                            |
| PlasmaShadow         | 183 ± 5                   | -                | -                 | **216 ± 5**       | -                       | 0.85x                            |
| Posterize            | **12979 ± 1121**          | -                | 3111 ± 95         | 836 ± 30          | 4247 ± 26               | 3.06x                            |
| RGBShift             | **3391 ± 104**            | -                | -                 | 896 ± 9           | -                       | 3.79x                            |
| Rain                 | **2043 ± 115**            | -                | -                 | 1493 ± 9          | -                       | 1.37x                            |
| RandomCrop128        | **111859 ± 1374**         | 45395 ± 934      | 21408 ± 622       | 2946 ± 42         | 31450 ± 249             | 2.46x                            |
| RandomGamma          | **12444 ± 753**           | -                | 3504 ± 72         | 230 ± 3           | -                       | 3.55x                            |
| RandomResizedCrop    | **4347 ± 37**             | -                | -                 | 661 ± 16          | 837 ± 37                | 5.19x                            |
| Resize               | **3532 ± 67**             | 1083 ± 21        | 2995 ± 70         | 645 ± 13          | 260 ± 9                 | 1.18x                            |
| Rotate               | **2912 ± 68**             | 1739 ± 105       | 2574 ± 10         | 256 ± 2           | 258 ± 4                 | 1.13x                            |
| SaltAndPepper        | **629 ± 6**               | -                | -                 | 480 ± 12          | -                       | 1.31x                            |
| Saturation           | **1596 ± 24**             | -                | 495 ± 3           | 155 ± 2           | -                       | 3.22x                            |
| Sharpen              | **2346 ± 10**             | -                | 1101 ± 30         | 201 ± 2           | 220 ± 3                 | 2.13x                            |
| Shear                | **1299 ± 11**             | -                | 1244 ± 14         | 261 ± 1           | -                       | 1.04x                            |
| Snow                 | **611 ± 9**               | -                | -                 | 143 ± 1           | -                       | 4.28x                            |
| Solarize             | **11756 ± 481**           | -                | 3843 ± 80         | 263 ± 6           | 1032 ± 14               | 3.06x                            |
| ThinPlateSpline      | **82 ± 1**                | -                | -                 | 58 ± 0            | -                       | 1.41x                            |
| VerticalFlip         | **32386 ± 936**           | 16830 ± 1653     | 19935 ± 1708      | 2872 ± 37         | 4696 ± 161              | 1.62x                            |

<!-- BENCHMARK_RESULTS_END -->

## Analysis

The benchmark results show that Albumentations is generally the fastest library for most image transformations. This is due to its optimized implementation and use of OpenCV for many operations.

Some key observations:
- Albumentations is particularly fast for geometric transformations like resize, rotate, and affine
- For some specialized transformations, other libraries may be faster
- The performance gap is most significant for complex transformations

## Recommendations

Based on the benchmark results, we recommend:

1. Use Albumentations for production workloads where performance is critical
2. Consider the specific transformations you need and check their relative performance
3. For GPU acceleration, consider Kornia, especially for batch processing
