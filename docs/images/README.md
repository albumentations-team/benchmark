<!-- This file is auto-generated. Do not edit directly. -->

# Image Augmentation Benchmarks

This directory contains benchmark results for image augmentation libraries.

## Overview

The image benchmarks measure the performance of various image augmentation libraries on standard image
transformations. The benchmarks are run on a single CPU thread to ensure consistent and comparable results.

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

### Image Benchmark Results

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

- albumentations: 2.0.7
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.8.0
- torchvision: 0.20.1

## Performance Comparison

Number shows how many uint8 images per second can be processed on one CPU thread. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

| Transform            | albumentations<br>2.0.7   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.8.0   | torchvision<br>0.20.1   | Speedup<br>(Alb/fastest other)   |
|:---------------------|:--------------------------|:-----------------|:------------------|:------------------|:------------------------|:---------------------------------|
| Affine               | **1382 ± 20**             | -                | 1328 ± 16         | 248 ± 6           | 188 ± 2                 | 1.04x                            |
| AutoContrast         | **1668 ± 23**             | -                | -                 | 541 ± 8           | 344 ± 1                 | 3.08x                            |
| Blur                 | **8194 ± 445**            | 386 ± 4          | 5381 ± 125        | 265 ± 11          | -                       | 1.52x                            |
| Brightness           | **13679 ± 481**           | 2108 ± 32        | 1076 ± 32         | 1127 ± 27         | 854 ± 13                | 6.49x                            |
| CLAHE                | **646 ± 3**               | -                | 555 ± 14          | 165 ± 3           | -                       | 1.16x                            |
| CenterCrop128        | **117247 ± 2368**         | -                | -                 | -                 | -                       | N/A                              |
| ChannelDropout       | **10355 ± 408**           | -                | -                 | 2283 ± 24         | -                       | 4.54x                            |
| ChannelShuffle       | **7841 ± 130**            | -                | 1252 ± 26         | 1328 ± 44         | 4417 ± 234              | 1.78x                            |
| CoarseDropout        | **22683 ± 1287**          | -                | 1190 ± 22         | -                 | -                       | 19.06x                           |
| ColorJitter          | **1002 ± 13**             | 418 ± 5          | -                 | 104 ± 4           | 87 ± 1                  | 2.39x                            |
| Contrast             | **12727 ± 634**           | 1379 ± 25        | 717 ± 5           | 1109 ± 41         | 602 ± 13                | 9.23x                            |
| CornerIllumination   | **483 ± 12**              | -                | -                 | 452 ± 3           | -                       | 1.07x                            |
| Elastic              | 303 ± 4                   | -                | **395 ± 14**      | 1 ± 0             | 3 ± 0                   | 0.77x                            |
| Equalize             | **1264 ± 7**              | -                | 814 ± 11          | 306 ± 1           | 795 ± 3                 | 1.55x                            |
| Erasing              | **31095 ± 4087**          | -                | -                 | 1210 ± 27         | 3577 ± 49               | 8.69x                            |
| GaussianBlur         | **2450 ± 7**              | 387 ± 4          | 1460 ± 23         | 254 ± 5           | 127 ± 4                 | 1.68x                            |
| GaussianIllumination | **727 ± 13**              | -                | -                 | 436 ± 13          | -                       | 1.67x                            |
| GaussianNoise        | **347 ± 7**               | -                | 263 ± 9           | 125 ± 1           | -                       | 1.32x                            |
| Grayscale            | **34262 ± 290**           | 6088 ± 107       | 3100 ± 24         | 1201 ± 52         | 2600 ± 23               | 5.63x                            |
| HSV                  | **1120 ± 50**             | -                | -                 | -                 | -                       | N/A                              |
| HorizontalFlip       | **14043 ± 474**           | 8808 ± 1012      | 9599 ± 495        | 1297 ± 13         | 2486 ± 107              | 1.46x                            |
| Hue                  | **1877 ± 59**             | -                | -                 | 150 ± 1           | -                       | 12.54x                           |
| Invert               | **35660 ± 5346**          | -                | 3682 ± 79         | 2881 ± 43         | 4244 ± 30               | 8.40x                            |
| JpegCompression      | **1349 ± 7**              | 1202 ± 19        | 687 ± 26          | 120 ± 1           | 889 ± 7                 | 1.12x                            |
| LinearIllumination   | 499 ± 15                  | -                | -                 | **708 ± 6**       | -                       | 0.71x                            |
| MedianBlur           | **1187 ± 27**             | -                | 1152 ± 14         | 6 ± 0             | -                       | 1.03x                            |
| MotionBlur           | **4469 ± 137**            | -                | 928 ± 37          | 159 ± 1           | -                       | 4.82x                            |
| Normalize            | **1929 ± 34**             | -                | -                 | 1251 ± 14         | 1018 ± 7                | 1.54x                            |
| OpticalDistortion    | **646 ± 7**               | -                | -                 | 174 ± 0           | -                       | 3.71x                            |
| Pad                  | **47085 ± 641**           | -                | -                 | -                 | 4889 ± 183              | 9.63x                            |
| Perspective          | **1142 ± 19**             | -                | 908 ± 8           | 154 ± 3           | 147 ± 5                 | 1.26x                            |
| PlankianJitter       | **3151 ± 64**             | -                | -                 | 2150 ± 52         | -                       | 1.47x                            |
| PlasmaBrightness     | **179 ± 4**               | -                | -                 | 85 ± 1            | -                       | 2.10x                            |
| PlasmaContrast       | **154 ± 4**               | -                | -                 | 84 ± 0            | -                       | 1.83x                            |
| PlasmaShadow         | 192 ± 4                   | -                | -                 | **216 ± 5**       | -                       | 0.89x                            |
| Posterize            | **13939 ± 738**           | -                | 3111 ± 95         | 836 ± 30          | 4247 ± 26               | 3.28x                            |
| RGBShift             | **3406 ± 34**             | -                | -                 | 896 ± 9           | -                       | 3.80x                            |
| Rain                 | **2016 ± 24**             | -                | -                 | 1493 ± 9          | -                       | 1.35x                            |
| RandomCrop128        | **112466 ± 1804**         | 45395 ± 934      | 21408 ± 622       | 2946 ± 42         | 31450 ± 249             | 2.48x                            |
| RandomGamma          | **14036 ± 847**           | -                | 3504 ± 72         | 230 ± 3           | -                       | 4.01x                            |
| RandomResizedCrop    | **4123 ± 125**            | -                | -                 | 661 ± 16          | 837 ± 37                | 4.93x                            |
| Resize               | **3454 ± 68**             | 1083 ± 21        | 2995 ± 70         | 645 ± 13          | 260 ± 9                 | 1.15x                            |
| Rotate               | **2816 ± 119**            | 1739 ± 105       | 2574 ± 10         | 256 ± 2           | 258 ± 4                 | 1.09x                            |
| SaltAndPepper        | **613 ± 3**               | -                | -                 | 480 ± 12          | -                       | 1.28x                            |
| Saturation           | **1337 ± 21**             | -                | 495 ± 3           | 155 ± 2           | -                       | 2.70x                            |
| Sharpen              | **2307 ± 20**             | -                | 1101 ± 30         | 201 ± 2           | 220 ± 3                 | 2.09x                            |
| Shear                | **1270 ± 4**              | -                | 1244 ± 14         | 261 ± 1           | -                       | 1.02x                            |
| Snow                 | **781 ± 27**              | -                | -                 | 143 ± 1           | -                       | 5.46x                            |
| Solarize             | **12048 ± 541**           | -                | 3843 ± 80         | 263 ± 6           | 1032 ± 14               | 3.14x                            |
| ThinPlateSpline      | **79 ± 2**                | -                | -                 | 58 ± 0            | -                       | 1.36x                            |
| VerticalFlip         | **31809 ± 178**           | 16830 ± 1653     | 19935 ± 1708      | 2872 ± 37         | 4696 ± 161              | 1.60x                            |

## Analysis

The benchmark results show that Albumentations is generally the fastest library for most image
transformations. This is due to its optimized implementation and use of OpenCV for many operations.

Some key observations:
- Albumentations is particularly fast for geometric transformations like resize, rotate, and affine
- For some specialized transformations, other libraries may be faster
- The performance gap is most significant for complex transformations

## Recommendations

Based on the benchmark results, we recommend:

1. Use Albumentations for production workloads where performance is critical
2. Consider the specific transformations you need and check their relative performance
3. For GPU acceleration, consider Kornia, especially for batch processing
