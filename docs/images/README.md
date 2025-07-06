<!-- This file is auto-generated. Do not edit directly. -->

# Image Augmentation Benchmarks

This directory contains benchmark results for image augmentation libraries.

## Overview

The image benchmarks measure the performance of various image augmentation libraries on standard image
transformations. The benchmarks are run on a single CPU thread to ensure consistent and comparable results.

## Methodology

1. **Image Loading**: Images are loaded using library-specific loaders to ensure optimal format compatibility:
   - OpenCV (BGR → RGB) for Albumentationsx and imgaug
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
./run_single.sh -l albumentationsx -d /path/to/images -o /path/to/output
```

Number shows how many uint8 images per second can be processed on one CPU thread. Larger is better.
The Speedup column shows how many times faster Albumentationsx is compared to the fastest other
library for each transform.

| Transform            | albumentationsx<br>N/A   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.8.1   | torchvision<br>0.22.1   | Speedup<br>(Albx/fastest other)   |
|:---------------------|:-------------------------|:-----------------|:------------------|:------------------|:------------------------|:----------------------------------|
| Affine               | **1390 ± 10**            | -                | 1258 ± 13         | 348 ± 1           | 264 ± 2                 | 1.10x                             |
| AutoContrast         | **1575 ± 28**            | -                | 573 ± 2           | 604 ± 10          | 355 ± 1                 | 2.61x                             |
| Blur                 | **7012 ± 100**           | 367 ± 1          | -                 | 484 ± 18          | -                       | 14.49x                            |
| Brightness           | **10810 ± 123**          | 1134 ± 3         | 1079 ± 8          | 1735 ± 76         | 1217 ± 9                | 6.23x                             |
| CLAHE                | 618 ± 5                  | -                | **622 ± 5**       | -                 | -                       | 0.99x                             |
| CenterCrop128        | 112797 ± 1437            | -                | 25992 ± 167       | -                 | **222025 ± 16620**      | 0.51x                             |
| ChannelDropout       | **10337 ± 440**          | -                | -                 | 4258 ± 204        | -                       | 2.43x                             |
| ChannelShuffle       | **7654 ± 109**           | -                | -                 | 1721 ± 86         | 4680 ± 42               | 1.64x                             |
| CoarseDropout        | **15498 ± 437**          | -                | 1286 ± 8          | -                 | -                       | 12.05x                            |
| ColorJitter          | **969 ± 19**             | 367 ± 4          | 257 ± 2           | 110 ± 2           | 93 ± 1                  | 2.64x                             |
| Contrast             | **14073 ± 1659**         | 861 ± 10         | 3722 ± 33         | 1770 ± 119        | 758 ± 7                 | 3.78x                             |
| CornerIllumination   | 453 ± 14                 | -                | -                 | **514 ± 8**       | -                       | 0.88x                             |
| Elastic              | **315 ± 4**              | -                | -                 | 2 ± 0             | 4 ± 0                   | 83.93x                            |
| Equalize             | **1207 ± 13**            | -                | 849 ± 5           | 306 ± 2           | 777 ± 8                 | 1.42x                             |
| Erasing              | **22757 ± 4357**         | -                | -                 | -                 | 10099 ± 439             | 2.25x                             |
| GaussianBlur         | **2398 ± 42**            | -                | 1499 ± 6          | 466 ± 4           | 172 ± 13                | 1.60x                             |
| GaussianIllumination | **676 ± 22**             | -                | -                 | 509 ± 23          | -                       | 1.33x                             |
| GaussianNoise        | **329 ± 4**              | -                | 304 ± 3           | 129 ± 1           | -                       | 1.08x                             |
| Grayscale            | **16484 ± 421**          | 4173 ± 133       | 3381 ± 92         | 1870 ± 14         | 2441 ± 57               | 3.95x                             |
| HorizontalFlip       | **13361 ± 379**          | 7691 ± 231       | 9726 ± 523        | 1239 ± 2          | 4521 ± 237              | 1.37x                             |
| Hue                  | **1776 ± 43**            | -                | -                 | 150 ± 1           | -                       | 11.80x                            |
| Invert               | **26603 ± 2449**         | -                | 4038 ± 54         | 6402 ± 213        | 20259 ± 2361            | 1.31x                             |
| JpegCompression      | **1271 ± 16**            | 1106 ± 5         | 746 ± 3           | 155 ± 2           | 862 ± 8                 | 1.15x                             |
| LinearIllumination   | 462 ± 10                 | -                | -                 | **897 ± 31**      | -                       | 0.51x                             |
| MedianBlur           | **1168 ± 26**            | -                | 1160 ± 2          | 6 ± 0             | -                       | 1.01x                             |
| MotionBlur           | **4251 ± 72**            | -                | 925 ± 9           | 138 ± 1           | -                       | 4.60x                             |
| Normalize            | **1478 ± 63**            | -                | -                 | 1340 ± 8          | 1159 ± 13               | 1.10x                             |
| OpticalDistortion    | **647 ± 4**              | -                | -                 | 212 ± 1           | -                       | 3.05x                             |
| Pad                  | **42601 ± 2081**         | 6690 ± 112       | 2181 ± 100        | -                 | 4676 ± 103              | 6.37x                             |
| Perspective          | **1176 ± 13**            | -                | 872 ± 15          | -                 | 214 ± 5                 | 1.35x                             |
| PlankianJitter       | **2978 ± 17**            | -                | -                 | 2092 ± 94         | -                       | 1.42x                             |
| PlasmaBrightness     | **172 ± 7**              | -                | -                 | 85 ± 1            | -                       | 2.03x                             |
| PlasmaContrast       | **150 ± 2**              | -                | -                 | 87 ± 2            | -                       | 1.73x                             |
| PlasmaShadow         | 193 ± 2                  | -                | -                 | **227 ± 3**       | -                       | 0.85x                             |
| Posterize            | 11663 ± 159              | 26 ± 0           | 3341 ± 20         | 984 ± 31          | **21346 ± 1713**        | 0.55x                             |
| RGBShift             | 985 ± 13                 | -                | -                 | **1283 ± 9**      | -                       | 0.77x                             |
| Rain                 | **2039 ± 18**            | -                | -                 | 2019 ± 28         | -                       | 1.01x                             |
| RandomCrop128        | 107961 ± 1828            | -                | 19938 ± 67        | 2897 ± 9          | **116051 ± 2134**       | 0.93x                             |
| RandomGamma          | **10811 ± 381**          | -                | 3818 ± 187        | 238 ± 1           | -                       | 2.83x                             |
| RandomResizedCrop    | **4285 ± 50**            | -                | -                 | 641 ± 5           | 860 ± 3                 | 4.98x                             |
| Resize               | **3433 ± 69**            | 1034 ± 2         | 2899 ± 77         | 645 ± 11          | 297 ± 2                 | 1.18x                             |
| Rotate               | **2775 ± 47**            | 2568 ± 26        | 1252 ± 5          | 327 ± 1           | 341 ± 10                | 1.08x                             |
| SaltAndPepper        | **604 ± 4**              | -                | -                 | 325 ± 2           | -                       | 1.86x                             |
| Saturation           | 1248 ± 37                | **1438 ± 44**    | -                 | 158 ± 5           | -                       | 0.87x                             |
| Sharpen              | **2222 ± 20**            | 606 ± 3          | 1177 ± 3          | 297 ± 11          | 327 ± 7                 | 1.89x                             |
| Shear                | **1254 ± 10**            | -                | -                 | 350 ± 2           | -                       | 3.58x                             |
| Solarize             | **10337 ± 168**          | -                | 4114 ± 11         | 585 ± 17          | 1338 ± 33               | 2.51x                             |
| ThinPlateSpline      | **85 ± 0**               | -                | -                 | 61 ± 0            | -                       | 1.39x                             |
| VerticalFlip         | 28911 ± 760              | 11477 ± 1149     | 20631 ± 2475      | 2857 ± 20         | **44757 ± 589**         | 0.65x                             |

## Analysis

The benchmark results show that Albumentationsx is generally the fastest library for most image
transformations. This is due to its optimized implementation and use of OpenCV for many operations.

Some key observations:
- Albumentationsx is particularly fast for geometric transformations like resize, rotate, and affine
- For some specialized transformations, other libraries may be faster
- The performance gap is most significant for complex transformations

## Recommendations

Based on the benchmark results, we recommend:

1. Use Albumentationsx for production workloads where performance is critical
2. Consider the specific transformations you need and check their relative performance
3. For GPU acceleration, consider Kornia, especially for batch processing
