# Benchmark Results

### System Information

- Platform: macOS-15.0.1-arm64-arm-64bit
- Processor: arm
- CPU Count: 10
- Python Version: 3.12.7

### Benchmark Parameters

- Number of images: 1000
- Runs per transform: 20
- Max warmup iterations: 1000


### Library Versions

- albumentations: 1.4.21
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.7.3
- torchvision: 0.20.1

## Performance Comparison

| Transform         | albumentations<br>1.4.21   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.7.3   | torchvision<br>0.20.1   |
|:------------------|:---------------------------|:-----------------|:------------------|:------------------|:------------------------|
| HorizontalFlip    | **8622 ± 891**             | 4121 ± 1001      | 6162 ± 393        | 571 ± 84          | 861 ± 38                |
| VerticalFlip      | **23951 ± 5013**           | 7775 ± 1376      | 11663 ± 2258      | 1590 ± 100        | 3156 ± 402              |
| Rotate            | 1163 ± 84                  | 1095 ± 82        | **1224 ± 75**     | 167 ± 11          | 160 ± 11                |
| Affine            | **907 ± 49**               | -                | 890 ± 34          | 181 ± 6           | 129 ± 16                |
| Equalize          | **852 ± 90**               | -                | 610 ± 35          | 184 ± 9           | 416 ± 44                |
| RandomCrop80      | **107764 ± 3630**          | 25192 ± 5964     | 12343 ± 2013      | 1492 ± 22         | 28767 ± 858             |
| ShiftRGB          | **2351 ± 276**             | -                | 1674 ± 63         | -                 | -                       |
| Resize            | **2372 ± 156**             | 632 ± 38         | 2025 ± 74         | 332 ± 18          | 180 ± 11                |
| RandomGamma       | **9014 ± 371**             | -                | 2592 ± 143        | 128 ± 10          | -                       |
| Grayscale         | **11373 ± 923**            | 3359 ± 65        | 1849 ± 75         | 628 ± 75          | 1497 ± 317              |
| RandomPerspective | 401 ± 24                   | -                | **596 ± 38**      | 98 ± 7            | 106 ± 4                 |
| GaussianBlur      | **1664 ± 144**             | 235 ± 18         | 1043 ± 142        | 165 ± 12          | 82 ± 3                  |
| MedianBlur        | 847 ± 36                   | -                | **849 ± 32**      | 4 ± 0             | -                       |
| MotionBlur        | **3928 ± 742**             | -                | 663 ± 36          | 75 ± 6            | -                       |
| Posterize         | **9034 ± 331**             | -                | 2400 ± 142        | 363 ± 69          | 3052 ± 380              |
| JpegCompression   | **906 ± 40**               | 754 ± 14         | 443 ± 79          | 69 ± 3            | 606 ± 42                |
| GaussianNoise     | 183 ± 5                    | 70 ± 1           | **204 ± 18**      | 65 ± 2            | -                       |
| Elastic           | 229 ± 17                   | -                | **251 ± 22**      | 1 ± 0             | 3 ± 0                   |
| Clahe             | **471 ± 18**               | -                | 422 ± 12          | 90 ± 2            | -                       |
| Brightness        | **9251 ± 709**             | 1297 ± 42        | 742 ± 39          | 519 ± 15          | 449 ± 14                |
| Contrast          | **9146 ± 1034**            | 880 ± 9          | 510 ± 9           | 476 ± 116         | 358 ± 4                 |
| CoarseDropout     | **14488 ± 2108**           | -                | 653 ± 85          | 526 ± 86          | -                       |
| Blur              | **5804 ± 305**             | 243 ± 9          | 3857 ± 385        | -                 | -                       |
| ColorJitter       | **700 ± 31**               | 252 ± 15         | -                 | 50 ± 4            | 47 ± 2                  |
| RandomResizedCrop | **2879 ± 158**             | -                | -                 | 321 ± 10          | 462 ± 47                |
| Normalize         | **1349 ± 65**              | -                | -                 | 645 ± 40          | 528 ± 20                |
| PlankianJitter    | **2155 ± 340**             | -                | -                 | 1023 ± 114        | -                       |
| HSV               | **1243 ± 44**              | -                | -                 | -                 | -                       |
