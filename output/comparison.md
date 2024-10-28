# Benchmark Results

### System Information

- Platform: macOS-15.0.1-arm64-arm-64bit
- Processor: arm
- CPU Count: 10
- Python Version: 3.12.7

### Benchmark Parameters

- Number of images: 1000
- Runs per transform: 10
- Max warmup iterations: 1000


### Library Versions

- albumentations: 1.4.20
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.7.3
- torchvision: 0.20.0

## Performance Comparison

| Transform         | albumentations<br>1.4.20   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.7.3   | torchvision<br>0.20.0   |
|:------------------|:---------------------------|:-----------------|:------------------|:------------------|:------------------------|
| HorizontalFlip    | **8325 ± 955**             | 4807 ± 818       | 6042 ± 788        | 390 ± 106         | 914 ± 67                |
| VerticalFlip      | **20493 ± 1134**           | 9153 ± 1291      | 10931 ± 1844      | 1212 ± 402        | 3198 ± 200              |
| Rotate            | **1272 ± 12**              | 1119 ± 41        | 1136 ± 218        | 143 ± 11          | 181 ± 11                |
| Affine            | **967 ± 3**                | -                | 774 ± 97          | 147 ± 9           | 130 ± 12                |
| Equalize          | **961 ± 4**                | -                | 581 ± 54          | 152 ± 19          | 479 ± 12                |
| RandomCrop80      | **118946 ± 741**           | 25272 ± 1822     | 11503 ± 441       | 1510 ± 230        | 32109 ± 1241            |
| ShiftRGB          | **1873 ± 252**             | -                | 1582 ± 65         | -                 | -                       |
| Resize            | **2365 ± 153**             | 611 ± 78         | 1806 ± 63         | 232 ± 24          | 195 ± 4                 |
| RandomGamma       | **8608 ± 220**             | -                | 2318 ± 269        | 108 ± 13          | -                       |
| Grayscale         | **3050 ± 597**             | 2720 ± 932       | 1681 ± 156        | 289 ± 75          | 1838 ± 130              |
| RandomPerspective | 410 ± 20                   | -                | **554 ± 22**      | 86 ± 11           | 96 ± 5                  |
| GaussianBlur      | **1734 ± 204**             | 242 ± 4          | 1090 ± 65         | 176 ± 18          | 79 ± 3                  |
| MedianBlur        | **862 ± 30**               | -                | 813 ± 30          | 5 ± 0             | -                       |
| MotionBlur        | **2975 ± 52**              | -                | 612 ± 18          | 73 ± 2            | -                       |
| Posterize         | **5214 ± 101**             | -                | 2097 ± 68         | 430 ± 49          | 3196 ± 185              |
| JpegCompression   | **845 ± 61**               | 778 ± 5          | 459 ± 35          | 71 ± 3            | 625 ± 17                |
| GaussianNoise     | 147 ± 10                   | 67 ± 2           | **206 ± 11**      | 75 ± 1            | -                       |
| Elastic           | 171 ± 15                   | -                | **235 ± 20**      | 1 ± 0             | 2 ± 0                   |
| Clahe             | **423 ± 10**               | -                | 335 ± 43          | 94 ± 9            | -                       |
| CoarseDropout     | **11288 ± 609**            | -                | 671 ± 38          | 536 ± 87          | -                       |
| Blur              | **4816 ± 59**              | 246 ± 3          | 3807 ± 325        | -                 | -                       |
| ColorJitter       | **536 ± 41**               | 255 ± 13         | -                 | 55 ± 18           | 46 ± 2                  |
| Brightness        | **4443 ± 84**              | 1163 ± 86        | -                 | 472 ± 101         | 429 ± 20                |
| Contrast          | **4398 ± 143**             | 736 ± 79         | -                 | 425 ± 52          | 335 ± 35                |
| RandomResizedCrop | **2952 ± 24**              | -                | -                 | 287 ± 58          | 511 ± 10                |
| Normalize         | **1016 ± 84**              | -                | -                 | 626 ± 40          | 519 ± 12                |
| PlankianJitter    | **1844 ± 208**             | -                | -                 | 813 ± 211         | -                       |
