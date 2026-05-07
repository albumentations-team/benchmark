# Generated Paper Data

Generated from local `gcp_runs/prod-*` artifacts. Throughput units are images/second.

## Coverage Summary

| Regime | Library | Rows | Full runs | Early-stopped | Unsupported | Median of measured rows (img/s) |
|---|---|---:|---:|---:|---:|---:|
| CPU micro | albumentationsx | 57 | 57 | 0 | 0 | 1387.6 |
| CPU micro | torchvision | 26 | 25 | 1 | 0 | 545.4 |
| CPU micro | kornia | 51 | 47 | 4 | 0 | 231.3 |
| CPU micro | pillow | 24 | 23 | 1 | 0 | 609.3 |
| GPU micro | torchvision | 25 | 24 | 1 | 0 | 3914.6 |
| GPU micro | kornia | 50 | 42 | 3 | 5 | 393.2 |
| CPU DataLoader | albumentationsx | 57 | 57 | 0 | 0 | 4614.8 |
| CPU DataLoader | torchvision | 26 | 26 | 0 | 0 | 3322.6 |
| CPU DataLoader | kornia | 51 | 51 | 0 | 0 | 1441.0 |
| CPU DataLoader | pillow | 26 | 25 | 1 | 0 | 3241.2 |
| GPU DataLoader | torchvision | 25 | 25 | 0 | 0 | 2561.9 |
| GPU DataLoader | kornia | 50 | 49 | 0 | 1 | 632.3 |
| GPU DataLoader | dali | 57 | 22 | 0 | 35 | 3785.2 |

## Absolute/Open Production DataLoader Category

This is the open production DataLoader category: CPU and GPU DataLoader rows compete together over the 57-recipe universe. Microbenchmarks remain separate.

| Implementation | Regime | Full measured / 57 | Median measured-row throughput (img/s) | Open-category wins |
|---|---|---:|---:|---:|
| AlbumentationsX CPU | CPU DataLoader | 57/57 | 4614.8 | 52 |
| DALI GPU | GPU DataLoader | 22/57 | 3785.2 | 3 |
| TorchVision CPU | CPU DataLoader | 26/57 | 3322.6 | 0 |
| Pillow CPU | CPU DataLoader | 25/57 | 3241.2 | 0 |
| TorchVision GPU | GPU DataLoader | 25/57 | 2561.9 | 1 |
| Kornia CPU | CPU DataLoader | 51/57 | 1441.0 | 1 |
| Kornia GPU | GPU DataLoader | 49/57 | 632.3 | 0 |

Open-category exception rows: DALI GPU wins CLAHE, GaussianNoise, and Resize; Kornia CPU wins ThinPlateSpline; TorchVision GPU wins RandomCrop224.

## Winner Counts

Measured winner counts for CPU micro:

| Library | Wins |
|---|---:|
| albumentationsx | 52 |
| pillow | 1 |

Largest measured winner gaps:

| Transform | Winner | Gap over second | Second |
|---|---|---:|---|
| RandomGamma | albumentationsx | 32.3x | kornia |
| MotionBlur | albumentationsx | 24.1x | kornia |
| RandomJigsaw | albumentationsx | 23.7x | kornia |
| Sharpen | albumentationsx | 18.5x | torchvision |
| RandomRotate90 | albumentationsx | 18.0x | kornia |
| Hue | albumentationsx | 14.8x | kornia |
| GaussianBlur | albumentationsx | 13.9x | pillow |
| ColorJiggle | albumentationsx | 13.5x | torchvision |
| PhotoMetricDistort | albumentationsx | 12.8x | torchvision |
| ColorJitter | albumentationsx | 12.4x | kornia |

Measured winner counts for CPU DataLoader:

| Library | Wins |
|---|---:|
| albumentationsx | 56 |
| kornia | 1 |

Largest measured winner gaps:

| Transform | Winner | Gap over second | Second |
|---|---|---:|---|
| RandomCrop224+MedianBlur+Normalize+ToTensor | albumentationsx | 24.5x | pillow |
| RandomCrop224+Elastic+Normalize+ToTensor | albumentationsx | 12.7x | torchvision |
| RandomCrop224+PlasmaBrightness+Normalize+ToTensor | albumentationsx | 6.1x | kornia |
| RandomCrop224+PlasmaContrast+Normalize+ToTensor | albumentationsx | 4.9x | kornia |
| RandomCrop224+CLAHE+Normalize+ToTensor | albumentationsx | 4.4x | kornia |
| RandomCrop224+Hue+Normalize+ToTensor | albumentationsx | 4.3x | kornia |
| RandomCrop224+Snow+Normalize+ToTensor | albumentationsx | 4.0x | kornia |
| RandomCrop224+MotionBlur+Normalize+ToTensor | albumentationsx | 3.9x | kornia |
| RandomCrop224+RandomRotate90+Normalize+ToTensor | albumentationsx | 3.5x | kornia |
| RandomCrop224+PhotoMetricDistort+Normalize+ToTensor | albumentationsx | 3.5x | torchvision |

Measured winner counts for GPU micro:

| Library | Wins |
|---|---:|
| torchvision | 19 |

Largest measured winner gaps:

| Transform | Winner | Gap over second | Second |
|---|---|---:|---|
| RandomCrop224 | torchvision | 48.8x | kornia |
| Posterize | torchvision | 26.8x | kornia |
| Resize | torchvision | 23.7x | kornia |
| Invert | torchvision | 20.6x | kornia |
| VerticalFlip | torchvision | 19.2x | kornia |
| HorizontalFlip | torchvision | 18.9x | kornia |
| Solarize | torchvision | 18.4x | kornia |
| ChannelShuffle | torchvision | 13.5x | kornia |
| RandomResizedCrop | torchvision | 13.2x | kornia |
| Grayscale | torchvision | 12.9x | kornia |

Measured winner counts for GPU DataLoader:

| Library | Wins |
|---|---:|
| dali | 16 |
| torchvision | 13 |
| kornia | 1 |

Largest measured winner gaps:

| Transform | Winner | Gap over second | Second |
|---|---|---:|---|
| RandomCrop224+CLAHE+Normalize+ToTensor | dali | 22.5x | kornia |
| RandomCrop224+SaltAndPepper+Normalize+ToTensor | dali | 10.2x | kornia |
| RandomCrop224+Posterize+Normalize+ToTensor | torchvision | 8.4x | kornia |
| RandomCrop224+Invert+Normalize+ToTensor | torchvision | 7.4x | kornia |
| RandomCrop224+Hue+Normalize+ToTensor | dali | 6.5x | kornia |
| RandomCrop224+Saturation+Normalize+ToTensor | dali | 6.5x | kornia |
| RandomCrop224+JpegCompression+Normalize+ToTensor | dali | 6.4x | kornia |
| RandomCrop224+Solarize+Normalize+ToTensor | torchvision | 6.4x | kornia |
| RandomCrop224+ColorJitter+Normalize+ToTensor | dali | 6.1x | torchvision |
| RandomCrop224+Grayscale+Normalize+ToTensor | torchvision | 6.0x | kornia |

## CPU DataLoader vs GPU DataLoader

| Library | Compared rows | GPU faster | CPU faster/equal | Median GPU/CPU ratio |
|---|---:|---:|---:|---:|
| torchvision | 25 | 13 | 12 | 1.00x |
| kornia | 49 | 4 | 45 | 0.42x |
| dali | 0 | 0 | 0 | -x |

Largest GPU wins:

| Library | Transform | CPU img/s | GPU img/s | GPU/CPU |
|---|---|---:|---:|---:|
| kornia | RandomCrop224+MedianBlur+Normalize+ToTensor | 87.6 | 327.0 | 3.73x |
| kornia | RandomCrop224+Elastic+Normalize+ToTensor | 102.6 | 316.3 | 3.08x |
| torchvision | RandomCrop224+Resize+Normalize+ToTensor | 1225.1 | 2829.3 | 2.31x |
| torchvision | RandomCrop224+Sharpen+Normalize+ToTensor | 1401.0 | 2303.8 | 1.64x |
| torchvision | RandomCrop224+Normalize+ToTensor | 3939.9 | 6252.6 | 1.59x |
| torchvision | RandomCrop224+HorizontalFlip+Normalize+ToTensor | 3760.4 | 5050.4 | 1.34x |
| torchvision | RandomCrop224+GaussianBlur+Normalize+ToTensor | 1458.1 | 1957.2 | 1.34x |
| torchvision | RandomCrop224+VerticalFlip+Normalize+ToTensor | 3828.0 | 5020.2 | 1.31x |
| torchvision | RandomCrop224+Invert+Normalize+ToTensor | 3794.0 | 4868.1 | 1.28x |
| kornia | RandomCrop224+PlasmaBrightness+Normalize+ToTensor | 439.7 | 561.6 | 1.28x |

Largest CPU wins:

| Library | Transform | CPU img/s | GPU img/s | GPU/CPU |
|---|---|---:|---:|---:|
| kornia | RandomCrop224+Rain+Normalize+ToTensor | 1479.3 | 306.2 | 0.21x |
| kornia | RandomCrop224+CLAHE+Normalize+ToTensor | 761.4 | 165.5 | 0.22x |
| kornia | RandomCrop224+Equalize+Normalize+ToTensor | 1268.2 | 320.9 | 0.25x |
| kornia | RandomCrop224+SaltAndPepper+Normalize+ToTensor | 1429.0 | 375.3 | 0.26x |
| kornia | RandomCrop224+CornerIllumination+Normalize+ToTensor | 1441.0 | 394.7 | 0.27x |
| kornia | RandomCrop224+LongestMaxSize+Normalize+ToTensor | 628.1 | 178.7 | 0.28x |
| torchvision | RandomCrop224+Perspective+Normalize+ToTensor | 2549.8 | 792.1 | 0.31x |
| kornia | RandomCrop224+LinearIllumination+Normalize+ToTensor | 1591.9 | 503.2 | 0.32x |
| kornia | RandomCrop224+SmallestMaxSize+Normalize+ToTensor | 553.6 | 179.0 | 0.32x |
| kornia | RandomCrop224+Resize+Normalize+ToTensor | 538.3 | 181.2 | 0.34x |

## AlbumentationsX CPU DataLoader vs GPU DataLoader

| GPU library | Compared rows | GPU faster than AlbumentationsX CPU | AlbumentationsX CPU faster/equal | Median GPU/AlbumentationsX ratio |
|---|---:|---:|---:|---:|
| torchvision | 25 | 2 | 23 | 0.55x |
| kornia | 49 | 0 | 49 | 0.13x |
| dali | 22 | 3 | 19 | 0.82x |

## GPU Memory

| Library | Transform | Throughput img/s | Peak allocated MB | Peak reserved MB | Status |
|---|---|---:|---:|---:|---|
| kornia | RandomCrop224+MedianBlur+Normalize+ToTensor | 327.0 | 4281.4 | 7202.0 | ok |
| dali | RandomCrop224+HorizontalFlip+Normalize+ToTensor | 3722.4 | 3112.0 | 3112.0 | ok |
| dali | RandomCrop224+Rotate+Normalize+ToTensor | 3808.5 | 3112.0 | 3112.0 | ok |
| dali | RandomCrop224+Shear+Normalize+ToTensor | 3771.7 | 3112.0 | 3112.0 | ok |
| dali | RandomCrop224+Brightness+Normalize+ToTensor | 3797.8 | 3096.0 | 3096.0 | ok |
| dali | RandomCrop224+GaussianNoise+Normalize+ToTensor | 3820.0 | 3096.0 | 3096.0 | ok |
| dali | RandomCrop224+VerticalFlip+Normalize+ToTensor | 3766.2 | 3096.0 | 3096.0 | ok |
| dali | RandomCrop224+ColorJitter+Normalize+ToTensor | 3817.8 | 3080.0 | 3080.0 | ok |
| dali | RandomCrop224+Equalize+Normalize+ToTensor | 3823.9 | 3080.0 | 3080.0 | ok |
| dali | RandomCrop224+Erasing+Normalize+ToTensor | 3791.4 | 3064.0 | 3064.0 | ok |
| dali | RandomCrop224+Pad+Normalize+ToTensor | 3756.1 | 3064.0 | 3064.0 | ok |
| dali | RandomCrop224+Contrast+Normalize+ToTensor | 3759.3 | 3048.0 | 3048.0 | ok |
| dali | RandomCrop224+GaussianBlur+Normalize+ToTensor | 3700.9 | 3048.0 | 3048.0 | ok |
| dali | RandomCrop224+Hue+Normalize+ToTensor | 3784.2 | 3032.0 | 3032.0 | ok |
| dali | RandomCrop224+JpegCompression+Normalize+ToTensor | 3786.2 | 3032.0 | 3032.0 | ok |
| dali | RandomCrop224+Normalize+ToTensor | 3589.1 | 3032.0 | 3032.0 | ok |
| dali | RandomCrop224+SaltAndPepper+Normalize+ToTensor | 3824.4 | 3032.0 | 3032.0 | ok |
| dali | RandomCrop224+CLAHE+Normalize+ToTensor | 3730.6 | 3016.0 | 3016.0 | ok |
| dali | RandomCrop224+ColorJiggle+Normalize+ToTensor | 3742.5 | 3016.0 | 3016.0 | ok |
| dali | RandomCrop224+Resize+Normalize+ToTensor | 3523.1 | 3016.0 | 3016.0 | ok |

## Unsupported And Early-Stopped Summary

| Regime | Library | Full measured | Early-stopped | Unsupported |
|---|---|---:|---:|---:|
| CPU micro | albumentationsx | 57 | 0 | 0 |
| CPU micro | torchvision | 25 | 1 | 0 |
| CPU micro | kornia | 47 | 4 | 0 |
| CPU micro | pillow | 23 | 1 | 0 |
| GPU micro | torchvision | 24 | 1 | 0 |
| GPU micro | kornia | 42 | 3 | 5 |
| CPU DataLoader | albumentationsx | 57 | 0 | 0 |
| CPU DataLoader | torchvision | 26 | 0 | 0 |
| CPU DataLoader | kornia | 51 | 0 | 0 |
| CPU DataLoader | pillow | 25 | 1 | 0 |
| GPU DataLoader | torchvision | 25 | 0 | 0 |
| GPU DataLoader | kornia | 49 | 0 | 1 |
| GPU DataLoader | dali | 22 | 0 | 35 |

Full row-level reasons are generated in the public paper-data supplement after figure generation.
