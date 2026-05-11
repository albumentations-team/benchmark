# Unsupported And Early-Stopped Rows

This supplement table preserves the full unsupported and slow-row detail used by the paper.

| Regime | Library | Transform | Status | Reason |
|---|---|---|---|---|
| 9ch GPU DataLoader | kornia | RandomCrop224+GaussianIllumination+Normalize+ToTensor | unsupported | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| 9ch CPU micro | kornia | Affine | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | AutoContrast | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Blur | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Brightness | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | ChannelDropout | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | ChannelShuffle | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Contrast | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | CornerIllumination | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Elastic | early_stopped | Elastic slower than threshold: 0.288 sec/image >= 0.050 |
| 9ch CPU micro | kornia | Erasing | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | GaussianBlur | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | GaussianIllumination | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | GaussianNoise | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Grayscale | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | HorizontalFlip | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Invert | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | JpegCompression | early_stopped | JpegCompression slower than threshold: 0.066 sec/image >= 0.050 |
| 9ch CPU micro | kornia | LinearIllumination | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | LongestMaxSize | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | MedianBlur | early_stopped | MedianBlur slower than threshold: 1.183 sec/image >= 0.050 |
| 9ch CPU micro | kornia | MotionBlur | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | OpticalDistortion | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Perspective | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | PlasmaBrightness | early_stopped | PlasmaBrightness slower than threshold: 0.272 sec/image >= 0.050 |
| 9ch CPU micro | kornia | PlasmaContrast | early_stopped | PlasmaContrast slower than threshold: 0.253 sec/image >= 0.050 |
| 9ch CPU micro | kornia | PlasmaShadow | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Posterize | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | RandomCrop224 | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | RandomGamma | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | RandomJigsaw | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | RandomResizedCrop | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | RandomRotate90 | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Resize | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Rotate | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Sharpen | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Shear | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | SmallestMaxSize | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | Solarize | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | ThinPlateSpline | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | kornia | VerticalFlip | unsupported | RuntimeError: /root/benchmark-work/repo/.venv_torch_stack/bin/python failed with exit code -9 |
| 9ch CPU micro | torchvision | Elastic | early_stopped | Elastic slower than threshold: 0.124 sec/image >= 0.050 |
| 9ch GPU micro | kornia | CornerIllumination | unsupported | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| 9ch GPU micro | kornia | Erasing | unsupported | NotImplementedError: "check_uniform_bounds" not implemented for 'Long' |
| 9ch GPU micro | kornia | JpegCompression | early_stopped | JpegCompression slower than threshold: 0.072 sec/image >= 0.050 |
| 9ch GPU micro | kornia | LinearIllumination | unsupported | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| 9ch GPU micro | kornia | Perspective | unsupported | NotImplementedError: "check_uniform_bounds" not implemented for 'Long' |
| 9ch GPU micro | kornia | PlasmaBrightness | early_stopped | PlasmaBrightness slower than threshold: 0.158 sec/image >= 0.050 |
| 9ch GPU micro | kornia | PlasmaContrast | early_stopped | PlasmaContrast slower than threshold: 0.148 sec/image >= 0.050 |
| 9ch GPU micro | kornia | PlasmaShadow | early_stopped | PlasmaShadow slower than threshold: 0.149 sec/image >= 0.050 |
| 9ch GPU micro | torchvision | Elastic | early_stopped | Elastic slower than threshold: 0.181 sec/image >= 0.050 |
| CPU DataLoader | pillow | RandomCrop224+Resize+Normalize+ToTensor | early_stopped | RandomCrop224+Resize+Normalize+ToTensor slower than threshold: 0.257 sec/image >= 0.050; RandomCrop224+Resize+Normalize+ToTensor slower than threshold: 0.253 sec/image >= 0.050 |
| GPU DataLoader | dali | RandomCrop224+AutoContrast+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'AutoContrast' |
| GPU DataLoader | dali | RandomCrop224+Blur+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Blur' |
| GPU DataLoader | dali | RandomCrop224+ChannelDropout+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'ChannelDropout' |
| GPU DataLoader | dali | RandomCrop224+ChannelShuffle+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'ChannelShuffle' |
| GPU DataLoader | dali | RandomCrop224+CornerIllumination+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'CornerIllumination' |
| GPU DataLoader | dali | RandomCrop224+Elastic+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Elastic' |
| GPU DataLoader | dali | RandomCrop224+EnhanceDetail+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'EnhanceDetail' |
| GPU DataLoader | dali | RandomCrop224+EnhanceEdge+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'EnhanceEdge' |
| GPU DataLoader | dali | RandomCrop224+GaussianIllumination+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'GaussianIllumination' |
| GPU DataLoader | dali | RandomCrop224+Grayscale+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Grayscale' |
| GPU DataLoader | dali | RandomCrop224+Invert+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Invert' |
| GPU DataLoader | dali | RandomCrop224+LinearIllumination+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'LinearIllumination' |
| GPU DataLoader | dali | RandomCrop224+LongestMaxSize+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'LongestMaxSize' |
| GPU DataLoader | dali | RandomCrop224+MedianBlur+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'MedianBlur' |
| GPU DataLoader | dali | RandomCrop224+MotionBlur+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'MotionBlur' |
| GPU DataLoader | dali | RandomCrop224+OpticalDistortion+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'OpticalDistortion' |
| GPU DataLoader | dali | RandomCrop224+Perspective+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Perspective' |
| GPU DataLoader | dali | RandomCrop224+PhotoMetricDistort+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'PhotoMetricDistort' |
| GPU DataLoader | dali | RandomCrop224+PlankianJitter+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'PlankianJitter' |
| GPU DataLoader | dali | RandomCrop224+PlasmaBrightness+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'PlasmaBrightness' |
| GPU DataLoader | dali | RandomCrop224+PlasmaContrast+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'PlasmaContrast' |
| GPU DataLoader | dali | RandomCrop224+PlasmaShadow+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'PlasmaShadow' |
| GPU DataLoader | dali | RandomCrop224+Posterize+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Posterize' |
| GPU DataLoader | dali | RandomCrop224+RGBShift+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'RGBShift' |
| GPU DataLoader | dali | RandomCrop224+Rain+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Rain' |
| GPU DataLoader | dali | RandomCrop224+RandomGamma+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'RandomGamma' |
| GPU DataLoader | dali | RandomCrop224+RandomJigsaw+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'RandomJigsaw' |
| GPU DataLoader | dali | RandomCrop224+RandomRotate90+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'RandomRotate90' |
| GPU DataLoader | dali | RandomCrop224+Sharpen+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Sharpen' |
| GPU DataLoader | dali | RandomCrop224+SmallestMaxSize+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'SmallestMaxSize' |
| GPU DataLoader | dali | RandomCrop224+Snow+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Snow' |
| GPU DataLoader | dali | RandomCrop224+Solarize+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Solarize' |
| GPU DataLoader | dali | RandomCrop224+ThinPlateSpline+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'ThinPlateSpline' |
| GPU DataLoader | dali | RandomCrop224+Transpose+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'Transpose' |
| GPU DataLoader | dali | RandomCrop224+UnsharpMask+Normalize+ToTensor | unsupported | DALI image pipeline does not implement transform 'UnsharpMask' |
| GPU DataLoader | kornia | RandomCrop224+GaussianIllumination+Normalize+ToTensor | unsupported | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| CPU micro | kornia | Elastic | early_stopped | Elastic slower than threshold: 0.268 sec/image >= 0.050 |
| CPU micro | kornia | MedianBlur | early_stopped | MedianBlur slower than threshold: 0.398 sec/image >= 0.050 |
| CPU micro | kornia | PlasmaBrightness | early_stopped | PlasmaBrightness slower than threshold: 0.065 sec/image >= 0.050 |
| CPU micro | kornia | PlasmaContrast | early_stopped | PlasmaContrast slower than threshold: 0.063 sec/image >= 0.050 |
| CPU micro | pillow | MedianBlur | early_stopped | MedianBlur slower than threshold: 0.167 sec/image >= 0.050 |
| CPU micro | torchvision | Elastic | early_stopped | Elastic slower than threshold: 0.122 sec/image >= 0.050 |
| GPU micro | kornia | Affine | unsupported | ValueError: Inputs must have same device Got center (cuda:0, torch.float32), angle (cpu, torch.float32) and scale (cpu, torch.float32) |
| GPU micro | kornia | CornerIllumination | unsupported | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| GPU micro | kornia | Erasing | unsupported | NotImplementedError: "check_uniform_bounds" not implemented for 'Long' |
| GPU micro | kornia | LinearIllumination | unsupported | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| GPU micro | kornia | Perspective | unsupported | NotImplementedError: "check_uniform_bounds" not implemented for 'Long' |
| GPU micro | kornia | PlasmaBrightness | early_stopped | PlasmaBrightness slower than threshold: 0.156 sec/image >= 0.050 |
| GPU micro | kornia | PlasmaContrast | early_stopped | PlasmaContrast slower than threshold: 0.146 sec/image >= 0.050 |
| GPU micro | kornia | PlasmaShadow | early_stopped | PlasmaShadow slower than threshold: 0.146 sec/image >= 0.050 |
| GPU micro | torchvision | Elastic | early_stopped | Elastic slower than threshold: 0.181 sec/image >= 0.050 |
