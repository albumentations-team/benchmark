# Unsupported And Early-Stopped Rows

This supplement table preserves the full unsupported and slow-row detail used by the paper.

| Regime | Library | Transform | Status | Reason |
|---|---|---|---|---|
| CPU DataLoader | pillow | RandomCrop224+Resize+Normalize+ToTensor | early_stopped | RandomCrop224+Resize+Normalize+ToTensor slower than threshold: 0.253 sec/image >= 0.050; RandomCrop224+Resize+Normalize+ToTensor slower than threshold: 0.257 sec/image >= 0.050 |
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
| GPU DataLoader | kornia | RandomCrop224+Affine+Normalize+ToTensor | unsupported | ValueError: Inputs must have same batch size dimension. Got center torch.Size([256, 2]), angle torch.Size([1]) and scale torch.Size([1, 2]) |
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
