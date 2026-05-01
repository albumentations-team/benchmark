# Video Paper Transform Set

Video micro benchmarks use three libraries:

- `albumentationsx`
- `torchvision`
- `kornia`

Video pipeline benchmarks may also include `dali` when DALI is available. The DALI-supported subset is smaller, but it does not add any extra transforms to the 2+ library set below.

A transform is included only when it exists in at least two selected video libraries.

Transform count: 52.

```text
Affine
AutoContrast
Blur
Brightness
CLAHE
CenterCrop224
ChannelDropout
ChannelShuffle
ColorJiggle
ColorJitter
Contrast
CornerIllumination
Elastic
Equalize
Erasing
GaussianBlur
GaussianIllumination
GaussianNoise
Grayscale
HorizontalFlip
Hue
Invert
JpegCompression
LinearIllumination
MedianBlur
MotionBlur
Normalize
OpticalDistortion
Pad
Perspective
PlankianJitter
PlasmaBrightness
PlasmaContrast
PlasmaShadow
Posterize
RGBShift
Rain
RandomCrop224
RandomGamma
RandomJigsaw
RandomResizedCrop
RandomRotate90
Resize
Rotate
SaltAndPepper
Saturation
Sharpen
Shear
Snow
Solarize
ThinPlateSpline
VerticalFlip
```
