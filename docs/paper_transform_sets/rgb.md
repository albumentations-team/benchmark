# RGB Paper Transform Set

RGB paper benchmarks use four libraries:

- `albumentationsx`
- `torchvision`
- `kornia`
- `pillow`

A transform is included only when it exists in at least two of these libraries.

Transform count: 58.

```text
Resize
RandomCrop224
RandomResizedCrop
CenterCrop224
HorizontalFlip
VerticalFlip
Pad
Rotate
Affine
Perspective
Elastic
ColorJitter
ChannelShuffle
Grayscale
RGBShift
GaussianBlur
GaussianNoise
Invert
Posterize
Solarize
Sharpen
AutoContrast
Equalize
Erasing
JpegCompression
RandomGamma
PlankianJitter
MedianBlur
MotionBlur
CLAHE
Brightness
Contrast
Blur
ChannelDropout
LinearIllumination
CornerIllumination
GaussianIllumination
Hue
PlasmaBrightness
PlasmaContrast
PlasmaShadow
Rain
SaltAndPepper
Saturation
Snow
OpticalDistortion
Shear
ThinPlateSpline
PhotoMetricDistort
ColorJiggle
LongestMaxSize
SmallestMaxSize
Transpose
RandomRotate90
RandomJigsaw
EnhanceEdge
EnhanceDetail
UnsharpMask
```
