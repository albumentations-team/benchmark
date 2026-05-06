# 9-Channel Paper Transform Set

9-channel paper benchmarks use three libraries:

- `albumentationsx`
- `torchvision`
- `kornia`

A transform is included only when it exists in at least two of these libraries.

`CenterCrop224` is omitted (redundant with `RandomCrop224` / `RandomResizedCrop`).

Transform count: 41.

```text
Resize
RandomCrop224
RandomResizedCrop
HorizontalFlip
VerticalFlip
Pad
Rotate
Affine
Perspective
Elastic
ChannelShuffle
Grayscale
GaussianBlur
GaussianNoise
Invert
Posterize
Solarize
Sharpen
AutoContrast
Erasing
JpegCompression
RandomGamma
MedianBlur
MotionBlur
Brightness
Contrast
Blur
ChannelDropout
LinearIllumination
CornerIllumination
GaussianIllumination
PlasmaBrightness
PlasmaContrast
PlasmaShadow
OpticalDistortion
Shear
ThinPlateSpline
LongestMaxSize
SmallestMaxSize
RandomRotate90
RandomJigsaw
```
