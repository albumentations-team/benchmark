# R10 Elastic calibration

R10 compares elastic deformations after cropping to 224×224. The benchmark uses similar
displacement strength (about 0.74 px RMS) and spatial scale (about 10 px) across
implementations. The [recipe catalog](../catalog/recipes/rgb.yaml) records the parameters.

AlbumentationsX 2.6.1 builds a displacement field from cubic splines on a control grid. AX 2.4.2
smoothed noise with a Gaussian filter. TorchVision and Kornia still smooth noise, but they
convert `alpha` to pixels differently. With the previous settings of `alpha=50` and `sigma=5`,
the field RMS was 0.91 px in AX 2.4.2, 1.15 px in TorchVision, and 258 px in Kornia. The new AX
implementation limits displacement amplitude on a 29×29 grid to preserve topology. We reduced
the target strength across implementations and kept the spatial scale.

| Implementation | Parameters | RMS, px | Scale, px |
| --- | --- | ---: | ---: |
| AX 2.6.1 | 29×29 grid, `displacement_range=(0.0097, 0.0097)` | 0.745 | 10.12 |
| TorchVision 0.29.2 | `alpha=32`, `sigma=5` | 0.738 | 9.88 |
| Kornia 0.9.0 | `alpha=32/224`, `sigma=5` on both axes | 0.738 | 9.88 |

We verified these values on CPU with PyTorch 2.15.0 using 24 fields with seeds 500–523. For AX,
we used its coefficient generator and sampling maps. For TorchVision, we took the field from
`ElasticTransform.make_params`. For Kornia, we subtracted the undeformed grid from the grid
passed to `grid_sample`. We multiplied the normalized TorchVision and Kornia coordinates by 112
to convert them to pixels.

We used the central 176×176 pixels of each map. RMS is the square root of the mean squared
displacement-vector length. To compute spatial scale, we first subtracted the spatial mean of
each field. We then averaged autocorrelation across the two axes and linearly interpolated
between integer lags to find where it first fell below `1/e`. Matching RMS and scale leaves
differences in the fields, boundary effects, and pixels. We measure throughput and memory
separately from calibration, together in one pass on an L4 under the production contract.
