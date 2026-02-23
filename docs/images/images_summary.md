# Images Augmentation Benchmark — Summary

*Speedup vs albumentationsx (reference). Higher = faster. Single CPU thread.*

| Library | Median speedup | Geometric mean | Max speedup | Fastest on % transforms |
|---------|---------------|----------------|-------------|--------------------------|
| **albumentationsx** (ref) | 1.0 | 1.0 | — | 96% |
| kornia | 5.74× | 5.42× | 54.3× | 4% |
| torchvision | 5.54× | 4.12× | 19.4× | 8% |
