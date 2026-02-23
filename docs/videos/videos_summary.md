# Videos Augmentation Benchmark — Summary

*Speedup vs albumentationsx (reference). Higher = faster. Single CPU thread.*

| Library | Median speedup | Geometric mean | Max speedup | Fastest on % transforms |
|---------|---------------|----------------|-------------|--------------------------|
| **albumentationsx** (ref) | 1.0 | 1.0 | — | 42% |
| kornia | 1.32× | 1.48× | 15.3× | 21% |
| torchvision | 0.08× | 0.10× | 1.5× | 56% |
