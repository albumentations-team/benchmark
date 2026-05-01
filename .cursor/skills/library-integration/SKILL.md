---
name: library-integration
description: Guides adding support for new image/video augmentation libraries to the benchmark suite. Use when integrating a new library, adding library support, or when the user mentions adding a new augmentation library to test.
---

# Library Integration

Add support for new augmentation libraries to the benchmark suite.

## Integration Checklist

```
- [ ] Create transform implementation file
- [ ] Create requirements file
- [ ] Register CLI spec maps and environment groups
- [ ] Test with sample data
- [ ] Generate baseline results
- [ ] Update documentation
```

## Step 1: Create Transform Implementation

Create `benchmark/transforms/{library}_impl.py`:

```python
"""Transform implementations for {library}."""

# Import library
import {library}

# Define library name
LIBRARY = "{library}"

def __call__(transform, image):
    """Apply transform to image.

    Adapt this to library's calling convention:
    - Some use transform(image)
    - Some use transform(image=image)
    - Some require specific formats
    """
    return transform(image)

# Define transforms to benchmark
TRANSFORMS = [
    {
        "name": "HorizontalFlip",
        "transform": {library}.HorizontalFlip(),
    },
    {
        "name": "VerticalFlip",
        "transform": {library}.VerticalFlip(),
    },
    # Add more transforms...
]
```

### Image Loading

Add loader to `benchmark/utils.py` if needed:

```python
def get_image_loader(library: str):
    """Get appropriate image loader for library."""
    if library == "new_library":
        def load_new_library(path):
            # Load in library's expected format
            return img
        return load_new_library
```

## Step 2: Create Requirements File

Create `requirements/{library}.txt`:

```txt
{library}>=1.0.0
numpy>=1.19.0
opencv-python>=4.5.0
```

Add to `requirements/requirements.txt` if base dependencies needed.

## Step 3: Register CLI Support

### For image benchmarks

Add the library to the image spec and requirement maps in `benchmark/cli.py`:

```python
_IMAGE_SPECS["newlib"] = "benchmark/transforms/newlib_impl.py"
_REQUIREMENTS["newlib"] = "requirements/newlib.txt"
```

### For video benchmarks

Add the library to `_VIDEO_SPECS` and `_VIDEO_REQUIREMENTS` in `benchmark/cli.py`.
If it can share dependencies with existing libraries, add it to the relevant `_ENV_GROUPS` entry instead of creating a separate venv.

## Step 4: Test Integration

```bash
# Test single library
python -m benchmark.cli run \
  --scenario image-rgb \
  --mode micro \
  --data-dir /path/to/test/images \
  --output test_output/newlib \
  --libraries newlib \
  --num-items 10 \
  --num-runs 1

# Verify JSON output
python -c "import json; print(json.load(open('test_output/newlib/image-rgb/micro/newlib_micro_results.json'))['metadata']['library_versions'])"
```

## Step 5: Generate Baseline Results

```bash
# Full benchmark run
python -m benchmark.cli run \
  --scenario image-rgb \
  --mode micro \
  --data-dir /path/to/full/dataset \
  --output output/newlib_rgb_micro \
  --libraries newlib \
  --num-items 2000 \
  --num-runs 5
```

## Step 6: Update Documentation

Create `docs/images/newlib_metadata.yaml` or `docs/videos/newlib_metadata.yaml`:

```yaml
library_name: NewLib
version: "1.0.0"
description: Brief description of the library
documentation: https://newlib.readthedocs.io
repository: https://github.com/org/newlib
```

Run comparison:
```bash
./tools/update_docs.sh
```

## Common Issues

### Import Errors
- Ensure library is in requirements file
- Check virtual environment activation
- Verify compatible Python version

### Transform Failures
- Check transform API matches library version
- Verify image format (RGB vs BGR, HWC vs CHW)
- Test transforms individually first

### Performance Issues
- Verify thread settings (`OMP_NUM_THREADS=1`)
- Check warmup convergence
- Monitor early stopping conditions

## Video Library Integration

For video libraries, create `{library}_video_impl.py` and adapt:

```python
import numpy as np


def __call__(transform, video):
    """Apply transform to one clip. Shape conventions:
    - (T, H, W, C) for Albumentations (NumPy) — use batch API when available
    - (T, C, H, W) for torch / Kornia tensors
    """
    # Albumentations: native multi-frame API (one param draw per clip)
    return np.ascontiguousarray(transform(images=video)["images"])
```

For Kornia/torchvision, see `kornia_video_impl.py` / `torchvision_video_impl.py`. Video loaders differ — check `benchmark/utils.py` (`get_video_loader`) and `benchmark/video_runner.py`.
