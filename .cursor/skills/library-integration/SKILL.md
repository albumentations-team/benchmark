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
- [ ] Update run scripts
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

## Step 3: Update Run Scripts

### For image benchmarks

Edit `run_all.sh`:

```bash
# Add to LIBRARIES array (line ~95)
LIBRARIES=("albumentationsx" "imgaug" "torchvision" "kornia" "augly" "newlib")

# Add to SPEC_FILES array
SPEC_FILES=(
    "benchmark/transforms/albumentationsx_impl.py"
    "benchmark/transforms/imgaug_impl.py"
    "benchmark/transforms/torchvision_impl.py"
    "benchmark/transforms/kornia_impl.py"
    "benchmark/transforms/augly_impl.py"
    "benchmark/transforms/newlib_impl.py"
)
```

### For video benchmarks

Edit `run_video_all.sh` similarly.

## Step 4: Test Integration

```bash
# Test single library
./run_single.sh \
  -d /path/to/test/images \
  -o test_output/newlib_results.json \
  -s benchmark/transforms/newlib_impl.py \
  -n 10 \
  -r 1

# Verify JSON output
python -c "import json; print(json.load(open('test_output/newlib_results.json'))['metadata']['library_versions'])"
```

## Step 5: Generate Baseline Results

```bash
# Full benchmark run
./run_single.sh \
  -d /path/to/full/dataset \
  -o output/newlib_results.json \
  -s benchmark/transforms/newlib_impl.py \
  -n 2000 \
  -r 5
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
def __call__(transform, video):
    """Apply transform to video tensor.

    Video shape conventions:
    - (T, H, W, C) for CPU libraries
    - (T, C, H, W) for GPU libraries
    """
    return transform(video)
```

Video loaders differ - check `benchmark/video_runner.py` for examples.
