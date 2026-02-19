---
name: transform-spec-validator
description: Validates custom transform specification files for benchmark runner, checking LIBRARY, __call__, and TRANSFORMS definitions. Use when creating custom transforms, validating transform specs, or when the user mentions transform validation or spec files.
---

# Transform Spec Validator

Validate custom transform specification files for the benchmark runner.

## Required Structure

Transform spec files must define three components:

```python
# 1. Library name
LIBRARY = "library_name"

# 2. Call function
def __call__(transform, image_or_video):
    """Apply transform to data."""
    return transform(image_or_video)

# 3. Transforms list
TRANSFORMS = [
    {
        "name": "TransformName",
        "transform": LibraryTransform()
    }
]
```

## Quick Validation

```bash
# Validate spec file
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('test', 'my_transforms.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Check required attributes
assert hasattr(module, 'LIBRARY'), 'Missing LIBRARY'
assert hasattr(module, '__call__'), 'Missing __call__'
assert hasattr(module, 'TRANSFORMS'), 'Missing TRANSFORMS'
assert callable(module.__call__), 'Module.__call__ not callable'

print('✓ Valid spec file')
"
```

## Component Validation

### 1. LIBRARY

Must be a string matching library name:

```python
# ✓ Good
LIBRARY = "albumentationsx"
LIBRARY = "imgaug"

# ✗ Bad
LIBRARY = None
library = "albumentationsx"  # Wrong name
```

### 2. __call__

Must be callable and accept transform + data:

```python
# ✓ Good - Standard pattern
def __call__(transform, image):
    return transform(image)

# ✓ Good - Keyword arguments
def __call__(transform, image):
    return transform(image=image)

# ✓ Good - Format conversion
def __call__(transform, image):
    # Convert if needed
    result = transform(image)
    return result

# ✗ Bad - Wrong signature
def __call__(transform):  # Missing image parameter
    return transform()
```

### 3. TRANSFORMS

Must be list of dicts with 'name' and 'transform' keys:

```python
# ✓ Good
TRANSFORMS = [
    {
        "name": "HorizontalFlip",
        "transform": A.HorizontalFlip(p=1.0)
    },
    {
        "name": "Rotate",
        "transform": A.Rotate(limit=45, p=1.0)
    }
]

# ✗ Bad - Missing keys
TRANSFORMS = [
    {
        "transform": A.HorizontalFlip()  # Missing 'name'
    }
]

# ✗ Bad - Wrong structure
TRANSFORMS = [
    A.HorizontalFlip()  # Not a dict
]
```

## Naming Conventions

### Transform Names

Use descriptive names that include key parameters:

```python
# ✓ Good - Clear and specific
{"name": "HorizontalFlip", ...}
{"name": "Rotate(limit=45)", ...}
{"name": "GaussNoise(var_limit=(10,50))", ...}
{"name": "ToGray(method=weighted_average)", ...}

# ✗ Bad - Too generic
{"name": "Transform1", ...}
{"name": "Rotate", ...}  # When testing multiple rotation limits
```

The name appears in result files and comparison tables.

## Testing Spec Files

### Test with Sample Data

```python
import importlib.util
import numpy as np

# Load spec
spec = importlib.util.spec_from_file_location('test', 'my_transforms.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Create test image
test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

# Test each transform
for t in module.TRANSFORMS:
    name = t['name']
    transform = t['transform']

    try:
        result = module.__call__(transform, test_image)
        assert result is not None
        assert result.shape == test_image.shape
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
```

### Test with Runner

```bash
# Quick test with small dataset
./run_single.sh \
  -d /path/to/test/images \
  -o test_output.json \
  -s my_transforms.py \
  -n 10 \
  -r 1
```

Check output:
```python
import json
with open('test_output.json') as f:
    data = json.load(f)

# Verify all transforms ran
for name in data['results']:
    metrics = data['results'][name]
    if not metrics['supported']:
        print(f"✗ {name}: not supported")
    elif metrics.get('early_stopped'):
        print(f"⚠ {name}: {metrics['early_stop_reason']}")
    else:
        print(f"✓ {name}: {metrics['median_throughput']:.2f} img/sec")
```

## Common Issues

### Import Errors

**Problem**: Module not found when runner loads spec file

**Fix**: Install library in venv before running:
```bash
# Check library in requirements
cat requirements/library.txt

# Or install manually in test venv
python -m venv .venv_test
source .venv_test/bin/activate
pip install library
python my_transforms.py  # Test imports
```

### Transform Failures

**Problem**: Transform raises exception during benchmark

**Causes**:
- Incorrect image format (RGB vs BGR, HWC vs CHW)
- Invalid parameters
- Version incompatibility

**Debug**:
```python
# Add to spec file temporarily
def __call__(transform, image):
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    result = transform(image)
    print(f"Result shape: {result.shape}, dtype: {result.dtype}")
    return result
```

### Probability Issues

**Problem**: Transform sometimes doesn't apply (p < 1.0)

**Fix**: Set probability to 1.0 for deterministic benchmarking:
```python
# ✓ Good - Always applies
transform = A.HorizontalFlip(p=1.0)

# ✗ Bad - Random application
transform = A.HorizontalFlip(p=0.5)
```

## Example Spec Files

### Image Transforms

See:
- `benchmark/transforms/albumentationsx_impl.py`
- `benchmark/transforms/imgaug_impl.py`
- `examples/custom_image_transforms.py`

### Video Transforms

See:
- `benchmark/transforms/albumentationsx_video_impl.py`
- `benchmark/transforms/kornia_video_impl.py`
- `examples/custom_video_specs_template.py`

## Parametric Testing

Test same transform with different parameters:

```python
TRANSFORMS = [
    # Test blur kernel sizes
    {"name": "GaussianBlur(kernel=3)", "transform": A.GaussianBlur(blur_limit=(3,3), p=1)},
    {"name": "GaussianBlur(kernel=7)", "transform": A.GaussianBlur(blur_limit=(7,7), p=1)},
    {"name": "GaussianBlur(kernel=15)", "transform": A.GaussianBlur(blur_limit=(15,15), p=1)},

    # Test noise levels
    {"name": "GaussNoise(var=10)", "transform": A.GaussNoise(var_limit=(10,10), p=1)},
    {"name": "GaussNoise(var=50)", "transform": A.GaussNoise(var_limit=(50,50), p=1)},
]
```

Analyze with:
```bash
python tools/analyze_parametric_results.py results.json
```

## Validation Checklist

Before running benchmarks:

```
- [ ] LIBRARY defined as string
- [ ] __call__ function defined and callable
- [ ] TRANSFORMS is list of dicts
- [ ] Each dict has 'name' and 'transform' keys
- [ ] Transform names are descriptive
- [ ] All transforms have p=1.0
- [ ] Imports work in isolation
- [ ] Test transforms on sample data
- [ ] Quick benchmark run succeeds
```
