---
name: performance-analysis
description: Analyzes benchmark results to identify slow transforms, warmup issues, and performance regressions. Compares speedups across libraries and generates optimization recommendations. Use when analyzing performance, investigating slow benchmarks, or comparing library results.
---

# Performance Analysis

Deep dive into benchmark results to identify performance issues and optimization opportunities.

## Loading and Inspecting Results

```python
import json

# Load results
with open('output/library_results.json') as f:
    data = json.load(f)

# Check metadata
system_info = data['metadata']['system_info']
library_versions = data['metadata']['library_versions']
params = data['metadata']['benchmark_params']

# Access results
results = data['results']
for transform_name, metrics in results.items():
    print(f"{transform_name}: {metrics['median_throughput']:.2f} img/sec")
```

## Identifying Slow Transforms

Transforms with `time_per_image >= 0.05` sec are considered slow for image benchmarks. This is `<=20 img/s`, below the practical floor for DataLoader training pipelines.

```python
slow_transforms = {}
for name, metrics in results.items():
    if 'mean_time' in metrics:
        time_per_img = metrics['mean_time'] / params['num_images']
        if time_per_img > 0.05:
            slow_transforms[name] = time_per_img

# Sort by slowest
sorted_slow = sorted(slow_transforms.items(), key=lambda x: x[1], reverse=True)
```

Check for early stopping:
```python
for name, metrics in results.items():
    if metrics.get('early_stopped'):
        print(f"{name}: {metrics['early_stop_reason']}")
```

Early stopping is expected for transforms that are too slow for practical use. Micro and DataLoader pipeline runners both preflight slow transforms and write an `early_stopped` result instead of spending the full benchmark budget. Keep this enabled for paper sweeps so the benchmark does not appear stuck on transforms that are unusably slow.

## Analyzing Warmup Stability

Good warmup converges in < 500 iterations:

```python
import numpy as np

for name, metrics in results.items():
    warmup_iters = metrics.get('warmup_iterations', 0)
    if warmup_iters > 500:
        print(f"{name}: {warmup_iters} iterations (slow convergence)")

    # Check variance stability
    if not metrics.get('variance_stable', True):
        print(f"{name}: unstable variance")
```

## Comparing Libraries

```bash
# Generate comparison table
python -m tools.compare_results -r output/

# Generate speedup analysis
python -m tools.generate_speedup_plots \
  --results-dir output/ \
  --output-dir analysis/ \
  --type images \
  --reference-library albumentationsx
```

### Reading Speedup CSV

```python
import pandas as pd

# Load speedups
df = pd.read_csv('docs/images/images_speedups.csv', index_col=0)

# Find fastest library per transform
for transform in df.index:
    fastest = df.loc[transform].idxmax()
    speedup = df.loc[transform, fastest]
    print(f"{transform}: {fastest} ({speedup:.2f}×)")

# Overall statistics
print(f"Median speedup: {df['albumentationsx'].median():.2f}×")
print(f"Max speedup: {df['albumentationsx'].max():.2f}×")
print(f"Min speedup: {df['albumentationsx'].min():.2f}×")
```

## Checking Lazy Output Artifacts

When a library is unexpectedly faster, check whether it returns lazy or partially materialized outputs.

- Pillow/PIL must call `Image.load()` on returned `Image.Image` objects inside the timed adapter so the image operation is complete before timing stops.
- Do not add `np.asarray()`, pixel sums, checksums, or other cross-library output consumption to the timed benchmark. That measures extra conversion/validation work, not the transform API.
- Use local diagnostics for suspicious transforms: compare raw Pillow call time against Pillow call + `Image.load()`, and inspect output identity / memory sharing. Keep those diagnostics out of production benchmark timing.
- Crop/transpose/resize-like transforms are the first place to check because they are the most likely to expose views, reused objects, or deferred buffers.
- Treat benchmark-side reimplementations as suspect. If Pillow/Kornia/torchvision lacks a direct transform analogue, mark it unsupported instead of composing helpers that make the comparison about our glue code.

## Performance Regression Detection

Compare results across runs:

```python
import json

def compare_results(old_file, new_file, threshold=0.1):
    """Detect regressions > threshold (10% by default)."""
    with open(old_file) as f:
        old = json.load(f)
    with open(new_file) as f:
        new = json.load(f)

    regressions = []
    for name in old['results']:
        if name not in new['results']:
            continue

        old_throughput = old['results'][name]['median_throughput']
        new_throughput = new['results'][name]['median_throughput']

        change = (new_throughput - old_throughput) / old_throughput
        if change < -threshold:  # Negative = slower
            regressions.append({
                'transform': name,
                'old_throughput': old_throughput,
                'new_throughput': new_throughput,
                'change_pct': change * 100
            })

    return regressions
```

## Parametric Analysis

For custom transforms with parameters:

```bash
python tools/analyze_parametric_results.py parametric_results.json
```

This shows:
- Best/worst configurations per transform
- Performance impact of parameter choices
- Optimal settings for your use case

## Common Performance Issues

### Repeated Media Loading
**Symptom**: `Loading RGB images` appears before every transform in a pyperf micro run.
**Cause**: The per-transform pyperf subprocess is bypassing the per-library media cache.
**Fix**: Ensure the parent loads media once per library, writes the temporary media cache, and passes `--media-cache` to transform subprocesses and pyperf workers. Do not benchmark by rereading images from disk per transform.

### Constructor Warning Spam
**Symptom**: Warnings from unrelated transforms, for example `ShiftScaleRotate` or `ElasticTransform`, appear while benchmarking `Solarize`.
**Cause**: Transform specs are eagerly constructing all transforms during import.
**Fix**: Build transforms lazily and pass `BENCHMARK_TRANSFORMS_FILTER` into pyperf subprocesses so only the measured transform is instantiated.

### Slow Warmup
**Symptom**: `warmup_iterations > 500`
**Causes**:
- JIT compilation (first-time overhead)
- Memory allocation patterns
- Cache effects

**Analysis**:
```python
# Check warmup variance
warmup_throughputs = [...]  # From debug output
import numpy as np
recent = np.mean(warmup_throughputs[-10:])
overall = np.mean(warmup_throughputs)
stability = abs(recent - overall) / overall
```

### High Variance
**Symptom**: `std_throughput / median_throughput > 0.15`
**Causes**:
- Background processes
- Thermal throttling
- Memory pressure

**Fix**: First increase measured work per sample (`--num-items` or pyperf `--min-time`/loops if exposed), then increase `--num-runs`. Run on an idle, plugged-in machine; on Linux use `python -m pyperf system tune`.

### Early Stopping
**Symptom**: `early_stopped=True`
**Reasons**:
1. Transform too slow (`>=0.05 sec/image`, i.e. `<=20 img/s`)
2. Preflight timeout (> 60 sec total for images)

**Analysis**: Check `early_stop_reason` for details. Early stopping is expected policy for very slow transforms in both micro and DataLoader pipeline modes; do not force exhaustive runs unless the user explicitly asks for slow-transform measurements.

### Cloud Setup Dominates Runtime
**Symptom**: Runs spend most time copying data or rebuilding venvs.
**Fix**:
- Stage a single dataset archive/object, not individual images.
- Reuse joined environments where dependency sets are compatible.
- Reuse the local venv cache and detached GCP GCS venv cache.
- Use `--no-refresh-requirements` for local reruns with fixed dependency versions.
- Use reduced production-path runs with small `--num-items` / selected `--transforms` before full cloud runs.

## Thread Configuration

Verify single-threaded execution:

```python
# Check thread settings in results
thread_settings = data['metadata']['thread_settings']

# Should all be '1'
assert thread_settings['OMP_NUM_THREADS'] == '1'
assert thread_settings['MKL_NUM_THREADS'] == '1'
```

Multi-threading invalidates comparisons between libraries.

## Optimization Recommendations

Based on results:

**Slow transform (> 0.05 sec/img)**:
- Profile with cProfile or py-spy
- Check for unnecessary copies
- Look for algorithmic improvements

**High variance (> 15%)**:
- Increase `num_runs`
- Run on isolated system
- Check thermal conditions

**Slow warmup (> 500 iters)**:
- Accept if due to JIT
- Otherwise investigate memory allocation
- Consider caching strategies

**Low throughput vs reference**:
- Compare implementations
- Check data format conversions
- Profile hot paths
