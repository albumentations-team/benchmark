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

Transforms with `time_per_image > 0.1` sec are considered slow:

```python
slow_transforms = {}
for name, metrics in results.items():
    if 'mean_time' in metrics:
        time_per_img = metrics['mean_time'] / params['num_images']
        if time_per_img > 0.1:
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

**Fix**: Run benchmark on idle system with `num_runs=10` for better statistics.

### Early Stopping
**Symptom**: `early_stopped=True`
**Reasons**:
1. Transform too slow (> 0.1 sec/image)
2. Timeout (> 60 sec total)

**Analysis**: Check `early_stop_reason` for details.

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

**Slow transform (> 0.1 sec/img)**:
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
