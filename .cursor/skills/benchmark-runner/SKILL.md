---
name: benchmark-runner
description: Automates running image/video augmentation benchmarks for single or multiple libraries, validates outputs, generates comparison reports, and updates documentation. Use when running benchmarks, comparing library performance, or when the user mentions benchmark, run_all.sh, run_single.sh, or performance testing.
---

# Benchmark Runner

Run augmentation benchmarks with standardized configurations and automatic result validation.

## Running Image Benchmarks

### Single library
```bash
./run_single.sh \
  -d /path/to/images \
  -o output/library_results.json \
  -s benchmark/transforms/library_impl.py \
  -n 2000 \
  -r 5
```

### All libraries
```bash
./run_all.sh \
  -d /path/to/images \
  -o output/ \
  -n 2000 \
  -r 5
```

## Running Video Benchmarks

### Single library
```bash
./run_video_single.sh \
  -d /path/to/videos \
  -o output_videos/library_results.json \
  -s benchmark/transforms/library_video_impl.py
```

### All libraries
```bash
./run_video_all.sh \
  -d /path/to/videos \
  -o output_videos/
```

## Standard Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `-n` | 2000 | Number of images/videos |
| `-r` | 5 | Number of benchmark runs |
| `--max-warmup` | 1000 | Maximum warmup iterations |
| `--warmup-window` | 5 (images), 20 (videos) | Variance window size |
| `--warmup-threshold` | 0.05 | Stability threshold |

## Generating Reports

After running benchmarks:

```bash
# Compare image results
python -m tools.compare_results -r output/

# Compare video results
python -m tools.compare_video_results -r output_videos/

# Generate speedup plots
python -m tools.generate_speedup_plots \
  --results-dir output/ \
  --output-dir docs/images \
  --type images \
  --reference-library albumentationsx

# Update all documentation
./tools/update_docs.sh
```

## Validating Results

Check result JSON structure:
- `metadata.system_info` - system configuration
- `metadata.library_versions` - library versions
- `metadata.benchmark_params` - benchmark settings
- `results[transform_name].median_throughput` - performance metric
- `results[transform_name].warmup_iterations` - convergence info

## Custom Transforms

Create Python file with:
```python
LIBRARY = "library_name"

def __call__(transform, image):
    return transform(image)

TRANSFORMS = [
    {
        "name": "TransformName",
        "transform": LibraryTransform()
    }
]
```

Then run with `-s your_file.py`

## Workflow

1. **Prepare data**: Ensure images/videos are in target directory
2. **Run benchmark**: Use appropriate script
3. **Validate output**: Check JSON exists and has expected structure
4. **Generate reports**: Create comparison tables and plots
5. **Update docs**: Run update_docs.sh if updating documentation
