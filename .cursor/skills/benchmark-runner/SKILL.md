---
name: benchmark-runner
description: Automates running image/video augmentation benchmarks for single or multiple libraries, validates outputs, generates comparison reports, and updates documentation. Use when running benchmarks, comparing library performance, or when the user mentions benchmark, benchmark.cli, pyperf, GCP benchmark runs, or performance testing.
---

# Benchmark Runner

Run augmentation benchmarks with standardized configurations and automatic result validation.

## Running Image Benchmarks

Use the unified CLI. Legacy `run_all.sh` / `run_single.sh` examples are stale for this repo.

### Single library
```bash
python -m benchmark.cli run \
  --scenario image-rgb \
  --mode micro \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro \
  --libraries albumentationsx \
  --num-items 2000 \
  --num-runs 5 \
  --timer pyperf
```

### All libraries
```bash
python -m benchmark.cli run \
  --scenario image-rgb \
  --mode micro \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro \
  --libraries albumentationsx torchvision kornia pillow \
  --num-items 2000 \
  --num-runs 5 \
  --timer pyperf
```

## Running Video Benchmarks

Use the unified CLI (`python -m benchmark.cli run --media video ...`). Legacy `run_video_*.sh` scripts are not in-repo.

### Google Cloud (detached)

Default `--cloud gcp` path: uploads repo + `job.json` to GCS, creates a VM with a startup script that downloads one **dataset archive/object** from `gs://` (for example `val.tar`), unpacks/stages it on local disk, runs the same `benchmark.cli run` flags (including `--spec`, warmup, `--multichannel`), writes artifacts under `gs://<results-base>/<run_id>/`, then deletes the VM. See README **Google Cloud (detached)** and `benchmark/cloud/gcp.py`. Use `--gcp-attached` for blocking SSH/debug runs.

## Optimization Policies

- Stage datasets as one archive/object in cloud runs; do not copy individual images one by one for each VM.
- Keep timed data local to the benchmark machine. Detached GCP runs unpack to local disk before running.
- Micro benchmarks preload the requested number of media items once per library, in that library's native format.
- Pyperf runs may use per-transform subprocesses, but those subprocesses must reuse the per-library media cache and must not decode images again.
- Construct only the transform being measured in pyperf subprocesses. Avoid eager construction of all transforms because some libraries warn or do setup in constructors.
- Use joined environments for compatible libraries (`torch_stack` for torchvision/Kornia/Pillow image runs, `torch_video` for torchvision/Kornia video runs).
- Cache environments by resolved requirements, Python version, media type, and environment group; reuse the GCS venv cache for detached GCP unless deliberately rebuilding.
- Preflight slow transforms and record an early-stop payload instead of spending the full pyperf budget on transforms that exceed the slow threshold.
- Preserve single-thread internal execution for micro benchmarks; pipeline benchmarks can use production-style workers/threading and must record those settings.
- Prefer `--no-refresh-requirements` for local reruns when dependency versions are intentionally fixed.

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

### Video specs (Albumentations)

Albumentations supports multi-frame input as `(T, H, W, C)`. Use the batch key `images`, not a per-frame loop:

```python
def __call__(transform, video):
    return np.ascontiguousarray(transform(images=video)["images"])
```

See `benchmark/transforms/albumentationsx_video_impl.py` and `.cursor/rules/video_custom_transforms_architecture.mdc`.

## Workflow

1. **Prepare data**: Ensure images/videos are in target directory
2. **Run benchmark**: Use appropriate script
3. **Validate output**: Check JSON exists and has expected structure
4. **Generate reports**: Create comparison tables and plots
5. **Update docs**: Run update_docs.sh if updating documentation
