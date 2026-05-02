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
  --num-runs 5
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
  --num-runs 5
```

## Running Video Benchmarks

Use the unified CLI (`python -m benchmark.cli run --media video ...`). Legacy `run_video_*.sh` scripts are not in-repo.

### Google Cloud (detached)

Default `--cloud gcp` path: uploads repo + `job.json` to GCS, creates a VM with a startup script that downloads one **dataset archive/object** from `gs://` (for example `val.tar`), unpacks/stages it on local disk, runs the same `benchmark.cli run` flags (including `--spec`, warmup, `--multichannel`), writes artifacts under `gs://<results-base>/<run_id>/`, then deletes the VM. See README **Google Cloud (detached)** and `benchmark/cloud/gcp.py`. Use `--gcp-attached` for blocking SSH/debug runs.

## Optimization Policies

- Treat `benchmark/matrix.py` as the source of truth for built-in scenario/library/mode support, spec paths,
  requirements, joined environment groups, paper transform sets, device support, pipeline scopes, and backend names.
- Treat `benchmark/policy.py` as the source of truth for media defaults and slow-transform preflight thresholds. Do not
  duplicate image/video defaults in individual runners.
- Use `benchmark/jobs.py` for command construction and `benchmark/orchestrator.py` for backend dispatch. Do not add
  backend-specific branches to `benchmark/cli.py`; DALI should remain a `dali_pipeline` job backend.
- `benchmark/runner.py` is a compatibility/simple-timer runner. Production CLI micro runs use
  `benchmark/pyperf_micro_runner.py`; production DataLoader runs use `benchmark/pipeline_runner.py`.
- Stage datasets as one archive/object in cloud runs; do not copy individual images one by one for each VM.
- Keep timed data local to the benchmark machine. Detached GCP runs unpack to local disk before running.
- Micro benchmarks preload the requested number of media items once per library, in that library's native format.
- Micro specs measure only the named transform in native layout. Never add `Normalize`, `ToTensor`, axis conversion, or
  DataLoader collation work to `*_impl.py`.
- Pipeline specs (`*_pipeline_impl.py`) own recipe-level `Normalize+ToTensor`: AlbumentationsX uses `ToTensorV2`, Pillow
  uses `torchvision.transforms.PILToTensor` before normalization, and torchvision/Kornia already operate on tensors. The
  pipeline runner should use default PyTorch collation and should not guess or repair channel layouts.
- Pyperf runs may use per-transform subprocesses, but those subprocesses must reuse the per-library media cache and must not decode images again.
- Construct only the transform being measured in pyperf subprocesses. Avoid eager construction of all transforms because some libraries warn or do setup in constructors.
- Use joined environments for compatible libraries (`torch_stack` for torchvision/Kornia/Pillow image runs, `torch_video` for torchvision/Kornia video runs).
- Cache environments by resolved requirements, Python version, media type, and environment group; reuse the GCS venv cache for detached GCP unless deliberately rebuilding.
- Requirement lock refresh is expected once per library or joined-environment launch when refresh is enabled. Do not add extra cross-library refresh orchestration unless it removes real work without changing dependency freshness semantics. Prefer `--no-refresh-requirements` for repeated local reruns with fixed locks.
- Pipeline result filenames include key sweep parameters: `library_scope_n{num_items|all}_r{num_runs}_w{workers}_b{batch_size}[_dev-{device}]_results.json`.
- Preflight slow transforms in both micro and pipeline modes, then record an early-stop payload instead of spending the full benchmark budget on transforms that exceed the slow threshold. Defaults: images skip at `>=0.05 sec/image` (`<=20 img/s`), videos skip at `>=2.0 sec/video`.
- Keep the slow-transform guard enabled for paper/DataLoader sweeps. It prevents the benchmark from appearing stuck on transforms that are too slow for practical training use. Use `--disable-slow-skip` only when the user explicitly asks to measure slow transforms exhaustively.
- Preserve single-thread internal execution for micro benchmarks; pipeline benchmarks can use production-style workers/threading and must record those settings.
- Watch for lazy or partially lazy outputs. The timed call must force each library to finish its own transform work without adding cross-library work. For Pillow/PIL, call `Image.load()` on returned `Image.Image` objects inside the adapter. Do **not** add NumPy conversion, checksums, or `np.asarray()` to the timed benchmark for fairness; use those only in local diagnostics.
- Only benchmark transforms a library supports directly. Do not build large benchmark-side helper implementations to imitate another library's API. For Pillow, keep direct `Image` / `ImageOps` / `ImageFilter` operations and skip Albumentations-style composites such as random crops, `PadIfNeeded`, `SafeRotate`, `ShiftScaleRotate`, `LongestMaxSize`, and `SmallestMaxSize`.
- Paper runs do not use every transform from `benchmark/transforms/specs.py`. Use `--transform-set paper` to select transforms supported by at least two selected libraries; the fixed lists are `docs/paper_transform_sets/rgb.md`, `docs/paper_transform_sets/9ch.md`, and `docs/paper_transform_sets/video.md`.
- Keep benchmarks fair but fast. Avoid repeated decode, loader construction, conversion, synchronization, checksums, materialization, or dependency work unless it is explicitly part of the named measurement scope or needed to make lazy work complete.
- Prefer `--no-refresh-requirements` for local reruns when dependency versions are intentionally fixed.
- All long-running loops must expose visual progress with tqdm and a descriptive `desc`. Use labels such as `scenario/mode` for library loops, `Load images (<library>, <channels>ch)` for media loading, `Micro transforms (<library>, <media>)`, `Pyperf micro transforms (<library>, <media>)`, and `Pipeline transforms (<library>, <scope>, w=<workers>, b=<batch_size>)`. Never add anonymous progress bars.

## Architecture And Tests

When changing benchmark orchestration, update the architecture docs and tests:

- Docs: `docs/benchmark_architecture.md`, `docs/benchmark_scope.md`, and relevant README sections.
- Matrix tests: `tests/test_matrix.py`.
- Job/orchestrator tests: `tests/test_jobs_orchestrator.py`.
- Pipeline runner tests: `tests/test_pipeline_runner.py`.
- Shared policy tests: `tests/test_slow_threshold.py`.

## Standard Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `-n` | 2000 | Number of images/videos |
| `-r` | 5 | Number of benchmark runs |
| `--max-warmup` | 1000 | Maximum warmup iterations |
| `--warmup-window` | 5 (images), 20 (videos) | Variance window size |
| `--warmup-threshold` | 0.05 | Stability threshold |
| `--slow-threshold-sec-per-item` | 0.05 image / 2.0 video | Early-stop threshold for impractically slow transforms |
| `--slow-preflight-items` | 10 images / 3 videos | Items used for slow-transform preflight |

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
