---
name: benchmark-runner
description: Automates running image/video augmentation benchmarks for single or multiple libraries, validates outputs, generates comparison reports, and updates documentation. Use when running benchmarks, comparing library performance, or when the user mentions benchmark, benchmark.cli, pyperf, GCP benchmark runs, or performance testing.
---

# Benchmark Runner

Run augmentation benchmarks with standardized configurations and automatic result validation.

## Running Image Benchmarks

Use the config-first CLI for paper, cloud, or repeatable runs. Legacy `run_all.sh`, `run_single.sh`, and flag-only
commands are stale for this repo; start from YAML and use CLI flags only as overrides.

### Config-first runs
```bash
python -m benchmark.cli plan --config configs/examples/local_rgb_micro_cpu.yaml
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml
python -m benchmark.cli run --config configs/paper/gcp_g2_rgb_gpu_smoke.yaml --gcp-dry-run
```

`benchmark/parser.py` owns parser construction and provided-flag tracking. `benchmark/config/models.py` defines
`BenchmarkRunConfig`. `benchmark/config/resolve.py` loads YAML, applies supported CLI overrides, writes
`resolved_config.yaml`, and shapes typed GCP payloads. `benchmark/config/argv.py` builds attached-mode CLI argv from
typed configs for cloud debug runs. `benchmark/config/plan.py` prints generated jobs and
expected files. `benchmark/config/env.py` embeds the resolved config in result metadata. `benchmark/cloud/paths.py` and
`benchmark/output_naming.py` keep dry-run plans aligned with real VM paths and result filenames.

Use checked-in configs under `configs/examples/` for local smoke runs and `configs/paper/` for paper/GCP runs. Prefer
`--num-items`, `--num-runs`, `--device`, `--workers`, `--batch-size`, and `--output` as overrides instead of editing many
flags by hand.

### Single library
```bash
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro \
  --libraries albumentationsx \
  --num-items 2000 \
  --num-runs 5
```

### All libraries
```bash
python -m benchmark.cli run --config configs/examples/local_rgb_micro_cpu.yaml \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro \
  --libraries albumentationsx torchvision kornia pillow \
  --num-items 2000 \
  --num-runs 5
```

## Running Video Benchmarks

Use the unified CLI with a YAML config, for example
`python -m benchmark.cli run --config configs/examples/local_video_micro_cpu.yaml --data-dir /path/to/videos`.
Legacy `run_video_*.sh` scripts are not in-repo.

### Google Cloud (detached)

Default `--cloud gcp` path: uploads repo + typed `job.json` to GCS, creates a VM with a startup script that downloads one **dataset tarball** from `gs://` (for example `val.tar` or `ucf101.tar`), unpacks/stages media files on local disk, writes `/root/benchmark-work/job_config.yaml`, runs `benchmark.cli run --resolved-config /root/benchmark-work/job_config.yaml`, writes artifacts under `gs://<results-base>/<run_id>/`, then deletes the VM. `benchmark/cloud/launch.py` owns launch option resolution and typed job payload assembly; `benchmark/cloud/gcp.py` owns the lower-level GCP runner. See README **Google Cloud (detached)** and `benchmark/cloud/paths.py`. Use `--gcp-attached` for blocking SSH/debug runs.

## Optimization Policies

- Treat `benchmark/matrix.py` as the source of truth for built-in scenario/library/mode support, spec paths,
  requirements, joined environment groups, paper transform sets, device support, pipeline scopes, and backend names.
- Treat `benchmark/policy.py` as the source of truth for media defaults and slow-transform preflight thresholds. Do not
  duplicate image/video defaults in individual runners.
- Treat `benchmark/config/models.py` as the source of truth for run shape and validation. Add user-facing config fields
  there first, then update `benchmark/config/resolve.py`, `benchmark/config/plan.py`, examples under `configs/`, and tests.
- Use `benchmark/output_naming.py` for result filenames and `benchmark/cloud/paths.py` for detached GCP VM paths. Do not
  duplicate filename or VM staging inference in CLI, planner, or cloud code.
- Use `benchmark/jobs.py` for command construction and `benchmark/orchestrator.py` for backend dispatch. Do not add
  backend-specific branches to `benchmark/cli.py`; DALI should remain a `dali_pipeline` job backend.
- `benchmark/runner.py` is a compatibility/simple-timer runner. Production CLI micro runs use
  `benchmark/pyperf_micro_runner.py`; production DataLoader runs use `benchmark/pipeline_runner.py`.
- Stage datasets as one tarball in cloud runs; do not copy individual images/videos one by one for each VM. On macOS, create dataset tarballs with `COPYFILE_DISABLE=1`, `tar --no-xattrs`, and excludes for `.DS_Store`, AppleDouble `._*`, and `__MACOSX`.
- Keep timed data local to the benchmark machine. Detached GCP runs unpack to local disk before running.
- Micro benchmarks preload the requested number of media items once per library, in that library's native format. Video micro preloads fixed-length clips from `--clip-length` (16 frames for `video-16f`), not full source videos. Torchvision video clips stay `uint8` tensors so `torchvision.transforms.v2.JPEG` can run; Kornia video clips use float16 on CUDA.
- Micro specs measure only the named transform in native layout, then force returned outputs into contiguous memory before
  timing stops. Never add `Normalize`, `ToTensor`, axis conversion, or DataLoader collation work to `*_impl.py`.
- Pipeline specs (`*_pipeline_impl.py`) own recipe-level `Normalize+ToTensor`: AlbumentationsX uses `ToTensorV2`, Pillow
  uses `torchvision.transforms.PILToTensor` before normalization, and torchvision/Kornia already operate on tensors. Video
  pipeline specs are separate from video micro specs and use `crop + transform + Normalize + ToTensor` recipe semantics for
  AlbumentationsX, torchvision, and Kornia. The pipeline runner should use default PyTorch collation and should not guess
  or repair channel layouts.
- Pyperf runs may use per-transform subprocesses, but those subprocesses must reuse the per-library media cache and must not decode images again.
- Construct only the transform being measured in pyperf subprocesses. Avoid eager construction of all transforms because some libraries warn or do setup in constructors.
- Use joined environments for compatible libraries (`torch_stack` for torchvision/Kornia/Pillow image runs, `torch_video` for torchvision/Kornia video runs).
- Cache environments by resolved requirements, Python version, media type, and environment group; reuse the GCS venv cache for detached GCP unless deliberately rebuilding.
- Requirement lock refresh is expected once per library or joined-environment launch when refresh is enabled. Do not add extra cross-library refresh orchestration unless it removes real work without changing dependency freshness semantics. Prefer `--no-refresh-requirements` for repeated local reruns with fixed locks.
- Pipeline result filenames include key sweep parameters: `library_scope_n{num_items|all}_r{num_runs}_w{workers}_b{batch_size}[_dev-{device}]_results.json`.
- Preflight slow transforms in both micro and pipeline modes, then record an early-stop payload instead of spending the full benchmark budget on transforms that exceed the slow threshold. Defaults: images skip at `>=0.05 sec/image` (`<=20 img/s`), videos skip at `>=2.0 sec/video`.
- Keep the slow-transform guard enabled for paper/DataLoader sweeps. It prevents the benchmark from appearing stuck on transforms that are too slow for practical training use. Use `--disable-slow-skip` only when the user explicitly asks to measure slow transforms exhaustively.
- Preserve single-thread internal execution for micro benchmarks; pipeline benchmarks can use production-style workers/threading and must record those settings.
- Watch for lazy or partially lazy outputs. Micro timing must force each library to finish its own transform work and return
  contiguous outputs: NumPy arrays use `np.ascontiguousarray`, tensor-like outputs use `.contiguous()`, and Pillow
  `Image.Image` outputs are converted to contiguous NumPy arrays. Do not add checksums or unrelated validation inside the
  timed benchmark.
- Only benchmark transforms a library supports directly. Do not build large benchmark-side helper implementations to imitate another library's API. For Pillow, keep direct `Image` / `ImageOps` / `ImageFilter` operations and skip Albumentations-style composites such as random crops, `PadIfNeeded`, `SafeRotate`, `ShiftScaleRotate`, `LongestMaxSize`, and `SmallestMaxSize`.
- Kornia excludes `benchmark/transforms/kornia_unstable.py` **only** from video DataLoader/pipeline recipes. Keep those
  transforms in Kornia image micro, image pipeline, 9-channel, and video micro unless a separate failure is observed.
- Paper runs do not use every transform from `benchmark/transforms/specs.py`. Use `--transform-set paper` for the curated
  lists in `docs/paper_transform_sets/*.md`; within each scenario's library set, keep transforms with at least two
  implementations. Video DataLoader/pipeline support is additionally filtered by dedicated video recipe specs.
- Keep benchmarks fair but fast. Avoid repeated decode, loader construction, conversion, synchronization, checksums, materialization, or dependency work unless it is explicitly part of the named measurement scope or needed to make lazy work complete.
- Prefer `--no-refresh-requirements` for local reruns when dependency versions are intentionally fixed.
- All long-running loops must expose visual progress with tqdm and a descriptive `desc`. Use labels such as `scenario/mode` for library loops, `Load images (<library>, <channels>ch)` for media loading, `Micro transforms (<library>, <media>)`, `Pyperf micro transforms (<library>, <media>)`, and `Pipeline transforms (<library>, <scope>, w=<workers>, b=<batch_size>)`. Never add anonymous progress bars.

## Architecture And Tests

When changing benchmark orchestration, update the architecture docs and tests:

- Docs: `docs/benchmark_architecture.md`, `docs/benchmark_scope.md`, and relevant README sections.
- Config tests: `tests/test_config_models.py`, `tests/test_config_plan.py`, `tests/test_cloud_paths.py`, and
  `tests/test_output_naming.py`.
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
