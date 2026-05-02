# Benchmark Architecture

The benchmark runner is organized around a small set of shared policy and orchestration modules. Keep new benchmark
features in these modules unless there is a strong reason to put logic directly in a runner.

## Control Plane

- `benchmark/parser.py` owns argument parser construction and provided-flag tracking for CLI overrides.
- `benchmark/cli.py` handles commands. It resolves old flags or YAML files into typed run configs before launching work.
- `benchmark/config/models.py` defines the Pydantic run config schema for selection, data, execution, output, and cloud
  settings. Config validation catches unsupported combinations before local work or VM creation starts.
- `benchmark/config/argv.py` builds compatibility `benchmark.cli run` argv from typed configs for cloud fallback paths.
- `benchmark/config/resolve.py` loads YAML configs, applies supported CLI overrides, writes `resolved_config.yaml`, and
  converts old flag-based commands into typed configs.
- `benchmark/config/plan.py` expands a resolved config into a dry-run plan: generated jobs, expected result files, and
  cloud target settings.
- `benchmark/config/transform_sets.py` expands named transform sets into concrete transform names before configs are
  written to metadata or executed.
- `benchmark/config/env.py` owns the environment handoff that embeds resolved run configs in result metadata.
- `benchmark/matrix.py` is the declarative benchmark matrix: scenarios, modes, library spec files, requirements, joined
  environment groups, paper transform-set files, device support, pipeline scopes, and backend selection.
- `benchmark/jobs.py` defines immutable `BenchmarkJob` objects and builds subprocess commands for micro and pipeline jobs.
- `benchmark/orchestrator.py` executes jobs and owns backend dispatch. The `dali_pipeline` backend runs in a subprocess
  (`benchmark/dali_pipeline_worker.py`) using the DALI venv Python from `benchmark/envs.py`, so DALI is imported only after
  `requirements/dali-video.txt` is installed or refreshed.
- `benchmark/envs.py` owns virtualenv creation, requirement lock refresh, dependency cache keys, and dependency installs.
- `benchmark/output_naming.py` owns result filename rules shared by execution and dry-run plans, including device suffixes.
- `benchmark/cloud/paths.py` owns detached-VM path constants and GCS-to-staged-data path inference shared by plans and
  GCP launch code.
- `benchmark/cloud/stage_dataset.py` owns detached-cloud dataset tarball validation and extraction. It filters media files
  by benchmark media type and ignores macOS archive junk such as `.DS_Store`, AppleDouble `._*`, and `__MACOSX`.
- `benchmark/policy.py` owns shared media policy: default item counts, warmup limits, item labels, throughput units, and
  slow-transform preflight defaults.
- `benchmark/devices.py` owns CUDA/MPS device resolution, transform/sample movement, support validation, and
  synchronization helpers shared by micro and pipeline runners.

## Data And Spec Loading

- `benchmark/specs/load.py` loads Python transform spec files and validates the `LIBRARY`, `__call__`, and `TRANSFORMS`
  contract.
- `benchmark/media/loaders.py` loads image/video samples for micro benchmarks. RGB and 9-channel image benchmarks share
  the same loader path; 9-channel samples are synthesized by wrapping the library image loader with
  `make_multichannel_loader`. Video micro samples are decoded as fixed-length clips using the scenario `clip_length`, so
  `video-16f` preloads 16 frames per source video instead of whole videos.
- `benchmark/transforms/image_recipe_specs.py` and `benchmark/transforms/video_recipe_specs.py` define DataLoader recipe
  transform sets. Pipeline scenarios use dedicated `*_pipeline_impl.py` specs so crop, augmentation, normalization, and
  tensor conversion stay in the library-owned recipe layer.
- `benchmark/runner.py` is now the compatibility/simple-timer runner. Production CLI micro runs use
  `benchmark/pyperf_micro_runner.py`; production DataLoader runs use `benchmark/pipeline_runner.py`.

## Timing Engines

- `benchmark/pyperf_micro_runner.py` runs augmentation-only micro benchmarks with pyperf. It preloads media once per
  library, reuses the media cache across per-transform subprocesses, constructs only the measured transform, and applies
  the shared slow-skip policy from `benchmark/policy.py`. Micro specs keep each library's native image layout and do not
  add DataLoader recipe steps such as `Normalize` or `ToTensor`. For image `torchvision` and `kornia`, `--device` moves
  samples and transforms to CUDA/MPS before timing and synchronizes device work without copying outputs back to CPU.
- `benchmark/pipeline_runner.py` runs DataLoader-style recipes. It measures one of three scopes:
  `memory_dataloader_augment`, `decode_dataloader_augment`, or `decode_dataloader_augment_batch_copy`. Pipeline specs own
  recipe-level tensor conversion (`Normalize+ToTensor`) so the runner can use PyTorch default collation without
  benchmark-side channel-layout guesses. Video pipeline recipes mirror RGB recipes: crop or crop-transform, then the
  measured transform, then `Normalize+ToTensor`/tensor-ready conversion. Default collation stacks fixed-shape tensor recipe
  outputs in every DataLoader scope; `decode_dataloader_augment_batch_copy` additionally materializes the collated tensor
  batch on CUDA/MPS when requested. For image `torchvision` and `kornia` with `--device`, DataLoader workers still load
  CPU samples, then the runner copies the collated batch to the selected device and applies the recipe once at batch
  level.
- DALI video pipeline runs are represented as `BenchmarkJob(backend="dali_pipeline")` and dispatched by
  `benchmark/orchestrator.py` via `benchmark/dali_pipeline_worker.py`, not by CLI special cases.

## Scenario Flow

```text
benchmark.cli
  -> benchmark.config resolves YAML/flags into BenchmarkRunConfig
  -> benchmark.config.transform_sets expands named transform sets
  -> benchmark.config.env installs resolved config metadata for runners
  -> benchmark.config.plan expands generated jobs for dry-run/debug output
  -> benchmark.scenarios resolves scenario/mode/libraries
  -> benchmark.matrix resolves spec/env/backend policy
  -> benchmark.jobs builds BenchmarkJob
  -> benchmark.orchestrator executes the job
  -> pyperf_micro_runner or pipeline_runner writes result JSON
```

## Extension Rules

- Add new scenario/library/mode support in `benchmark/matrix.py` first.
- Add new config fields in `benchmark/config/models.py` first, with validation and YAML examples when the field is
  user-facing.
- Add new shared defaults in `benchmark/policy.py`, not separately in micro and pipeline runners.
- Add new device behavior in `benchmark/devices.py`, then plumb it through jobs/runners.
- Add new command construction to `benchmark/jobs.py`, not inline in `benchmark/cli.py`.
- Add new result filename policy in `benchmark/output_naming.py`, not separately in execution and plan code.
- Add new detached GCP VM path policy in `benchmark/cloud/paths.py`, not separately in cloud launch and plan code.
- Add new backend dispatch to `benchmark/orchestrator.py`, not as a CLI branch.
- Keep transform implementations explicit and library-specific. Do not create benchmark-side recreations for transforms a
  library does not directly support.
- Add tests for matrix invariants and command construction whenever the benchmark matrix changes.

## Test Coverage

Architecture-sensitive tests live in:

- `tests/test_config_models.py`: typed config loading, CLI override precedence, validation failures, and compatibility
  conversion from old flags.
- `tests/test_config_plan.py`: config-to-plan expansion for micro, pipeline, decode, expected outputs, and cloud fields.
- `tests/test_cloud_paths.py`: detached GCP VM path constants and staged-data path inference.
- `tests/test_output_naming.py`: result filename policy for micro, pipeline, manual, and device-suffixed outputs.
- `tests/test_matrix.py`: scenario/mode/library matrix, spec paths, requirements, paper transform sets, device policy.
- `tests/test_jobs_orchestrator.py`: job command construction, pyperf sidecar cleanup, DALI backend dispatch, GCP attached
  cleanup on failure.
- `tests/test_pipeline_runner.py`: tiny DataLoader execution, device resolution, shared slow-skip defaults.
- `tests/test_pyperf_micro_runner.py`: pyperf micro timing helpers, media caching, device-resident micro setup.
- `tests/test_stage_dataset.py`: GCP dataset tarball planning and extraction for image/video media.
- `tests/test_slow_threshold.py`: shared slow-threshold formatting and policy defaults.
