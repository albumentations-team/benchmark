# Benchmark Architecture

The benchmark runner is organized around a small set of shared policy and orchestration modules. Keep new benchmark
features in these modules unless there is a strong reason to put logic directly in a runner.

## Control Plane

- `benchmark/cli.py` parses arguments and keeps backwards-compatible helper wrappers for existing tests and scripts.
- `benchmark/matrix.py` is the declarative benchmark matrix: scenarios, modes, library spec files, requirements, joined
  environment groups, paper transform-set files, device support, pipeline scopes, and backend selection.
- `benchmark/jobs.py` defines immutable `BenchmarkJob` objects and builds subprocess commands for micro and pipeline jobs.
- `benchmark/orchestrator.py` executes jobs and owns backend dispatch. The `dali_pipeline` backend runs in a subprocess
  (`benchmark/dali_pipeline_worker.py`) using the DALI venv Python from `benchmark/envs.py`, so DALI is imported only after
  `requirements/dali-video.txt` is installed or refreshed.
- `benchmark/envs.py` owns virtualenv creation, requirement lock refresh, dependency cache keys, and dependency installs.
- `benchmark/policy.py` owns shared media policy: default item counts, warmup limits, item labels, throughput units, and
  slow-transform preflight defaults.

## Data And Spec Loading

- `benchmark/specs/load.py` loads Python transform spec files and validates the `LIBRARY`, `__call__`, and `TRANSFORMS`
  contract.
- `benchmark/media/loaders.py` loads image/video samples for micro benchmarks. RGB and 9-channel image benchmarks share
  the same loader path; 9-channel samples are synthesized by wrapping the library image loader with
  `make_multichannel_loader`.
- `benchmark/runner.py` is now the compatibility/simple-timer runner. Production CLI micro runs use
  `benchmark/pyperf_micro_runner.py`; production DataLoader runs use `benchmark/pipeline_runner.py`.

## Timing Engines

- `benchmark/pyperf_micro_runner.py` runs augmentation-only micro benchmarks with pyperf. It preloads media once per
  library, reuses the media cache across per-transform subprocesses, constructs only the measured transform, and applies
  the shared slow-skip policy from `benchmark/policy.py`. Micro specs keep each library's native image layout and do not
  add DataLoader recipe steps such as `Normalize` or `ToTensor`.
- `benchmark/pipeline_runner.py` runs DataLoader-style recipes. It measures one of three scopes:
  `memory_dataloader_augment`, `decode_dataloader_augment`, or `decode_dataloader_augment_batch_copy`. Pipeline specs own
  recipe-level tensor conversion (`Normalize+ToTensor`) so the runner can use PyTorch default collation without
  benchmark-side channel-layout guesses.
- DALI video pipeline runs are represented as `BenchmarkJob(backend="dali_pipeline")` and dispatched by
  `benchmark/orchestrator.py` via `benchmark/dali_pipeline_worker.py`, not by CLI special cases.

## Scenario Flow

```text
benchmark.cli
  -> benchmark.scenarios resolves scenario/mode/libraries
  -> benchmark.matrix resolves spec/env/backend policy
  -> benchmark.jobs builds BenchmarkJob
  -> benchmark.orchestrator executes the job
  -> pyperf_micro_runner or pipeline_runner writes result JSON
```

## Extension Rules

- Add new scenario/library/mode support in `benchmark/matrix.py` first.
- Add new shared defaults in `benchmark/policy.py`, not separately in micro and pipeline runners.
- Add new command construction to `benchmark/jobs.py`, not inline in `benchmark/cli.py`.
- Add new backend dispatch to `benchmark/orchestrator.py`, not as a CLI branch.
- Keep transform implementations explicit and library-specific. Do not create benchmark-side recreations for transforms a
  library does not directly support.
- Add tests for matrix invariants and command construction whenever the benchmark matrix changes.

## Test Coverage

Architecture-sensitive tests live in:

- `tests/test_matrix.py`: scenario/mode/library matrix, spec paths, requirements, paper transform sets, device policy.
- `tests/test_jobs_orchestrator.py`: job command construction, pyperf sidecar cleanup, DALI backend dispatch, GCP attached
  cleanup on failure.
- `tests/test_pipeline_runner.py`: tiny DataLoader execution, device resolution, shared slow-skip defaults.
- `tests/test_slow_threshold.py`: shared slow-threshold formatting and policy defaults.
