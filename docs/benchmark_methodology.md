# Benchmark Methodology

This document describes the benchmark methodology in prose. It is intended to be the source material for the paper,
website, and longer-form writing about the benchmark. It explains what is measured, what is deliberately excluded, and
where those decisions live in the implementation.

The benchmark is designed around one principle: compare libraries under explicit, reproducible measurement scopes instead
of mixing incompatible claims. A primitive transform profiler, a CPU DataLoader pipeline, a GPU batch pipeline, and a DALI
graph answer different questions. The benchmark keeps those regimes separate, records the scope in result metadata, and
expects analysis and figures to preserve that separation.

## Control Flow

Benchmark execution starts from a checked-in YAML config, not from a loose collection of flags. `benchmark/cli.py` loads
the config through `benchmark/config/resolve.py`, validates it with the Pydantic schema in `benchmark/config/models.py`,
applies supported CLI overrides, expands named transform sets through `benchmark/config/transform_sets.py`, and writes the
resolved config into the output directory. This gives each run a concrete, inspectable contract before timing begins.

The resolved config is expanded into immutable jobs by `benchmark/config/plan.py` and `benchmark/jobs.py`. A job contains
the library, scenario, mode, media type, transform filter, data directory, output file, worker settings, batch size,
thread policy, device option, slow-transform policy, and backend. The important design decision is that the CLI does not
decide backend behavior ad hoc. Scenario support, library support, device support, requirement groups, paper transform-set
files, pipeline scopes, and backend names are centralized in `benchmark/matrix.py`.

Once jobs exist, `benchmark/orchestrator.py` dispatches them to the appropriate timing engine. Production micro jobs run
through `benchmark/pyperf_micro_runner.py`. Production DataLoader jobs run through `benchmark/pipeline_runner.py`. DALI
pipeline jobs are represented as normal benchmark jobs and dispatched through the same orchestrator path, with DALI import
and execution isolated behind `benchmark/dali_pipeline_worker.py`. This separation keeps config parsing, job construction,
environment setup, and measurement logic independently reviewable.

## Scenarios And Library Sets

The benchmark matrix defines three public scenario families: RGB images, 9-channel images, and fixed-length video clips.
RGB image benchmarks compare AlbumentationsX, torchvision, Kornia, and Pillow where each library has a meaningful direct
implementation. The 9-channel image scenario compares AlbumentationsX, torchvision, and Kornia; Pillow is excluded because
its practical direct API is RGB/PIL-image oriented. Video scenarios compare AlbumentationsX, torchvision, and Kornia, with
DALI available only in separately labeled native-pipeline scopes where its architecture is comparable.

The matrix also separates micro and pipeline modes. Micro mode is a transform profiler. Pipeline mode is a training-style
DataLoader measurement. The same library can therefore appear in both modes, but the results must not be treated as the
same claim. Micro rows answer "how expensive is this transform implementation when data is already decoded and in the
library's native representation?" Pipeline rows answer "how fast is this recipe when it runs through a DataLoader-like
path with collation and, when requested, decode and device transfer?"

## Transform Selection

The shared transform catalog lives in `benchmark/transforms/specs.py`. Paper transform sets are fixed markdown files under
`docs/paper_transform_sets/`, one for each scenario. The named transform set is expanded before execution and stored in
run metadata, so a result can be traced back to the exact transform universe used for that run.

The eligibility rule is intentionally conservative. A transform belongs in a scenario-level paper set only when it exists
in at least two selected libraries for that scenario. This avoids turning the benchmark into a catalogue-size contest
where one library is penalized for implementing operations that no other competitor exposes. At the same time, each
library reports only the transforms it supports directly. Missing support is recorded as unsupported or absent coverage;
it is not treated as fast.

The benchmark does not recreate missing library features with large benchmark-side compatibility implementations simply
to fill a table cell. That decision matters because substantial helper code can dominate the measured path and make a
library look slow for reasons unrelated to the library itself. Library-specific transform specs should map explicit
library APIs to the canonical catalog by behavior and parameters, not only by name. When a known device-specific issue
exists after transform-set expansion, the narrow exclusion belongs in `benchmark/transform_filters.py` rather than in the
global paper transform set. This keeps CPU rows, other-library rows, and unaffected scenarios intact.

## Environment Isolation

Each library or compatible library group runs in an isolated virtual environment managed by `benchmark/envs.py`.
AlbumentationsX has its own environment. For image benchmarks, torchvision, Kornia, and Pillow share the `torch_stack`
environment because their dependency sets are compatible and because sharing avoids redundant setup. For video
benchmarks, torchvision and Kornia share `torch_video`. DALI uses a separate environment because importing DALI should
only happen after its own requirements are installed.

Requirements are declared in `requirements/*.in` and materialized into `requirements/*.txt`. When requirement refresh is
enabled, `benchmark/envs.py` asks `uv pip compile` to refresh the lock file before installing dependencies. Dependency
installation is cached by resolved requirement contents, Python version, media type, and environment group. This gives two
properties at once: benchmark runs can use fresh compatible dependencies by default, and repeated local or cloud reruns do
not reinstall packages when the resolved dependency contract has not changed.

The reason for environment isolation is practical reproducibility. A benchmark comparing Python libraries is sensitive to
transitive dependency versions, compiled extension availability, CUDA bindings, and import-time side effects. Isolating
environments prevents one library's dependency constraints from silently changing another library's measured behavior.

## Media Loading

Media loading is scope-specific. Micro benchmarks preload the requested number of images or video clips once per library
before timing transform rows. Image samples are loaded with the library-native loader selected through shared media
helpers, and 9-channel image samples are synthesized from the same path by wrapping the image loader with a multichannel
loader. Video micro samples are decoded as fixed-length clips according to the scenario, so a `video-16f` run preloads
16-frame clips rather than full source videos.

Preloading is deliberate in micro mode. The goal is to isolate augmentation cost from disk traversal, file decode, Python
DataLoader overhead, and batch collation. A micro transform row must not reread files from disk, decode media, normalize,
convert to tensor, or repair channel layouts unless that work is part of the named library transform itself.

Pipeline mode uses a different data model. In `decode_dataloader_augment`, the dataset stores paths and loads or decodes
inside the DataLoader path. In `memory_dataloader_augment`, decoded samples are preloaded once and the DataLoader path
measures worker scheduling, augmentation, collation, and recipe execution without disk/decode cost. In
`decode_dataloader_augment_batch_copy`, the benchmark additionally materializes the collated batch tensor and copies it to
CUDA or MPS when a device is requested. These scopes are separate because they answer separate production questions.

Cloud runs stage datasets as one tarball on the VM's local disk before timing begins. The benchmark does not time against
mounted buckets or network paths. `benchmark/cloud/stage_dataset.py` validates and extracts the tarball before the control
environment exists, so it intentionally stays stdlib-only and avoids importing Pydantic or `benchmark.config`. This makes
cloud bootstrap failures easier to diagnose and prevents dependency setup from becoming a prerequisite for dataset
staging.

## Micro Timing

Production micro timing is implemented in `benchmark/pyperf_micro_runner.py`. The runner applies the `micro-single`
thread policy before measurement so the benchmark measures one augmentation stream rather than whatever thread fan-out a
library happens to choose by default. This is the relevant primitive number for DataLoader-style training, where many
workers scale out by process and each worker effectively consumes a CPU core. Hidden per-transform multithreading can make
single-row numbers look attractive while oversubscribing real training pipelines.

Micro rows use pyperf because transform timings can be small and noisy. The runner preloads media once, writes a
per-library media cache, and then fans out one transform per pyperf subprocess when multiple transforms are requested.
Each subprocess lazily constructs only the transform being measured. This avoids repeatedly paying media-loading cost
while still giving pyperf process isolation for the timed target.

The timed loop applies the transform to every preloaded item for the selected number of loops. The runner synchronizes the
selected device before and after the loop. It also materializes outputs inside the timed section: Pillow images are forced
through contiguous NumPy conversion, tensor-like outputs are made contiguous, and lazy or view-like results are not allowed
to count as finished work before the underlying computation is realized. This is essential for fair comparison because
some libraries can otherwise return objects that defer real work until later.

GPU image micro rows are device-resident profilers for torchvision and Kornia. Samples and transforms are moved to CUDA,
MPS, or the automatically resolved accelerator before timing starts. The timed loop includes device synchronization but
does not include host-to-device transfer. That exclusion is intentional: micro GPU timing is a primitive transform
profiler, not an end-to-end input-pipeline measurement.

## DataLoader Pipeline Timing

Pipeline timing is implemented in `benchmark/pipeline_runner.py`. Pipeline benchmarks measure recipe throughput using a
PyTorch DataLoader path. They are not primitive transform-speed measurements. For non-crop image transforms, the recipe is
random crop, measured transform, normalization, and tensor conversion. For crop transforms, the crop itself replaces the
fixed random crop. Video pipeline specs follow the same idea with clip-shaped data. The recipe specs own normalization and
tensor conversion so the runner receives fixed-shape outputs that PyTorch default collation can stack without
benchmark-side layout guesses.

The pipeline runner records batch size, worker count, minimum run time, minimum batches, thread policy, media type,
scenario, device option, and pipeline scope. It warms the DataLoader path once before timed runs. Each timed run builds a
fresh DataLoader, iterates until both the minimum time and minimum batch constraints are satisfied, materializes produced
batches, synchronizes the selected device, and stores throughput in items per second. Summary statistics are generated by
`benchmark/results.py`.

The main production pipeline thread policy is `pipeline-default`. Controlled comparisons can use
`pipeline-single-worker`. The worker initializer is part of the timing setup so DataLoader workers receive the intended
thread policy rather than inheriting uncontrolled library defaults. This is separate from micro timing because production
DataLoader performance depends on worker count, batch size, collation, library thread behavior, and the fixed recipe
overhead around the measured transform.

## GPU Pipeline Timing

GPU image pipeline rows are separate from CPU pipeline rows. For torchvision and Kornia, PyTorch DataLoader workers still
prepare fixed-shape samples on CPU before collation. The collated batch is then copied to the selected device, the measured
augmentation and normalization run on the accelerator, and the benchmark synchronizes before stopping the timer. This
models the practical constraint that DataLoader workers cannot independently return GPU tensors for normal multi-worker
training without changing the architecture of the input pipeline.

Kornia can apply a batched augmentation with per-image random parameters using `same_on_batch=False`, so its GPU batch
path uses the batched transform. TorchVision v2 does not expose an equivalent batched random-transform API for every
operation in this benchmark, so the TorchVision GPU image path applies the measured augmentation in a per-sample loop and
then normalizes the whole batch. The per-sample loop is slower, but it preserves the intended semantics: each image should
receive its own random parameters rather than sharing one random transform across the batch.

CUDA DataLoader rows also record peak allocated and reserved memory during timed runs. `benchmark/pipeline_runner.py`
resets peak memory statistics immediately before the run, captures allocation state before timing, and stores per-run GPU
memory details under each transform result. GPU augmentation consumes accelerator memory that could otherwise be used by
the model, optimizer, activations, or a larger batch. Therefore GPU memory is part of the production tradeoff, not an
incidental diagnostic.

Pyperf micro rows do not report peak GPU memory because their timed loops execute inside pyperf worker processes. The
metadata says this explicitly. This prevents readers from confusing a missing micro memory field with a measured zero.

## DALI Pipeline Timing

DALI is treated as a native pipeline backend, not as another micro transform spec. DALI image and video jobs are declared
in `benchmark/matrix.py`, constructed as normal jobs, and dispatched through `benchmark/orchestrator.py`. The backend then
runs through `benchmark/dali_pipeline_worker.py` and DALI-specific adapters.

This design avoids comparing DALI against scopes it does not naturally implement. DALI graph execution, mixed decode,
pipeline scheduling, and batch production are different from a Python function that transforms one already-decoded sample.
When DALI is included, it must be labeled as a DALI pipeline row with its own supported subset and unsupported results.
The benchmark consumes produced batches so lazy graph scheduling is not mistaken for completed augmentation work.

## Slow-Transform Guard

The shared slow-transform policy lives in `benchmark/policy.py`. Image transforms default to an early-stop threshold of
0.05 seconds per image, equivalent to 20 images per second. Video transforms default to 2.0 seconds per video. The same
policy object also defines preflight item counts and maximum preflight duration.

Both micro and pipeline timing engines run a preflight before spending the full benchmark budget on a transform unless
slow skipping is explicitly disabled. If the preflight shows that the transform is below the practical throughput floor,
or if the preflight itself exceeds the maximum allowed duration, the result is recorded as an early-stopped row with the
preflight throughput and reason. The row remains visible in output data. It is not silently dropped.

This guard exists because a full sweep multiplies by libraries, transforms, runs, item counts, DataLoader epochs, worker
settings, and device modes. One or two impractically slow rows can dominate wall time and block the rest of the benchmark.
Early stopping says exactly what happened: the transform is supported enough to run, but too slow under the selected
scope to justify a full measurement budget.

## Unsupported Rows And Coverage

Unsupported rows are part of the methodology. A library may lack a transform, fail on a device-specific code path, require
CPU-only input for a transform that otherwise has GPU support, or hit a known stability problem. Those outcomes are
recorded as unsupported results with reasons rather than hidden by changing the global transform universe.

Coverage and throughput must be interpreted together. A library that implements fewer transforms can appear fast over its
measured subset because difficult rows are missing. A library with broader coverage can expose more slow or hard cases.
For this reason, paper figures and website summaries should not use one merged leaderboard across micro, CPU DataLoader,
GPU DataLoader, and DALI regimes. They should report measured throughput alongside coverage, unsupported rows,
early-stopped rows, and the denominator of the fixed transform set.

## Result Metadata And Statistics

Every result JSON contains metadata built by `benchmark/results.py`. The metadata records system information, library
versions, thread settings, environment and git state, GPU snapshot, dataset fingerprint, timing backend, measurement
scope, data source, whether decode is included, whether collation is included, whether host-to-device transfer is
included, whether DataLoader workers are included, scenario, mode, library or decoder, benchmark parameters, and the
resolved run config payload when available.

Per-transform results store the raw throughputs and times that contributed to the summary. They also store median,
mean, standard deviation, coefficient of variation, approximate 95 percent confidence interval, percentile throughput
values, number of successful runs, and an unstable flag when variation exceeds the configured threshold. Unsupported and
early-stopped rows use the same output structure where possible, but with zero successful full runs and an explicit
reason.

The purpose of this metadata is reproducibility and interpretation. A throughput number without measurement scope,
device, worker count, batch size, dependency versions, thread policy, and dataset fingerprint is not enough to support a
paper claim. The benchmark records those facts at execution time so downstream figure generation and narrative writing can
defend the comparison.

## Cloud Execution

Cloud execution is an execution environment for the same benchmark jobs, not a separate methodology. Detached GCP runs
upload the repository and typed job definition to Google Cloud Storage, create a VM, stage one dataset tarball onto local
disk, write the resolved run config on the VM, execute `python -m benchmark.cli run --resolved-config ...`, upload results
and logs, and delete the VM unless the run was configured to keep it.

The methodology requirement is that benchmark data must be local to the machine doing the timing. The cloud path stages
tarballs locally for that reason. Uploading or reading thousands of individual files from a bucket during measurement
would mix network storage behavior into augmentation throughput. Packaging media as one tarball also makes detached runs
more reliable and easier to reproduce.

## Interpreting Claims

The benchmark intentionally supports more than one kind of claim, but each claim must name its regime. Micro results are
best for implementation profiling, regression detection, and primitive transform-cost comparisons. CPU DataLoader results
are best for production-style training input-pipeline comparisons. GPU DataLoader results are best for evaluating whether
accelerator augmentation is worth its transfer, synchronization, semantic, and memory costs. DALI rows are native-pipeline
comparisons and should be discussed as such.

The safest interpretation rule is simple: do not mix scopes unless the argument is explicitly about the difference
between scopes. A statement about primitive transform speed should come from micro rows. A statement about production
training throughput should come from DataLoader rows. A statement about GPU augmentation should mention host-to-device
transfer, synchronization, per-sample randomness, and peak memory when those are part of the measured path.
