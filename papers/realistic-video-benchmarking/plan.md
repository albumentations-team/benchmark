# Realistic Video Benchmarking Plan

## Goal

Turn the current video benchmark into a paper-quality evaluation that answers two separate questions:

1. How fast are equivalent video augmentations when we benchmark the transform itself?
2. How much does that speed matter once we include data loading, host-to-device copy, video decode, and an actual training step?

The key paper claim should be:

> Video augmentation benchmarking is an end-to-end systems problem. Rankings change once clip length, temporal consistency, dataloader overhead, decode cost, and CPU parallelism are measured explicitly.

## Current State In Repo

The current benchmark is useful, but it is not yet fair enough for the paper we want:

- `benchmark/runner.py` preloads all videos before timing.
- For tensor libraries, `_load_videos()` moves tensors to GPU before benchmarking, so host-to-device copy is excluded.
- `benchmark/utils.py` uses different loaders per library, so decode and dtype/layout conversion are library-coupled.
- Built-in AlbumentationsX and Albumentations (MIT) video specs use `transform(images=video)["images"]` (batch API): one parameter draw per clip, consistent across frames.
- `benchmark/transforms/kornia_video_impl.py` uses `same_on_batch=True`, so all frames in a clip get identical transform parameters.

That means the current transform-only benchmark compares:

- CPU vs GPU transform execution
- with videos already loaded
- without dataloader overhead
- without copy overhead
- without decode overhead
- with aligned clip-consistent semantics between Albumentations (batch `images`) and Kornia (`same_on_batch=True`)

This is still useful as a kernel benchmark, but it cannot be the only result in the paper.

## High-Level Deliverables

We should add four benchmark layers and keep them separate in code, docs, and paper figures.

### 1. Transform-only benchmark

Purpose: preserve the current benchmark idea, but make it fairer and more realistic.

Changes:

- Keep the existing benchmark family as the "transform-only / preloaded" benchmark.
- Add explicit clip-length regimes instead of one hidden default.
- Lower the tested frame count for the main benchmark if current `T` is not representative of training.
- Add temporal semantics explicitly in metadata and optional modes:
  - `clip_consistent`: identical sampled params across all frames (current Albumentations path: `images=` batch API)
  - optional `frame_independent`: separate params per frame for ablations only (would require a different `__call__` path, not the default benchmark)

Main output:

- throughput vs clip length
- latency vs clip length
- CPU vs GPU crossover point

### 2. Dataloader + copy benchmark

Purpose: measure the realistic cost of using a CPU augmentation pipeline during GPU training.

Pipeline:

- dataset returns decoded clip
- dataloader workers apply transform
- collate batch
- host-to-device copy

What this captures:

- Python worker overhead
- batching and collation cost
- pinned memory effects
- host-to-device transfer cost
- overlap limitations

Main output:

- samples/sec
- batches/sec
- copy time
- dataloader wait time
- GPU idle fraction if measurable

### 3. Decode + dataloader + copy benchmark

Purpose: add video decoding to the realistic pipeline.

Pipeline:

- open compressed video
- sample frames
- decode frames
- apply transform
- collate
- copy to GPU

This benchmark is important because decode may dominate augmentation.

Main output:

- same metrics as above
- plus decode time contribution
- plus regime labeling: decode-bound vs augment-bound vs copy-bound

### 4. Decode + dataloader + copy + train-step benchmark

Purpose: answer the only question users really care about: does library choice change training throughput?

Pipeline:

- decode
- sample clip
- augment
- collate
- copy
- forward
- backward
- optimizer step

This must be a small, stable training benchmark, not a full research training run.

Main output:

- end-to-end train samples/sec
- step time
- dataloader stall fraction
- GPU utilization proxy if available

## Benchmark Methodology

## A. Temporal Semantics Must Be Explicit

For the paper default:

- all frames in a video clip get exactly the same sampled transform parameters

This matches your stated Albumentations video behavior target and is the fairest comparison against Kornia's `same_on_batch=True`.

Implementation work:

- add a shared concept of video transform mode:
  - `clip_consistent`
  - `frame_independent`
- make Albumentations video transform application clip-consistent
- audit torchvision behavior per transform and document whether it is clip-consistent already or needs explicit handling

Important:

- `clip_consistent` should be the default for the paper
- `frame_independent` should exist only as an ablation / appendix mode

## B. Clip Length Must Be Swept

Do not publish one arbitrary frame count.

Use at least:

- `T in {8, 16, 32, 64}`

If compute is tight:

- primary figure: `T in {8, 16, 32}`
- appendix: `64`

Why:

- small `T` reflects common short training clips
- larger `T` reveals the GPU amortization regime
- the crossover point itself is part of the paper result

## C. CPU Scaling Must Be Swept

Do not keep video evaluation locked to single-thread-only CPU and call that complete.

For end-to-end benchmarks, sweep:

- `num_workers in {0, 1, 2, 4, 8}`
- extend to `16` on larger machines if available

Also record:

- physical/logical CPU counts
- OpenCV threads per worker
- pinned memory on/off
- prefetch factor

Important distinction:

- transform-only benchmark can remain single-threaded for kernel fairness; treat it as a profiler for algorithmic
  implementation quality, not as the main user-facing throughput claim
- pipeline benchmarks must include worker scaling and production-style library threading

Do not make the main pipeline table an artificial single-core comparison. Albumentations gets production parallelism through dataloader workers; torchvision, Kornia, OpenCV, and DALI should also be allowed to use their normal or recommended production threading/execution model. Record internal thread counts, dataloader workers, and CPU utilization so the result is auditable. If needed, add a controlled appendix where internal threads are forced to `1`, but keep the main pipeline figure user-facing.

For still-image RGB benchmarks in the broader paper package:

- dataset source: same convention as `imread_benchmark` — download `ILSVRC2012_img_val.tar`, unpack it to `imagenet/val`, and point benchmark runs at that directory;
- micro/profiler: `2,000` images from the unpacked ImageNet validation set, preloaded, one internal thread for every library;
- pipeline/user guidance: full unpacked ImageNet validation set (`50,000` images) from disk, worker sweeps, production-style threading;
- fallback subset for iteration: at least `10,000` images, clearly labeled as a sweep/smoke, not the final paper run.

RGB micro hardware should be representative of CPUs that feed model training jobs, not a survey of every available cloud
CPU. Use Apple Silicon locally, modern Intel and AMD cloud CPUs (`c4-standard-16`, `c4d-standard-16`), cloud Arm
(`c4a-standard-16`) only if Arm portability is claimed, and the host CPUs on GPU training machines (`g2-standard-16`,
`a2-highgpu-1g`) if comparing against L4/A100 pipeline behavior. The more important paper results are DataLoader image
pipelines and video augmentation on GPU, especially torchvision video GPU execution.

## D. Decode Must Be Separate From Transform

Need three data regimes:

1. predecoded / preloaded clips
2. decoded-on-the-fly clips
3. compressed video files with random access sampling

This will let us say when augmentation differences matter and when decode hides them.

## E. Metrics To Record

Every benchmark result should record:

- library
- transform
- transform mode (`clip_consistent` / `frame_independent`)
- clip length
- spatial size
- batch size
- dtype
- layout
- device
- num_workers
- pin_memory
- prefetch_factor
- decode backend
- predecoded vs compressed input
- median throughput
- p95 latency if feasible
- mean step time
- host RAM peak
- GPU memory peak

For end-to-end pipeline benchmarks also record:

- time spent waiting on dataloader
- copy time
- decode time if separately instrumented
- train-step time

## Concrete Implementation Plan

## Phase 1. Transform semantics and clip control in the current video benchmark

**Done in repo:** AlbumentationsX and Albumentations (MIT) use `transform(images=video)["images"]` for clip-consistent augmentation.

Files likely touched for the rest:

- `benchmark/transforms/torchvision_video_impl.py` (audit clip consistency per transform)
- `benchmark/runner.py`
- `benchmark/utils.py`

Remaining tasks:

- store transform mode (`clip_consistent` / `frame_independent`) and clip length in output metadata
- add CLI flags for clip-length truncation / frame sampling
- stop hiding frame count inside the dataset itself
- optional: explicit `frame_independent` code path for ablation only

Acceptance criteria:

- default benchmark remains clip-consistent for Albumentations batch API
- results JSON includes enough metadata to distinguish semantic mode when multiple modes exist
- current benchmark still runs as a fast transform-only benchmark

## Phase 2. Add a "fair transform-only" video benchmark config

Tasks:

- create a standard short-clip benchmark config for the paper
- choose default clip lengths representative of common training
- generate outputs for CPU and GPU libraries under matched semantics

Recommended default:

- paper main table: `T=16`
- scaling figure: `T in {8, 16, 32, 64}`

Reason:

- short enough to make CPU pipelines look realistically usable
- large enough to reveal GPU crossover

## Phase 3. Add dataloader + copy benchmark

New code likely needed:

- `benchmark/video_pipeline_benchmark.py`
- `benchmark/video_dataset.py`
- `benchmark/video_collate.py`
- `tests/...` for pipeline config sanity

Tasks:

- build a benchmark dataset that returns clips in a library-neutral decoded representation
- apply library-specific transforms in worker process or main process as appropriate
- measure:
  - dataset iteration time
  - batch collation time
  - host-to-device copy time
- support worker sweep and pin-memory sweep

Critical fairness rule:

- decode and sampling should be shared before library-specific transform application whenever possible

That avoids accidental "decode backend benchmark" leakage into the transform-only comparison.

## Phase 4. Add decode + dataloader + copy benchmark

Tasks:

- add on-the-fly decode benchmark path from compressed videos
- support uniform temporal sampling with explicit sampled frame count
- log decode backend and sampling policy in metadata
- compare against predecoded benchmark to isolate decode cost

Main paper figure:

- stacked time breakdown:
  - decode
  - augmentation
  - collation/copy

## Phase 5. Add decode + dataloader + copy + train-step benchmark

Tasks:

- create a tiny, stable training harness with one video backbone
- benchmark a fixed number of steps, not full epochs
- report steady-state train throughput and dataloader stall time

Suggested setup:

- one lightweight model
- one heavier model
- fixed optimizer and mixed precision policy
- fixed input size and clip length

The goal is not SOTA accuracy. The goal is to measure pipeline bottlenecks.

## Phase 6. Plotting and paper outputs

Need figures, not just tables.

Must-have figures:

1. throughput vs clip length for transform-only benchmark
2. throughput vs num_workers for dataloader + copy benchmark
3. stacked breakdown for decode / augment / copy / train
4. regime map showing when CPU Albumentations remains competitive vs when GPU wins

Must-have appendix tables:

- full transform table for `T=16`
- semantic-mode ablation (`clip_consistent` vs `frame_independent`)
- hardware + software config

## Proposed Repo Structure

Add a dedicated paper workspace:

```text
papers/realistic-video-benchmarking/
  plan.md
  outline.md
  figures/
  notes/
```

And likely new benchmark code:

```text
benchmark/
  video_pipeline_benchmark.py
  video_dataset.py
  video_trainstep_benchmark.py
```

## Open Decisions

These should be resolved early because they change the whole matrix:

1. What should be the paper's default clip length: `8`, `16`, or `32`?
2. Do we want one canonical resolution or a small sweep?
3. Should decode be standardized through one shared backend first, or benchmark each ecosystem's natural decode path?
4. Which training harness should be the end-to-end benchmark:
   - minimal synthetic classifier
   - lightweight action-recognition model
   - both light and heavy models

## Recommended Default Decisions

Unless experiments prove otherwise, start with:

- default paper clip length: `16`
- clip-length sweep: `8, 16, 32, 64`
- one canonical spatial resolution for the first paper pass
- transform semantics: `clip_consistent`
- transform-only benchmark kept as a separate microbenchmark section
- end-to-end benchmark reported as the main practical result

## Minimal Milestone Order

1. ~~Fix Albumentations video semantics~~ (done: `transform(images=video)["images"]`).
2. Add clip-length control and rerun transform-only video benchmarks at realistic `T`.
3. Add dataloader + copy benchmark with worker sweep.
4. Add decode + dataloader + copy benchmark.
5. Add short train-step benchmark.
6. Generate figures and write paper outline.

## What Success Looks Like

At the end, the paper should be able to say all of the following without hand-waving:

- Albumentations is competitive or preferable in realistic short-clip CPU-centered regimes.
- GPU augmentation wins only after specific clip-length / batch / pipeline crossover points.
- transform-only benchmarks overstate GPU advantages when copy and decode are ignored.
- temporal consistency semantics matter and must be standardized for fair video comparisons.
