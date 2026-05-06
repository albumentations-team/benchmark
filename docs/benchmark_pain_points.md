# Benchmark Pain Points And Guardrails

This note records practical issues that made augmentation library benchmarking difficult. It is not tied to a specific
paper. Use it as a checklist before changing benchmark scope, adding a library, adding transforms, or interpreting
results.

## Main Pain Points

### Transform coverage is uneven

Different libraries expose different numbers of transforms. A library with a smaller transform catalog can look better in
aggregate because it simply does not implement difficult or slow operations. A library with broader coverage can be
penalized because its slow rows are visible.

Guardrails:

- Report coverage separately from throughput.
- Keep unsupported, missing, and early-stopped rows visible in result summaries.
- Avoid a single global leaderboard when transform denominators differ.
- Prefer paired comparisons over the same transform set whenever possible.

### Transform names are not enough

Libraries often use similar names for operations that are not exactly equivalent. Even after matching names, parameter
defaults may differ: interpolation, border handling, fill values, probability semantics, output dtype, channel layout, RNG
behavior, and clipping rules can all change both quality and speed.

Guardrails:

- Maintain an explicit canonical transform catalog.
- Map each library implementation to that catalog by behavior, not only by name.
- Pin or pass important parameters explicitly instead of relying on library defaults.
- Document known semantic mismatches rather than hiding them inside benchmark code.

### Faithful implementations require real engineering

Some benchmark-side adapters and recipes require substantial effort to match a transform while still keeping the
implementation fast and general. A naive compatibility wrapper can make a library look slower for reasons unrelated to
the library itself.

Guardrails:

- Keep library-specific transform implementations explicit and reviewable.
- Optimize adapters that are part of the measured path.
- Avoid recreating missing transforms for a library just to force full coverage.
- Separate benchmark helper overhead from library transform cost when possible.

### Lazy execution must be forced

Some paths can defer work through lazy graph execution, asynchronous GPU kernels, or unevaluated pipeline outputs. If the
benchmark does not force materialization, it may time scheduling rather than augmentation.

Guardrails:

- Materialize transform outputs inside the timed scope when that scope claims to include the work.
- Synchronize CUDA/MPS work before stopping timers.
- For DALI-like pipelines, make sure the benchmark consumes produced batches.
- State clearly whether tensor batch materialization and device transfer are included.

### Transform-set selection changes the story

Selecting only transforms supported by two or more libraries is more defensible than comparing a library against many
operations nobody else implements. It still creates tradeoffs. Kornia has broad coverage, so slow Kornia rows remain in
the benchmark. DALI has narrower coverage, so it may avoid transforms that are slow in some libraries and look better on
measured rows.

Guardrails:

- Define the transform eligibility rule before running the sweep.
- Publish the fixed transform set used for each scenario.
- Report the coverage denominator, for example `48/57` measured recipes.
- Do not treat "not implemented" as faster than "implemented but slow."

### Very slow transforms can dominate runtime

Some transforms are fast in one library and impractically slow in another. `Elastic` is fast in AlbumentationsX, slow in
Kornia and TorchVision, and not implemented in DALI. Without a preflight guard, a few slow library-specific rows can
dominate total benchmark time and make large sweeps impractical.

Guardrails:

- Run a pre-compute or preflight pass before full timing.
- Early-stop image transforms below the practical throughput floor, currently `20 img/s`.
- Keep early-stopped rows in the output with their reason and preflight throughput.
- Disable slow skipping only for a targeted slow-transform study.

### Micro benchmarks and DataLoader benchmarks answer different questions

Micro benchmarks measure the named transform in isolation. DataLoader benchmarks measure a minimal training-style recipe,
currently `Compose + RandomCrop224 + <transform> + Normalize` for non-crop transforms. For very fast transforms such as
flips, the fixed crop, composition, normalization, collation, and transfer costs can dominate the measured time.

Guardrails:

- Do not use DataLoader recipe results as primitive transform-speed claims.
- Do not use micro results as full training-pipeline claims.
- Keep fixed recipe costs visible when interpreting fast operations.
- Remember that fast primitive transforms still matter in advanced pipelines with many cheap operations, because fixed
  crop and normalize costs are paid once while many internal transforms may accumulate.

### GPU benchmarks have extra semantic constraints

GPU paths are not just CPU paths moved to another device. DALI does not have the same micro-benchmark shape as the
PyTorch libraries. PyTorch DataLoader workers cannot independently apply GPU transforms before collation, so image data
must first be prepared on CPU, usually with crop or resize/crop shape normalization, then collated and copied to GPU.

TorchVision also does not expose a batched random-transform mode that applies different random parameters per sample in
the same way Kornia can with `same_on_batch=False`. To preserve per-sample randomness, TorchVision GPU augmentation may
need a per-sample loop over the batch.

Guardrails:

- State where CPU preparation ends and GPU augmentation starts.
- Include batch copy and synchronization in the scope only when the benchmark mode says so.
- Preserve per-sample random parameters, even if that requires a slower implementation.
- Compare DALI against equivalent pipeline scopes, not against unavailable micro scopes.

### GPU augmentation consumes training memory

GPU augmentation throughput is not the only practitioner-facing cost. Augmentation kernels, intermediate tensors, graph
state, and framework caches consume accelerator memory that would otherwise be available for model parameters,
activations, optimizer state, or a larger training batch. A GPU augmentation pipeline can therefore be unattractive even
when it is fast enough, and especially unattractive when RGB production throughput does not beat the CPU DataLoader
baseline.

Guardrails:

- Record peak allocated and peak reserved GPU memory for GPU DataLoader rows.
- Report memory together with throughput for GPU production pipelines.
- Treat memory as part of the production tradeoff, not as a secondary implementation detail.
- Avoid recommending GPU augmentation from microbenchmark speed alone when it reduces feasible training batch size or
  model capacity.

### Pipeline results can hide fast-transform differences

In the minimal DataLoader recipe, very fast transforms may contribute little to the total runtime because crop,
normalization, DataLoader overhead, collation, and device transfer are fixed costs. That does not mean fast-transform
optimization is irrelevant. In richer augmentation policies, many individually cheap transforms can add up.

Guardrails:

- Interpret pipeline rows as end-to-end recipe throughput.
- Use micro rows when evaluating the cost of individual cheap transforms.
- Use richer recipe benchmarks when the question is "what happens with many fast transforms together?"

### Coverage affects who can be second best

AlbumentationsX has the broadest transform coverage in this benchmark. Missing coverage in other libraries can strongly
affect which competitor appears second best, especially when summaries consider only measured rows. The important
conclusion should combine coverage and speed rather than treating missing transforms as neutral.

Guardrails:

- Pair speed claims with support claims.
- Avoid ranking libraries only over the subset each library happens to implement.
- Separate "fast on measured rows" from "broadly usable for this transform universe."

## Future Benchmark Checklist

Before running a new large benchmark sweep:

1. Define the benchmark regime: micro, CPU DataLoader, GPU DataLoader, DALI pipeline, video, or another scope.
2. Freeze the transform eligibility rule and publish the resulting transform list.
3. Match transform names and parameters explicitly across libraries.
4. Decide which operations must be materialized and synchronized inside the timed scope.
5. Run a slow-transform preflight and keep early-stopped rows in the output.
6. Report coverage, unsupported rows, early-stopped rows, and measured throughput separately.
7. Keep micro and pipeline conclusions separate unless the comparison is explicitly about the difference between them.
8. For GPU paths, document CPU preparation, batch transfer, per-sample randomness, and synchronization.
9. For GPU paths, record and report peak allocated/reserved memory.
10. Interpret aggregate rankings only together with coverage denominators and resource costs.
