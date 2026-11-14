# Greenfield plan

## Goal

This repository measures how quickly RGB JPEGs on a VM's local disk become synchronized CUDA
batches ready for a model. The results help readers choose an augmentation implementation. The
benchmark stops at the model input, so it measures neither training speed nor model quality.

Only RGB is runnable today. Nine-channel images, video, and volumes are future benchmark
families. Each needs its own input format, recipes, dataset, batch size, and output contract
before it has runnable code, data, or results in the repository.

The [execution contract](benchmark_execution_contract.md) defines the required behavior. This
plan explains how the project fits together and what a new family needs before it can run.

## Current RGB run

One production cell is identified by `(family, implementation, recipe, seed)`. In a single pass
it measures throughput and peak process GPU memory, validates `CUDA float16 B×3×224×224` output,
and immediately publishes immutable JSON to GCS. A subsequent launch validates published cells
and runs only the missing ones.

The fixed workload uses:

- The first 10,000 JPEGs in lexicographic order from ImageNet validation.
- 57 manually matched recipes and seven implementations.
- A batch size of 256, one warm-up batch, and 32 measured batches.
- Seeds 137, 138, and 139.
- One Standard `g2-standard-16` VM with an NVIDIA L4.
- 15 persistent DataLoader workers, `pin_memory=True`, and `prefetch_factor=2`.

One VM restores or builds the locked RGB environment and downloads the source archive and
dataset once. It reads the selected files outside the timer to prewarm them, then processes the
missing cells in sequence. Each job contains one library implementation and all of its recipes
and seeds. Jobs do not split the work by duration, recipe, or seed.

## Data path and measurement

```text
JPEG on disk
  → library reader and decoder
  → recipe
  → collate
  → pinned H2D transfer
  → remaining GPU recipe
  → GPU normalization
  → output validation and CUDA synchronization
```

Throughput excludes `Compose` initialization, imports, and preflight. The NVML monitor starts
before pipeline construction. The GPU memory peak therefore covers construction, reader,
decoder, workers, buffers, and CUDA allocations.

Normalization always runs on the GPU. The AlbumentationsX, Pillow, and TorchVision CPU paths
transfer `uint8`. Kornia CPU runs its recipe in `float32` and casts the assembled host batch to
`float16` before H2D. The TorchVision and Kornia GPU paths run only the CPU stages needed to
make samples batchable. They run the remaining recipe and normalization on the GPU in `float16`. DALI uses its native file reader, mixed decoder, and GPU graph.

## Architecture

```text
family config (archive + selection) + cloud config + recipe catalog + environment lock
                                      |
                                      v
                             immutable RunRecord and matrix
                                      |
                                      v
                               one resumable GCP L4 VM
                                      |
                                      v
                              immutable GCS cell records
                                      |
                                      v
                     aggregation and paper from complete rows
```

`run_config.py` defines the family settings. `frozen_rgb_run.py` hashes the inputs into the run
record. `gcp_controller.py` provisions or resumes one labeled VM. `guest_worker.py` runs missing
cells. `rgb_executor.py` defines the timed path and validates its output. The adapters provide
the reader, decoder, and recipe pipeline for each library.

## Invariants

1. Changing the recipes, archive identity or selection rule, package lock, code archive,
   hardware, or timing boundary creates a new run record.
2. Throughput and GPU memory belong to the same cell and are measured together.
3. Microbenchmarks, isolated H2D transfers, training loops, capacity sweeps, and decoder-only
   matrices answer separate questions outside production.
4. Each successful cell is published to GCS immediately. A relaunch reads and validates that
   cell and reuses its result.
5. Throughput and GPU memory are compared on the intersection of supported recipes. Coverage
   comes from a separate catalog census.
6. If a cell is missing or invalid, fix the execution path and obtain the data before
   interpreting it in the paper.

## Adding a benchmark family

For nine-channel images, video, or volumes, start with a dataset archive in one documented
sample format. Then add:

1. A family config with the output shape, dtype, batch size, throughput unit, seeds, and timing
   window.
2. A compact data selection rule, recipe catalog, and support matrix.
3. Implementations with GPU-only normalization and output validation.
4. One L4 preflight batch for each implementation–recipe pair. Paths other than DALI use one
   temporary DataLoader worker. DALI uses its native graph.
5. A new immutable run and a complete matrix.

Choose the batch size, data sources, and recipes for the new family and measure its results
independently.
