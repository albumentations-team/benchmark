---
name: augmentation-benchmark-contract
description: Use before planning, changing, running, resuming, aggregating, plotting, or interpreting RGB, multichannel, video, or volume augmentation benchmarks in this repository.
---

# Augmentation benchmark contract

Read `docs/benchmark_execution_contract.md` before changing code, launching,
resuming, aggregating, plotting, or interpreting results. It is the source of
truth; this skill is the short operational checklist.

## Current production question

The only runnable matrix is RGB `dataloader_disk`: local ImageNet JPEG →
library-native reader and decoder → recipe → collate → pinned H2D → model-ready
CUDA batch. RGB uses one Standard `g2-standard-16` VM with one L4 and the
frozen values in `configs/families/rgb.yaml`.

One cell is exactly `(family, implementation, recipe, seed)`. One pass records
both primary outcomes:

1. end-to-end throughput; and
2. peak process GPU memory from pipeline construction through the final CUDA
   synchronization.

Never create a memory-only, micro, H2D-only, training, capacity, or smoke
matrix unless the user explicitly adds that research question.

## Non-negotiable execution rules

- **Normalize never runs on CPU.** CPU paths may decode, form a collatable
  shape, and apply their CPU recipe. The collated batch moves to CUDA first;
  conversion to `float16` and Normalize then happen on GPU. DALI performs its
  native GPU normalization in the graph.
- The timed output is CUDA `float16` BCHW `B×3×224×224`, materialized and
  synchronized. `Resize` is exactly `Resize(224) → Normalize → ToTensor`.
- Measure recipe execution, not `Compose` init or imports. Start the NVML
  process-memory monitor before pipeline construction.
- Keep batch, workers, prefetch policy, recipe parameters, dataset ordering,
  hardware, and timing boundary identical across rows. Pairwise summaries use
  the exact supported recipe intersection; coverage is a separate census.
- The VM runs one implementation sequentially through all of its recipes and
  seeds. It writes every valid cell to GCS immediately. Resume validates those
  immutable cells and runs only the missing ones. Do not split by duration,
  recipe, seed, or arbitrary shard.
- The current runner exposes only RGB. 9ch, video, and volume become runnable
  only after each has its own data format, config, recipes, GPU-only Normalize
  implementation, and one-batch L4 preflight per implementation-recipe pair:
  non-DALI uses one temporary DataLoader worker and DALI uses its native graph.

## Before a production launch

1. Confirm a clean Git worktree and build the source archive from that commit.
2. Validate the RGB config (including archive identity and selection), catalog, lock, and matrix.
3. Verify output validation and the same-pass GPU-memory field in the result
   schema.
4. Use `augbench launch-rgb`; it searches eligible zones, protects an active
   labelled VM from duplication, and resumes only validated missing cells.
5. If the VM exits before a cell, inspect its `runs/<run_id>/logs/` bootstrap
   log; do not retry blindly.
6. A new run may reclaim only terminal or suspended `augbench` VMs; never
   delete active VMs or GCS result artifacts.

If a required measurement is absent, repair the execution path and generate the
smallest missing part of the production matrix. Never substitute prose for data.
