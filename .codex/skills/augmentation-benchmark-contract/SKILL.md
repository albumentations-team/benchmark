---
name: augmentation-benchmark-contract
description: Use before planning, changing, running, resuming, aggregating, plotting, or interpreting RGB, multichannel, video, or volume augmentation benchmarks in this repository.
---

# Augmentation benchmark contract

Read `docs/benchmark_execution_contract.md` before changing code, launching,
resuming, aggregating, plotting, or interpreting results. It is the source of
truth; this skill is the short operational checklist.

## Current production question

The image runner supports RGB and a separate grayscale Colorize matrix: local ImageNet JPEG →
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

## RGB workload and reference effect

- Treat the RGB workload as image classification on one 8-bit RGB JPEG per sample.
  The augmentation receives only the image. Boxes, masks, keypoints, and
  `additional_targets` are outside this workload.
- Strip EXIF during shared JPEG extraction before timing. All decoders consume
  the prepared files in stored pixel orientation.
- Use stock AlbumentationsX transforms as the reference. Construct one AX transform
  per catalog stage. Its own sampler runs during execution. Use one top-level
  `A.Compose` for the declared recipe. Keep `A.Lambda`, `A.OneOf`, and `A.Sequential`
  out of the AX adapter. Put required conversions in explicit catalog stages.
- Run native TorchVision and Kornia operators. Adjust their public parameters
  toward the AX effect. Add per-image sampling in the adapter when the operator
  accepts fixed values. Keep the native image operation.
  Mark a path unsupported when it would need a custom implementation of the target.
  Assemble the preprocessing paths Pillow, Pillow-AllSIMD, and DALI from native
  primitives. Compare observable outputs and record material differences that
  remain after calibration.
- Check the paths against the [RGB reference policy](docs/benchmark_execution_contract.md#rgb-workload-and-reference-implementations)
  when changing a recipe. Keep throughput claims scoped to the measured image-only workload.

## Non-negotiable execution rules

- **Normalize never runs on CPU.** CPU paths may decode, form a collatable
  shape, and apply their CPU recipe. The collated batch moves to CUDA first.
  AX uses its batch `InputNormalizer` with FP32 arithmetic and FP16 output.
  DALI performs native GPU normalization in the graph. Other paths use
  GPU normalization in FP16.
- The timed output is CUDA `float16` BCHW `B×3×224×224`, materialized and
  synchronized. The catalog's `Normalize` and `ToTensor` entries configure the
  batch output. The `Resize` image recipe contains only `Resize(224)`. AX stacks
  HWC `uint8` images into a BHWC batch, transfers it, then changes the batch view
  to BCHW on CUDA for `InputNormalizer`. Keep `ToTensorV2` out of AX image recipes.
- Measure recipe execution, not `Compose` init or imports. Start the NVML
  process-memory monitor before pipeline construction.
- Keep batch, workers, prefetch policy, recipe parameters, dataset ordering,
  hardware, and timing boundary identical across rows. Pairwise summaries use
  the exact supported recipe intersection; coverage is a separate census.
- The VM runs each implementation in its own process through all of its recipes and
  seeds. It writes every valid cell to GCS immediately. Resume validates those
  immutable cells and runs only the missing ones. Keep each implementation on
  one VM within a run.
- Grayscale Colorize starts with an offline-prepared one-channel JPEG archive. Measure
  native grayscale decode, crop, and Colorize, then normalize the RGB batch on CUDA.
  Keep this dataset and run identity separate from RGB.
- 9ch, video, and volume become runnable
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
smallest correction run that restores the comparison. Before regenerating the
report, validate the replacement cells and preserve their original run and cell IDs.
