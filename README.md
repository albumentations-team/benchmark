# Augmentation Benchmark

This repository measures the practical augmentation path from JPEG files on disk to a synchronized, GPU-ready batch. Its [benchmark-only preprint](paper/main.tex) asks two separate questions: which transformations AlbumentationsX exposes, and how the implementations compare when they can execute a manually matched recipe.

Each successful new production cell records two primary results in the same pass: end-to-end throughput and peak process GPU memory. The metrics belong to the exact `(family, implementation, recipe, seed)` row, so both speed and memory are compared recipe by recipe. A separate memory benchmark is not part of the design.

The current RGB run uses the first 10,000 lexicographically sorted `val/*.JPEG` members of a SHA-256-verified ImageNet-validation archive. All rows use batches of 256 on one Standard `g2-standard-16` VM with an NVIDIA L4. The timed boundary is native read and decode → recipe → DataLoader and collation → pinned host-to-device copy → GPU normalization → synchronized `float16` BCHW batch. Normalize never runs on CPU. For GPU-tail paths, the CPU runs only the explicit recipe prefix required to form collatable samples; no hidden crop is added.

AlbumentationsX decodes with SimpleJPEG; Pillow decodes with Pillow; TorchVision and Kornia decode with `torchvision.io`; DALI uses its native file reader and mixed decoder. The operation parameters are manually matched recipe by recipe.

The benchmark does not measure model-training speed. Microbenchmarks, memory-only paths, isolated H2D runs, and capacity sweeps are outside the current study.

Read [the execution contract](docs/benchmark_execution_contract.md) before changing a run, an input, or an interpretation. It specifies the frozen inputs, the one-VM lifecycle, cache reuse, resume rules, and publication checks.

## Run the RGB matrix

Commit the intended inputs first; production refuses a dirty worktree so that a
run can be reconstructed from its source archive. From the repository root,
run:

```bash
uv run augbench launch-rgb
```

The command validates existing immutable cells, resumes their exact run when
possible, and otherwise creates one labelled L4 VM. It never creates a second
VM for the same active run.

The [greenfield plan](docs/greenfield_plan.md) explains the architecture and execution rules. Production prewarms every prepared dataset file once outside timed cells, then uses the same deterministic pseudo-random file order for every implementation at a given seed. This removes a cold-cache advantage for the first implementation without removing filesystem reads or each library's decoder. After prewarm, file contents may be served by the Linux page cache, so the result is not a physical-disk-bandwidth measurement.

## Repository layout

```text
catalog/       operation census and the RGB recipe catalog
environments/  one locked RGB benchmark environment
infra/gcp/     VM bootstrap
src/augbench/  benchmark implementation
paper/         benchmark-only preprint and bibliography
.codex/skills/ repo-local rules for planning, running, and interpreting the benchmark
```

9-channel images, video, and volumes are independent first-class studies. Before each family becomes runnable, it receives its own prepared input format, recipes, dataset, batch size, implementations, and output contract. Their results are never mixed into RGB or into one another. Every family keeps the same two primary outputs: family-specific throughput and peak process GPU memory from the same production cell.
