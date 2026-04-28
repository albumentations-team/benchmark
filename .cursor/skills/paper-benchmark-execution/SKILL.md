---
name: paper-benchmark-execution
description: Executes the paper benchmark plan for RGB, multichannel, DataLoader, and video benchmarks. Use when the user mentions the paper benchmark, deadline plan, machine matrix, RGB micro, multichannel, DataLoader, video GPU, c4/c4d/g2 machines, or what to run next.
---

# Paper Benchmark Execution

Use `_internal/plans/paper_benchmark_execution_plan.md` as the source of truth.

## Rules

- Do not run every benchmark on every CPU.
- CPU rows run on CPU-only machines, usually `c4-standard-16`.
- GPU rows run only for GPU libraries/paths, usually `g2-standard-16` with L4.
- Do not run CPU-only rows on GPU VMs for hardware symmetry; label hardware per row instead.
- Respect the 32-vCPU quota by running at most two 16-vCPU CPU machines at once.
- Treat RGB micro as a profiler, not the main user-facing training throughput table.

## Core Matrix

Already done:

- MacBook M4 RGB micro.
- `n2-standard-16` RGB micro.
- `n2d-standard-16` RGB micro.

Core remaining:

- `c4-standard-16`: CPU-only paper tables.
- `c4d-standard-16`: RGB micro AMD sanity check only.
- `g2-standard-16`: torchvision/Kornia/DALI GPU video rows only.

## Execution Order

1. Inventory existing results and avoid rerunning completed `n2`/`n2d` baselines.
2. Smoke test each scenario with tiny `--num-items`, `--num-runs 1`, and short `--min-time`.
3. Run RGB micro on `c4-standard-16` and `c4d-standard-16`.
4. Run CPU suite on `c4-standard-16`: 9ch micro, RGB DataLoader, 9ch DataLoader, Albumentations video CPU micro.
5. Run GPU suite on `g2-standard-16`: torchvision/Kornia video GPU micro and GPU video DataLoader.
6. Pull and validate artifacts before generating plots/tables.

## Validation

After pulling results, run:

```bash
python -m tools.check_paper_coverage gcp_runs output
```

Use `--require-optional-libraries` only when DALI must be present.
