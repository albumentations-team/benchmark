# Benchmark execution

Before planning, changing, running, resuming, aggregating, plotting, or interpreting a benchmark, read both:

- [`.codex/skills/augmentation-benchmark-contract/SKILL.md`](.codex/skills/augmentation-benchmark-contract/SKILL.md)
- [`docs/benchmark_execution_contract.md`](docs/benchmark_execution_contract.md)

Every successful new production cell must measure end-to-end throughput and peak process GPU memory in the same pass. A separate memory run is forbidden unless the user explicitly adds a different research question.

The active work is RGB `dataloader_disk`. Each production `run_id` freezes its
recipes, hardware, dataset, package lock, code archive, and timing boundary.
Any such change requires a new run; never mutate or mix its existing results.
