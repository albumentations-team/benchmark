---
name: paper-coverage-validator
description: Validates whether benchmark artifacts cover the paper's required RGB micro and RGB DataLoader sections. Use when checking missing RGB runs, deciding what to run next, validating gcp_runs/output folders, or preparing paper tables.
---

# Paper Coverage Validator

Use `tools.check_paper_coverage` to avoid manual folder inspection.

## Command

```bash
python -m tools.check_paper_coverage gcp_runs output
```

Validate the RAM-only reduced production-path pass:

```bash
python -m tools.check_paper_coverage --profile ram-reduced gcp_runs output
```

Require optional libraries such as DALI:

```bash
python -m tools.check_paper_coverage gcp_runs output --require-optional-libraries
```

## Required Paper Core Sections

- `image-rgb/micro`: AlbumentationsX, torchvision, Kornia, Pillow; summary + pyperf.
- `image-rgb/pipeline`: AlbumentationsX, torchvision, Kornia, Pillow.

9-channel and video benchmarks are still supported by the codebase, docs, and transform sets, but they are not required
for the current RGB paper artifact.

## RAM-Reduced Profile

`--profile ram-reduced` intentionally checks only:

- RGB micro.
- RGB `memory_dataloader_augment`.

It does not require decode DataLoader, 9-channel, or video runs.

## What To Do With Failures

- Missing directory: run the scenario/mode.
- Missing summary JSON: rerun or fetch `results/` from the GCS run prefix.
- Missing pyperf JSON for a micro run: do not publish as final profiler data until raw pyperf is recovered or rerun.
