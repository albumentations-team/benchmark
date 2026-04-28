---
name: benchmark-results-publisher
description: Publishes benchmark result JSONs into README/docs safely. Use when updating README tables, using MacBook RGB results, preserving Pillow results for the website, running tools/update_docs.sh, or preventing pyperf raw JSON from polluting documentation.
---

# Benchmark Results Publisher

Use this when turning completed benchmark artifacts into documentation.

## Defaults

While the full paper matrix is still running, `tools/update_docs.sh` defaults to MacBook M4 RGB micro results:

```bash
./tools/update_docs.sh
```

Equivalent explicit path:

```bash
./tools/update_docs.sh \
  --image-results output/rgb_micro_macos_m4max/image-rgb/micro
```

Pass `--video-results` only when intentionally updating video README sections.

## Guardrails

- Summary result files are `*_results.json`, including scenario names like `albumentationsx_micro_results.json`.
- Raw pyperf files are `*.pyperf.json`; never load them as README columns.
- RGB public docs should include Pillow when the result exists.
- Do not commit `gcp_runs/`, `output/`, or `output_*/` artifacts.

## Validation

Run after docs changes:

```bash
python -m pytest tests/test_compare.py tests/test_update_readme.py
pre-commit run --all-files
```
