---
name: gcp-benchmark-triage
description: Triage detached GCP benchmark runs, DONE/FAILED sentinels, VM cleanup, vm.log, gcp_last_run.json, and partial result downloads. Use when GCP benchmark logs mention DONE, FAILED, exit_code.txt, VM disappeared, STOPPING, gcloud machine type errors, or missing artifacts.
---

# GCP Benchmark Triage

Detached runs upload artifacts under the run prefix recorded in `gcp_last_run.json`.

## Interpret Run State

- `DONE`: benchmark succeeded; fetch `results/`, then wait for VM cleanup/cache upload.
- `FAILED`: benchmark failed; fetch `vm.log`, `FAILED`, `exit_code.txt`, and any partial `results/`.
- `exit_code.txt` without `DONE`: use the exit code and `vm.log` as source of truth.
- VM gone with no terminal marker: treat as failure until logs prove otherwise.
- VM `STOPPING` after `DONE`: usually cache upload/self-delete; wait before relaunching quota-heavy jobs.

## Useful Commands

Read local metadata:

```bash
python - <<'PY' path/to/gcp_last_run.json
import json, sys
print(json.dumps(json.load(open(sys.argv[1])), indent=2))
PY
```

Fetch artifacts:

```bash
gcloud storage rsync --recursive "$RUN_PREFIX/results/" "$LOCAL_OUT/"
gcloud storage cat "$RUN_PREFIX/vm.log"
gcloud storage cat "$RUN_PREFIX/exit_code.txt"
```

Check machine availability before launching:

```bash
gcloud compute machine-types list \
  --project albumentations \
  --zones us-central1-b \
  --filter='name=(c4-standard-16 c4d-standard-16 g2-standard-16)'
```

## Common Gotchas

- `c3-standard-16` does not exist; use `c3d-standard-16` or `c3-standard-22`.
- Result directories contain both summary JSON and raw pyperf JSON; docs should load only `*_results.json`.
- Do not assume VM disappearance means success.
