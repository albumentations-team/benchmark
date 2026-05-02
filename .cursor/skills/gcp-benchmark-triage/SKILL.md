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

Check GPU quota:

```bash
python - <<'PY'
import json
import subprocess

raw = subprocess.check_output(
    ["gcloud", "compute", "project-info", "describe", "--project", "albumentations", "--format=json"],
    text=True,
)
for quota in json.loads(raw).get("quotas", []):
    metric = quota.get("metric", "")
    if "CPU" in metric or "GPU" in metric or "NVIDIA" in metric:
        print(f"{metric}: limit={quota.get('limit')} usage={quota.get('usage')}")
PY
```

## Common Gotchas

- `c3-standard-16` does not exist; use `c3d-standard-16` or `c3-standard-22`.
- G2/A2/A3/A4 GPU machine types require `--maintenance-policy TERMINATE`. They also need a CUDA-capable image; do not
  treat `g2-standard-16` as a CPU VM just because `--gcp-gpu-type` is omitted.
- `Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.` means the project has zero GPU quota anywhere. Changing zone
  will not help; request global GPU quota and regional L4/G2 quota before retrying.
- Current quota is **64 vCPUs** and **1 GPU**. CPU quota errors can still happen when overlapping VMs, stale
  `STOPPING` instances, or a second launch while a blipped client thinks the VM is gone consume the remaining vCPUs.
  A `g2-standard-16` GPU VM also consumes 16 vCPUs.
- `No image files found in dataset tarball` for a video scenario usually means the detached `job.json` has the wrong typed
  `run_config.selection.scenario` or media-derived staging fields. Detached runs should not carry flag argv payloads.
- Result directories contain both summary JSON and raw pyperf JSON; docs should load only `*_results.json`.
- Do not assume VM disappearance means success.
