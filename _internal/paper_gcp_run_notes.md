# Paper GCP Run Notes

Internal notes for paper-deadline benchmark runs. Use checked-in YAML configs as
the source of truth; CLI flags are only overrides for those configs.

## GCP Defaults

- Project: `albumentations`
- Default zone in the RGB micro proxy script: `us-central1-b`
- CLI default zone if not specified: `us-central1-a`
- ImageNet validation tarball: `gs://imagenet_validation/imagenet/val.tar`
- Results base prefix: `gs://imagenet_validation/augmentation-results`
- Default venv cache prefix for that results URI: `gs://imagenet_validation/augmentation-cache`
- Local GCP run output root used by the proxy script: `gcp_runs/`

For paper CPU rows, prefer `c4-standard-16`.

Detached GCP runs download the dataset tarball to the VM, unpack it to local
disk, run the benchmark against the local staged data, upload artifacts, then
self-delete the VM unless `--gcp-keep-instance` is set.

## RGB Micro Reduced Run, Paper Transform Set

One-run coverage pass: all RGB paper libraries, paper transform set, 1,000
images.

```bash
python -m benchmark.cli run --config configs/paper/gcp_c4_rgb_micro_cpu.yaml
```

## RGB DataLoader Reduced Run, RAM-Loaded Data

For the paper, use RAM-loaded data to isolate augmentation and batching from
JPEG decode/filesystem effects.

```bash
python -m benchmark.cli run --config configs/paper/gcp_c4_rgb_dataloader_cpu.yaml
```

## Pending GPU Image Smokes

These four smoke runs are not done yet:

```bash
python -m benchmark.cli run --config configs/paper/gcp_g2_rgb_micro_gpu_smoke.yaml
python -m benchmark.cli run --config configs/paper/gcp_g2_9ch_micro_gpu_smoke.yaml
python -m benchmark.cli run --config configs/paper/gcp_g2_rgb_dataloader_gpu_smoke.yaml
python -m benchmark.cli run --config configs/paper/gcp_g2_9ch_dataloader_gpu_smoke.yaml
```

Run one `g2-standard-16` job at a time unless quota has been increased.

## Completion And Debugging

Each detached run writes `gcp_last_run.json` in the local output directory with
the `run_prefix`, `instance_name`, project, and zone. Remote artifacts land under
`<results-base>/<run_id>/`, including:

- `results/`
- `vm.log`
- `run_meta.json`
- `exit_code.txt`
- `DONE` or `FAILED`

Useful commands:

```bash
gcloud storage cat gs://imagenet_validation/augmentation-results/<run_id>/vm.log
gcloud storage rsync --recursive gs://imagenet_validation/augmentation-results/<run_id>/results/ gcp_runs/<local-run-dir>/
```
