# Image Augmentation Benchmark Artifact

This repository contains the code and sanitized RGB result data needed to reproduce the paper figures and submission PDF. The benchmark is framed as an evaluation protocol first: micro timing, CPU DataLoader timing, PyTorch GPU DataLoader timing, and DALI graph timing answer different questions and must not be collapsed into one fastest-library contest.

The main paper result is scoped to ImageNet-size RGB production DataLoader pipelines: AlbumentationsX CPU has the highest measured throughput in this protocol. This should not be generalized to "CPU augmentation is always best," because TorchVision and Kornia CPU measurements are much lower in the same benchmark and different modalities may behave differently.

The GPU story is central to the artifact. GPU microbenchmarks, high GPU utilization, and the fact that an augmentation runs on GPU do not imply good production throughput. GPU augmentation also consumes memory on the same accelerator used for model training, so memory is reported as a production benchmark axis.

The 57 RGB recipes are defined before timing by a library-support rule. The benchmark first enumerates the AlbumentationsX, TorchVision, Kornia, and Pillow RGB transform catalogs and keeps every canonical transform that is directly implemented by at least two of those four libraries. DALI is evaluated against the same fixed universe where it has direct graph support.

Anonymous artifact URL: [https://anonymous.4open.science/r/benchmark-2BF9](https://anonymous.4open.science/r/benchmark-2BF9)

The codebase still includes 9-channel image and video benchmark support, but this anonymous artifact commits only the raw RGB result JSONs used by the current paper figures.

## Paper Data

Committed inputs:

- `docs/paper_raw/`: sanitized raw benchmark summary JSONs.
- `docs/paper_data/`: derived CSV, JSON, and Markdown tables generated from the raw JSONs.
- `docs/paper_figures/`: figures generated from `docs/paper_data/`.

Regenerate derived data, figures, and the LaTeX PDF:

```bash
python scripts/paper/generate_paper_data.py
python scripts/paper/generate_figures_and_insights.py
cd _internal/paper/neurips_2026_ed
latexmk -pdf main.tex
```

The main generated figures are:

![Open production DataLoader category](docs/paper_figures/open_dataloader_leaderboard.png)

![Supported recipes versus measured throughput](docs/paper_figures/coverage_vs_throughput.png)

![Paired GPU DataLoader throughput ratios](docs/paper_figures/gpu_vs_albumentationsx_cpu_ratios.png)

![GPU memory consumed by augmentation pipelines](docs/paper_figures/gpu_memory_vs_throughput.png)

![Rank-one counts by benchmark regime](docs/paper_figures/winner_counts.png)

## How To Interpret Results

- Support means the library/backend directly implements the recipe in that regime.
- Measured means support plus successful timing at or above the 20 images/s preflight throughput floor.
- Recipes not supported by a given implementation and early-stopped recipe-implementation pairs are not zero-throughput measurements. They are absent from that implementation's measured recipe count and should be read next to throughput. The 20 images/s floor is a pragmatic cutoff for very slow training-pipeline measurements, not a correctness failure.
- The 57 recipes were fixed before timing using the 2+ library support rule.
- The headline RGB result applies to the production DataLoader protocol in this artifact. Do not generalize it to video, multichannel data, very large images, native graph pipelines, or different hardware without measuring those regimes.
- GPU microbenchmark rank-one results do not imply production DataLoader rank-one results. Production timing includes CPU batch preparation, transfer, synchronization/materialization, random-parameter semantics, and GPU memory use.
- High GPU utilization is not enough evidence that augmentation is efficient. It can also mean slow augmentation kernels are occupying the accelerator while end-to-end image throughput remains poor.

## Setup

The runner creates one isolated virtual environment per compatible library group. For example, AlbumentationsX, TorchVision/Kornia/Pillow, and DALI use separate `.venv_*` directories so dependency conflicts do not leak between libraries. Requirement files in `requirements/` define those environments; by default, the runner refreshes compiled locks from the matching `.in` files before installing.

Install the control environment:

```bash
uv venv --python 3.13
uv pip install --python .venv -r requirements-dev.txt
```

Use `--no-refresh-requirements` for repeat local runs when the existing lock files and `.venv_*` environments should be reused.

## Data Layout

For local RGB runs, point `--data-dir` at a directory of images, such as an extracted ImageNet validation directory:

```bash
python -m benchmark.cli plan --config configs/examples/local_rgb_micro_cpu.yaml
```

The plan command validates the config, expands `transform_set: paper`, shows the jobs that would run, and lists expected result files without measuring anything.

ImageNet is not redistributed in this artifact. Users must provide local validation images or configure their own dataset path/object storage location.

## RGB Micro

RGB micro benchmarks preload decoded images into each library's native representation and time augmentation only. CPU micro runs force one internal thread per library. TorchVision and Kornia can also run device-resident CUDA/MPS micro timing.

CPU micro:

```bash
python -m benchmark.cli run \
  --config configs/examples/local_rgb_micro_cpu.yaml \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro_cpu
```

GPU micro:

```bash
python -m benchmark.cli run \
  --config configs/examples/local_rgb_micro_gpu.yaml \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro_gpu
```

## RGB DataLoader

RGB DataLoader benchmarks measure training-style recipes with `RandomCrop224 + transform + Normalize + ToTensor`. The CPU and TorchVision/Kornia GPU DataLoader examples use `memory_dataloader_augment`, so images are preloaded before timing and the timed scope excludes disk read/decode. GPU DataLoader measurements keep crop/pad preparation in workers, copy fixed-shape batches to the device, and time the GPU augmentation path. DALI currently uses the file/decode graph path.

CPU DataLoader:

```bash
python -m benchmark.cli run \
  --config configs/examples/local_rgb_dataloader_cpu.yaml \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_dataloader_cpu
```

TorchVision/Kornia GPU DataLoader:

```bash
python -m benchmark.cli run \
  --config configs/examples/local_rgb_dataloader_gpu.yaml \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_dataloader_gpu_memory
```

DALI GPU DataLoader:

```bash
python -m benchmark.cli run \
  --config configs/examples/local_rgb_dataloader_dali.yaml \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_dataloader_dali
```

DALI requires a CUDA-capable machine and the DALI requirement group to install successfully.

## Cloud Runs

Cloud configs under `configs/paper/` use placeholder project and bucket values. Replace them locally or pass overrides:

```bash
python -m benchmark.cli run \
  --config configs/paper/prod_c4_rgb_dataloader_cpu.yaml \
  --gcp-project YOUR_PROJECT \
  --gcp-gcs-data-uri gs://YOUR_BUCKET/datasets/imagenet/val.tar \
  --gcp-gcs-results-uri gs://YOUR_BUCKET/benchmark-runs \
  --gcp-dry-run
```

Detached cloud runs stage one dataset tarball onto the VM local disk before benchmarking. They upload result artifacts to the configured results prefix and delete the VM by default after completion.

## 9-Channel And Video Code

The implementation still supports 9-channel image and video scenarios:

```bash
python -m benchmark.cli plan --config configs/examples/local_9ch_micro_cpu.yaml
python -m benchmark.cli plan --config configs/examples/local_video_micro_cpu.yaml
```

This anonymous artifact does not commit 9-channel, video, or macOS result data. Those regimes should be treated as future or appendix context, not as evidence for the RGB production claim.

## Verification

Useful checks before publishing the anonymous branch:

```bash
python scripts/paper/generate_paper_data.py
python scripts/paper/generate_figures_and_insights.py
python _internal/paper/scripts/check_manuscript_consistency.py
cd _internal/paper/neurips_2026_ed
latexmk -pdf main.tex
python -m pytest tests/test_config_models.py tests/test_matrix.py tests/test_paper_transform_sets_policy.py tests/test_cli.py tests/test_cloud_paths.py tests/test_stage_dataset.py
```
