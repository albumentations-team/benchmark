# Augmentation input-pipeline benchmark

This benchmark measures eight ways to turn ImageNet JPEG files into a normalized batch on one NVIDIA L4 GPU. For each path it records how many images per second reach the GPU, how much GPU memory the process holds, and how fast a ResNet trains when that path feeds it. The paths are AlbumentationsX, Pillow, Pillow-AllSIMD, TorchVision and Kornia on the CPU and on the GPU, and NVIDIA DALI.

The author maintains AlbumentationsX. Every setting is fixed in the [execution contract](docs/benchmark_execution_contract.md) before a run, and the code, run manifests, and per-seed results are public. The [preprint](paper/main.tex) describes the method and results in full.

![When a batch arrives late, the GPU sits idle and completes fewer steps in the same time.](docs/generated/batch-wait.png)

The data loader prepares the next batch while the GPU trains on the current one. If the GPU finishes first, it waits. How long it waits depends on the JPEG decoder, the augmentations, where they run, and how many CPU cores prepare batches. Timing single operators cannot answer that, so this benchmark times the whole path, from files on disk to a batch the model can use.

<!-- BEGIN GENERATED RESULTS -->

## Results

- **Input pipeline alone.** DALI delivers the most images per second: 5,857, the median over the 31 recipes all paths share, while holding 912 MiB of GPU memory. The CPU paths deliver 1,317 to 3,124 images/s and hold 300 to 440 MiB. DALI decodes only the crop window when a recipe starts with a crop: 5,906 images/s on 21 recipes, against 4,825 on the 10 that need the whole image.
- **ResNet-18 training step (1,706 images/s without input).** AlbumentationsX and Pillow-AllSIMD keep the step at 95% or more of that rate on every recipe. On their heaviest recipe the others reach 15% (Kornia GPU), 23% (TorchVision GPU), 36% (Kornia CPU), 55% (TorchVision CPU), 80% (Pillow) and 92% (DALI).
- **ResNet-50 training step (528 images/s without input).** AlbumentationsX, Pillow, TorchVision CPU, Kornia CPU, DALI and Pillow-AllSIMD keep the step at 95% or more of that rate on every recipe. On their heaviest recipe the others reach 36% (Kornia GPU) and 49% (TorchVision GPU).

2,105 input-pipeline measurements cover 421 path–recipe pairs with 5 seeds each. Every measurement records throughput and peak process GPU memory in the same pass. [Per-seed results](docs/generated/results.json) · [training-step results](docs/generated/training-summary.md)

### Input pipeline alone

The timer runs from JPEG files on local disk to a normalized CUDA `float16` batch of 256×3×224×224. Each path uses its own decoder and places its work on the CPU or the GPU.

![Images per second on the 31 recipes all paths share: one dot per recipe, the bar marks the median. Right: peak process GPU memory.](docs/generated/common-summary.png)

Medians over the 31 recipes that all eight paths support:

| Path | Version | Images/s | Slowest to fastest recipe | Peak GPU memory (MiB) |
| --- | --- | ---: | ---: | ---: |
| DALI | 2.4.0 | 5,857 | 4,523–6,051 | 912 |
| AlbumentationsX | 2.6.9 | 3,124 | 2,542–3,418 | 300 |
| Pillow-AllSIMD | 13.0.0.post4 | 2,737 | 1,701–3,077 | 330 |
| TorchVision CPU | 0.29.2 | 2,524 | 1,690–3,012 | 330 |
| TorchVision GPU | 0.29.2 | 2,500 | 888–4,796 | 1,250 |
| Pillow | 13.0.0 | 2,499 | 579–3,005 | 330 |
| Kornia GPU | 0.9.2 | 1,568 | 724–1,833 | 538 |
| Kornia CPU | 0.9.2 | 1,317 | 476–1,830 | 440 |

### Each path against AlbumentationsX

The author maintains AlbumentationsX (AX), and AX transforms define the reference effect of each recipe. This section therefore compares every path with AX on the recipes both support. A recipe counts as faster or slower only when the exact 95% interval over seeds excludes equal throughput; the rest are tied.

![Each path's throughput divided by AX's on every recipe both support, log scale. Marks show whether the exact 95% interval over seeds excludes equal throughput.](docs/generated/ax-pairs.png)

| Path | Shared recipes | Path faster | Tied | AX faster | Median ratio, path / AX |
| --- | ---: | ---: | ---: | ---: | ---: |
| DALI | 51 | 51 | 0 | 0 | 1.82 |
| Pillow-AllSIMD | 46 | 1 | 2 | 43 | 0.90 |
| TorchVision GPU | 42 | 3 | 5 | 34 | 0.82 |
| TorchVision CPU | 42 | 0 | 0 | 42 | 0.81 |
| Pillow | 46 | 1 | 0 | 45 | 0.78 |
| Kornia GPU | 61 | 0 | 1 | 60 | 0.51 |
| Kornia CPU | 61 | 0 | 0 | 61 | 0.38 |

### Feeding a training step

Here every delivered batch feeds one training step of an untrained torchvision ResNet: `torch.compile`, bfloat16 autocast, SGD, on the same L4. The *ceiling* is the same step on a batch already in GPU memory, measured in the same process before the loader starts. A path at 100% keeps the model fully fed; below that, the model waits for input or shares the GPU with the path's own work.

![Training-step throughput as a share of the model-only ceiling, per path and recipe](docs/generated/training.png)

**ResNet-18, batch 256.** Ceiling 1,706 images/s. Each cell is the share of the ceiling, with images/s in parentheses.

| Path | Crop + flip | + ColorJitter | + ColorJitter + Erasing | + RandAugment |
| --- | ---: | ---: | ---: | ---: |
| AlbumentationsX | 100% (1,749) | 100% (1,744) | 100% (1,738) | 100% (1,726) |
| Pillow-AllSIMD | 98% (1,664) | – | – | 99% (1,656) |
| DALI | 97% (1,669) | 92% (1,565) | 92% (1,562) | 96% (1,631) |
| Pillow | 98% (1,697) | – | – | 80% (1,379) |
| TorchVision CPU | 98% (1,706) | 56% (964) | 55% (933) | 89% (1,503) |
| Kornia CPU | 82% (1,401) | 39% (680) | 36% (626) | 46% (799) |
| TorchVision GPU | 61% (1,048) | 26% (439) | 23% (378) | 34% (580) |
| Kornia GPU | 85% (1,469) | 17% (287) | 17% (283) | 15% (253) |

**ResNet-50, batch 128.** Ceiling 528 images/s. Each cell is the share of the ceiling, with images/s in parentheses.

| Path | Crop + flip | + ColorJitter | + ColorJitter + Erasing | + RandAugment |
| --- | ---: | ---: | ---: | ---: |
| AlbumentationsX | 99% (532) | 99% (527) | 100% (527) | 100% (527) |
| Pillow-AllSIMD | 99% (525) | – | – | 100% (528) |
| Pillow | 99% (532) | – | – | 99% (525) |
| TorchVision CPU | 99% (532) | 99% (528) | 99% (526) | 99% (526) |
| Kornia CPU | 98% (523) | 99% (524) | 99% (524) | 99% (524) |
| DALI | 99% (522) | 98% (515) | 98% (514) | 99% (520) |
| TorchVision GPU | 83% (439) | 53% (279) | 49% (255) | 63% (331) |
| Kornia GPU | 98% (521) | 40% (209) | 39% (207) | 36% (191) |

A dash means the path has no implementation of that recipe. Kornia GPU and TorchVision GPU augment on the GPU inside the training loop, so their work takes GPU time from the step.

### Measurement settings

One Standard `g2-standard-16` VM with one `nvidia-l4` per job. Input pipeline: 49,999 sorted ImageNet validation JPEGs (all but one CMYK file), 30 warm-up batches and 100 measured batches (25,600 images). Training step: the same files, 10 warm-up and 150 measured batches. Seeds: 137, 138, 139, 140, 141; every path reads the same file order for a given seed. DataLoader paths use 15 workers with prefetch factor 2; DALI uses the same number of threads.

NVML samples the process's GPU memory every 50 ms from before pipeline construction to the end of the cell. Short peaks between samples can be missed. CPU memory is not measured. Training runs: `8b5152c09b0e` (ResNet-18), `eb35667fb31d` (ResNet-50). Input-pipeline run: `d84d75894f2a`.

<!-- END GENERATED RESULTS -->

## How to read the results

- **Units.** Throughput is images per second. Each path–recipe pair is the median over five seeds. Each seed ran as a separate job, usually on a different VM, so the spread includes machine-to-machine variation.
- **Ties.** A path counts as faster or slower than another on a recipe only when the exact 95% interval of their throughput ratio over seeds excludes 1. Tied means the seeds cannot separate the two; it does not prove they are equal. The intervals assume independent seeds and paths that differ by a scale factor; the preprint's section *Reading the numbers* states the assumptions.
- **Same effect, different pixels.** AlbumentationsX transforms define what each recipe should do. The other paths use their own operators with parameters set toward that effect, so crop rounding, interpolation arithmetic, and samplers still differ. The [contract](docs/benchmark_execution_contract.md#rgb-workload-and-reference-implementations) lists the differences.
- **Training ceiling.** In the training-step results, 100% is the rate of the same compiled step on a batch already in GPU memory, measured in the same process. Below 100%, the model waits for input or shares the GPU with the path's own work.
- **One operating point.** One L4, 16 vCPUs, 15 workers, files in the page cache. Another GPU, more or fewer cores, cold storage, or larger images can change the ranking. The benchmark does not measure image quality or model accuracy.

## What is measured

| Path | JPEG decoder | Before the copy to the GPU | On the GPU |
| --- | --- | --- | --- |
| AlbumentationsX | SimpleJPEG | Augment `uint8`, `ToTensorV2`, copy `uint8` | Normalize |
| Pillow, Pillow-AllSIMD | Pillow (the AllSIMD build for the second) | Augment, copy `uint8` | Normalize |
| TorchVision CPU | TorchVision I/O | Augment `uint8`, copy | Normalize |
| TorchVision GPU | TorchCodec, hardware nvJPEG | Read JPEG bytes | Decode, augment each image, normalize |
| Kornia CPU | Kornia Rust reader | Augment `float32`, copy `float16` | Normalize |
| Kornia GPU | Kornia Rust reader | Decode, crop | Augment, normalize |
| DALI | Mixed CPU/GPU decoder | Read | Augment and normalize in the DALI graph |

The input-pipeline workload uses 49,999 ImageNet validation JPEGs (all but one CMYK file) and the 72 recipes in the [RGB catalog](catalog/recipes/rgb.yaml). Most recipes add one augmentation to a crop or resize, so each result isolates that augmentation's cost. The training-step workload uses the same files and a [ladder of training recipes](catalog/recipes/imagenet_train.yaml): random resized crop and flip, then ColorJitter, then random erasing, and RandAugment.

A [transformation census](catalog/operations.yaml) records which AlbumentationsX operations each other library can express, whether or not the benchmark times them.

## Reproduce

Read the [execution contract](docs/benchmark_execution_contract.md) before changing or launching a run. With `uv` and an authenticated `gcloud`, from a clean checkout:

```bash
uv run augbench launch --config configs/families/rgb.yaml
```

```bash
uv run augbench launch --config configs/families/train-resnet18.yaml
```

```bash
uv run augbench launch --config configs/families/train-resnet50.yaml
```

Each command freezes the code, data, and package lock into a run identity, provisions up to 20 L4 VMs, and runs only the cells that are still missing. Running it again resumes the same run. The [report instructions](paper/README.md) rebuild the tables, figures, this section, and the preprint from the stored cells.

## Repository layout

```text
catalog/       recipe catalogs and the transformation census
configs/       workload, model, and GCP configuration
environments/  locked benchmark environment
infra/gcp/     VM bootstrap
src/augbench/  runner, adapters, and fleet controller
paper/         preprint, report generators, generated tables
docs/          execution contract, FAQ, generated figures and results
```

## Citing

If you use this benchmark, its results, or its method, please cite the preprint:

```bibtex
@misc{iglovikov2026rgbinputpipelines,
  title={From JPEG to Training Step: Measuring RGB Input Pipelines on an NVIDIA L4},
  author={Vladimir Iglovikov},
  year={2026},
  eprint={2609.06635},
  archivePrefix={arXiv},
  primaryClass={cs.PF},
  url={https://arxiv.org/abs/2609.06635}
}
```
