# Augmentation Benchmark

## Citing

If you use this code, benchmark results, or methodology in your research, please cite
[RGB Input Pipelines: Throughput, GPU Memory, and Transformation Coverage](https://arxiv.org/abs/2609.06635).
Your citation makes the project's research impact visible to funders and helps sustain maintenance.

```bibtex
@misc{iglovikov2026rgbinputpipelines,
  title={RGB Input Pipelines: Throughput, GPU Memory, and Transformation Coverage},
  author={Vladimir Iglovikov},
  year={2026},
  eprint={2609.06635},
  archivePrefix={arXiv},
  primaryClass={cs.PF},
  url={https://arxiv.org/abs/2609.06635}
}
```

## About the benchmark

Eight RGB input paths take ImageNet JPEGs from disk to a normalized CUDA batch.
Each measurement records throughput and peak process GPU memory together.
The CPU paths include AlbumentationsX, Pillow, and Pillow-AllSIMD in fast mode.
TorchVision and Kornia each have a CPU and a GPU path. DALI uses its native GPU pipeline.

See the [current manuscript](paper/main.tex), [per-recipe results and seed observations](paper/generated/recipe-results.csv),
and [reproduction instructions](paper/README.md). The [September preprint](https://arxiv.org/abs/2609.06635)
reports the earlier seven-path experiment.

<!-- BEGIN GENERATED RESULTS -->

## RGB results

1314 measurements across 438 implementation–recipe pairs, with 3 seeds per pair. See the [per-seed results and manifests](docs/generated/results.json).

![Ready batches per second and peak process GPU memory on 34 common recipes. Each metric takes the median across seeds, then across recipes. The dashed line marks AX's batch rate.](docs/generated/common-summary.png)

### Mean per-recipe throughput ratios

All rows below use the same 34 recipes. Throughput is the arithmetic mean of per-recipe ratios: each path's median over 3 seeds divided by AX's median for that recipe. GPU memory is the median of those recipes' peak-memory medians.

| Measured path | Version | Mean throughput / AX (higher is faster) | Peak process GPU memory (MiB, lower is better) |
| --- | --- | ---: | ---: |
| DALI GPU | 2.4.0 | 1.18× | 1,872 |
| AlbumentationsX CPU | 2.6.3 | 1.00× | 404 |
| Pillow-AllSIMD CPU (fast) | 13.0.0.post4 | 0.89× | 330 |
| TorchVision GPU | 0.29.2 | 0.84× | 544 |
| TorchVision CPU | 0.29.2 | 0.82× | 330 |
| Pillow CPU | 13.0.0 | 0.74× | 330 |
| Kornia GPU | 0.9.1 | 0.53× | 390 |
| Kornia CPU | 0.9.1 | 0.45× | 292 |

Across 46 shared recipes, Pillow-AllSIMD was 1.66× as fast as Pillow on average, with a 1.16× median ratio. Its R27 MedianBlur ratio was 16.16×.

Each implementation ran in its own process. Peak VRAM includes the CUDA state that process retains. The per-seed baseline before pipeline construction is in the machine-readable results.

### AlbumentationsX against each library

Each chart uses the recipes shared by every path it shows. Compare rows within each chart.

![AX vs Kornia on 67 shared recipes](docs/generated/pairwise-kornia.png)

![AX vs TorchVision on 44 shared recipes](docs/generated/pairwise-torchvision.png)

![AX vs Pillow on 46 shared recipes](docs/generated/pairwise-pillow.png)

![AX vs DALI on 51 shared recipes](docs/generated/pairwise-dali.png)

### Measurement settings

Machine: `g2-standard-16`, `nvidia-l4`. Output: CUDA `float16` BCHW `256x3x224x224`. Dataset: 10,000 selected files. Seeds: 137, 138, 139. Workers: 15; prefetch setting: 2; persistent workers: true; dataset prewarm: true.

Warm-up batches: 1. Throughput measures 32 batches (8,192 images), ending after CUDA synchronization. Construction and worker startup are excluded; prefetch effects remain.

GPU memory is sampled by NVML every 50 ms from before pipeline construction through final synchronization and cleanup in the same pass. The peak includes process and library allocations. Shorter peaks may be missed; CPU and model memory are not measured.

![When a batch arrives late, the GPU sits idle and completes fewer steps in the same time.](docs/generated/batch-wait.png)

<!-- END GENERATED RESULTS -->

## How to read the comparisons

- Resize (R01) and RandomResizedCrop (R03) use bilinear interpolation without
  antialiasing. AX uses OpenCV `INTER_LINEAR`. The two Pillow paths use an
  affine resampling call because Pillow's bilinear resize filters when downsampling.
- Shear (R46) uses a fixed 10° X shear around the image center, bilinear
  interpolation, and zero-valued samples outside the frame on all eight paths.
  The CUDA preflight checked the corrected geometry before timing.
- Pillow and Pillow-AllSIMD augment on CPU, transfer the pinned `uint8` batch
  to CUDA, then normalize on GPU. The AllSIMD worker recorded AVX2 in fast mode.
- AX uses its batch `InputNormalizer` on GPU. Preflight verified the fused
  Triton path on a full `256×3×224×224` batch. The timed results cover the
  complete input pipeline.
- RandomCrop224 takes a crop at decoded resolution. DALI uses an integer-offset
  slice with centered zero padding where needed.
- DALI's R08 Affine mapping omits rotation and shear. Kornia's R08 mapping
  also omits shear. The [manuscript](paper/main.tex) describes the differences
  in geometry, sampling, and filtering.

Three seeds provide repeated executions with the same per-seed file order across
paths. The results measure input preparation on one machine type. Performance
during training also depends on the model and its use of the GPU.

## Update the README results

Use the [report generation commands](paper/README.md#regenerate-the-results)
to rebuild the figures, tables, and [machine-readable results](docs/generated/results.json)
from validated cells. Website builds consume that export, including the source
run ID for each observation. Rebuild the paper from the same inputs.

## Transformation coverage

The [coverage census](catalog/operations.yaml) records the APIs reviewed for
125 selected 2D RGB entries in AlbumentationsX 2.6.2. It records counterparts for 57 entries in
Kornia 0.9.0, 40 in TorchVision 0.29.2, and 46 in Pillow 13.0.0.
The mapping includes similar image effects. Sampling, border handling, and
annotation support can differ.

## Run the RGB matrix

Read [the execution contract](docs/benchmark_execution_contract.md) before changing or launching a run. Configure [GCP access](configs/cloud/gcp-l4.yaml) and access to the [dataset archive](configs/families/rgb.yaml), then commit the intended inputs. With `uv` and an authenticated `gcloud` installed, run from the repository root:

```bash
uv run augbench launch-rgb
```

The command requires a clean worktree, validates existing immutable cells, and resumes missing cells for the same run identity. It reuses an active VM for that run or provisions one L4 VM. Code and frozen inputs determine the run identity; changing them creates a new run. To inspect or regenerate the report without launching a VM, follow the [paper's reproduction instructions](paper/README.md).

## Repository layout

```text
catalog/       operation census and RGB recipes
configs/       workload and GCP configuration
environments/  locked RGB benchmark environment
infra/gcp/     VM bootstrap
src/augbench/  benchmark implementation
paper/         manuscript, bibliography, generated tables, and per-seed data
```

RGB is the only runnable study. 9-channel images, video, and volumes require separate datasets, recipes, implementations, output contracts, and runs.
