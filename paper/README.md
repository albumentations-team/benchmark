# Preprint and report generators

[*From JPEG to Training Step*](main.tex) is version 3 of [arXiv:2609.06635](https://arxiv.org/abs/2609.06635). It reports the input-pipeline benchmark for eight paths and the training-step benchmark for ResNet-18 and ResNet-50. Every number in it comes from the generated files in [`generated/`](generated/), which three scripts build from the stored cells.

## Build the PDF

With a LaTeX distribution, `latexmk`, and PGFPlots installed:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `paper/build/main.pdf`. The generated inputs are checked in, so the PDF builds without access to the cells.

## Regenerate the results

The report combines three immutable runs. All three are under `gs://imagenet_validation/augmentation-benchmark/runs/<run_id>/`, each with a `run.json` manifest and a `cells/` directory. A run's source archive is `code/<code_archive_sha256>/source.tar.gz` under the same root; the recipe catalog and package lock below come from it (or from the listed Git revision).

| Run | Run ID | Source revision |
| --- | --- | --- |
| Input pipeline, all eight paths | `d84d75894f2a0b5ad879c5df1c06ca1ec82690486652c62ed50295a241e19a1a` | `fd09812` |
| Training step, ResNet-18 | `8b5152c09b0eb23f4a6b0fb87d01650da8575cef8e37818049ac54974b866822` | `fd09812` |
| Training step, ResNet-50 | `eb35667fb31dfd0f2a47b5b3c50f61b646291a289d99a7c5f2e32b2aece05c61` | `fd09812` |

Put each run in its own directory: `run.json`, `cells/`, and the run's recipe catalog (`catalog/recipes/rgb.yaml` or `imagenet_train.yaml` at the source revision). Then, from the repository root:

```bash
rgb_args=(--run rgb/run.json --cells rgb/cells --recipes rgb/rgb.yaml)
uv run python -m paper.generate_results "${rgb_args[@]}"
uv run python -m paper.generate_training \
  --run resnet18/run.json resnet18/cells resnet18/imagenet_train.yaml \
  --run resnet50/run.json resnet50/cells resnet50/imagenet_train.yaml
uv run python -m paper.generate_readme "${rgb_args[@]}"
```

Run them in this order: `generate_training` writes the training figure and summary that `generate_readme` reads.

| Script | Writes |
| --- | --- |
| `generate_results` | `paper/generated/`: metrics, recipe and result tables, coverage, versions, figure sources, and the per-seed CSV with intervals |
| `generate_training` | `paper/generated/` training figure, macros, and table; `docs/generated/training-results.{csv,json}` and `training-summary.md` |
| `generate_readme` | The generated section of the repository README, the PNG figures, and `docs/generated/results.json` |

Throughput ratios and verdicts use the exact intervals in [`intervals.py`](intervals.py): an order-statistic interval for one cell's median, and a Hodges–Lehmann interval from the exact Mann–Whitney distribution for a ratio between two paths.
