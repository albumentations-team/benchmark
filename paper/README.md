# Preprint and report generators

[*From JPEG to Training Step*](main.tex) is version 3 of [arXiv:2609.06635](https://arxiv.org/abs/2609.06635). It reports the input-pipeline benchmark for eight paths and the training-step benchmark for ResNet-18 and ResNet-50. Every number in it comes from the generated files in [`generated/`](generated/), which three scripts build from the stored cells.

## Build the PDF

With a LaTeX distribution, `latexmk`, and PGFPlots installed:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `paper/build/main.pdf`. The generated inputs are checked in, so the PDF builds without access to the cells.

## Regenerate the results

The report combines four immutable runs. All four are under `gs://imagenet_validation/augmentation-benchmark/runs/<run_id>/`, each with a `run.json` manifest and a `cells/` directory. A run's source archive is `code/<code_archive_sha256>/source.tar.gz` under the same root; the recipe catalog and package lock below come from it (or from the listed Git revision).

| Run | Run ID | Source revision |
| --- | --- | --- |
| Input pipeline, all eight paths, AX 2.6.8 | `d89b7ecffa4c75b9ab078abf889f3fba91d06772c3eb908c1e7b67c63d41134f` | `c7ed1d7` |
| Input pipeline, AX only, AX 2.6.9 | `c6ee8be602a64fcb3379c4b573c050adcabf9fc41202df563a9b95e2c0531bdd` | `f76aa24` |
| Training step, ResNet-18 | `6fee5174725e31313be5f374ccb2d01a930daeb233281c891b0a3f86d636d7a8` | `9472eb7` |
| Training step, ResNet-50 | `cb7c84e476eda901a11b294c4790b8997d1332d4d8516c3afec66a32523f7cf6` | `9472eb7` |

The AX-only run replaces the AX cells of the full run. Each implementation ran in its own process, so the other cells stay valid; the generator checks that the two package locks differ only in the AlbumentationsX entry and that every replacement cell records that version.

Put each run in its own directory: `run.json`, `cells/`, the run's `catalog/recipes/*.yaml`, and, for the two input-pipeline runs, `environments/rgb/lock.txt`. Then, from the repository root:

```bash
rgb_args=(--run full/run.json --cells full/cells --recipes full/rgb.yaml
          --ax-run ax/run.json ax/cells ax/rgb.yaml --ax-locks full/lock.txt ax/lock.txt)
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
