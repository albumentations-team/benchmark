# RGB input-pipeline manuscript

[*From JPEG to CUDA Batch*](main.tex) compares eight input paths across 72 recipes on NVIDIA L4. The appendices list the recipe parameters, timing and memory results, and API coverage. The September preprint covers the earlier seven-path experiment. [September preprint](https://arxiv.org/abs/2609.06635)

- [Primary matrix CSV](generated/recipe-results.csv) contains three seeds per implementation and recipe.
- [TorchVision GPU repeat CSV](generated/torchvision-gpu-repeats.csv) contains individual observations for all six TorchVision GPU seeds.
- [Machine-readable results](../docs/generated/results.json) contain both run manifests and the figure checksums.

## Build the PDF

Run from the repository root with a LaTeX distribution and latexmk installed:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The PDF is at `paper/build/main.pdf`. The generated inputs are checked in.

## Regenerate the results

| Run | Measurements | Seeds |
| --- | ---: | --- |
| [Primary RGB matrix](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/8d20f12ba0833a29971aec166fa02a98fadd4a9faba013fe12ae7cd81caa920a) | 1,263 | 137–139 |
| [TorchVision GPU repeat](https://console.cloud.google.com/storage/browser/imagenet_validation/augmentation-benchmark/runs/540847a0b6fbcbdb8b4c58e8ea1be295d16e8e62ef3bda43477927088d26bded) | 126 | 140–142 |

With GCS access, download both manifests and cell directories. The runs share the source archive. Extract its `catalog/recipes/rgb.yaml` to use the exact measured recipe catalog. Run from the repository root:

```bash
gcs=gs://imagenet_validation/augmentation-benchmark
primary=8d20f12ba0833a29971aec166fa02a98fadd4a9faba013fe12ae7cd81caa920a
repeat=540847a0b6fbcbdb8b4c58e8ea1be295d16e8e62ef3bda43477927088d26bded
archive=24fafd751ec7e61c6ab4bd532934d7719fc16a233db9f13e154779dfbf26d733

mkdir -p _internal/report-inputs/{primary,repeat,source}
gcloud storage cp "$gcs/runs/$primary/run.json" _internal/report-inputs/primary/
gcloud storage rsync --recursive "$gcs/runs/$primary/cells" _internal/report-inputs/primary/cells
gcloud storage cp "$gcs/runs/$repeat/run.json" _internal/report-inputs/repeat/
gcloud storage rsync --recursive "$gcs/runs/$repeat/cells" _internal/report-inputs/repeat/cells
gcloud storage cp "$gcs/code/$archive/source.tar.gz" _internal/report-inputs/source.tar.gz
tar -xzf _internal/report-inputs/source.tar.gz -C _internal/report-inputs/source catalog/recipes/rgb.yaml

inputs=_internal/report-inputs
run_args=(--run "$inputs/primary/run.json" --cells "$inputs/primary/cells"
          --recipes "$inputs/source/catalog/recipes/rgb.yaml"
          --torchvision-repeat "$inputs/repeat/run.json" "$inputs/repeat/cells")
uv run python -m paper.generate_results "${run_args[@]}"
uv run python -m paper.generate_readme "${run_args[@]}"
```

The README generator uses pdflatex with TikZ and preview, plus Poppler's pdftocairo. The paper embeds vector drawings. The README uses PNGs from `docs/generated/`.

The generators validate both complete cell sets against their manifests and recipe checksum. The repeat must use the primary run's hardware and frozen inputs, with a separate set of seeds. The cross-library plots, tables, and ratios use seeds 137–139. Every observation retains its original run and cell IDs. The coverage appendix uses the versioned census, whose reviewed package versions are recorded separately from runtime versions. [Coverage census](../catalog/operations.yaml)
