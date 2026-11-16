# RGB input-pipeline preprint

[*From JPEG to CUDA Batch*](main.tex) compares batch delivery and peak
process GPU memory across eight input paths. Separate comparisons with
DALI, Pillow, Kornia, and TorchVision use the recipes shared by all paths in
each comparison. The appendices list the recipe sets, full results,
coverage, and reproduction details. The
[results CSV](generated/recipe-results.csv) retains all three seed
observations per recipe and implementation. The
[September preprint](https://arxiv.org/abs/2609.06635) covers seven paths.

## Build the PDF

From the repository root, with a LaTeX distribution and `latexmk` installed:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `paper/build/main.pdf`. The generated inputs are checked in.

## Regenerate the results

Download the run manifests, available `cells/` directories, and source
archives. Use the `catalog/recipes/rgb.yaml` from each run's source archive.
The [results export](../docs/generated/results.json) contains the manifests
and code-archive SHA-256 values. GCS access depends on bucket permissions.

| Artifact | Run ID |
| --- | --- |
| Primary RGB source, 492 cells | `03b5fd6c6fab74fd1029fc88012e03cb4b51d0a2e3c4b8be59ead6928efae6ee` |
| GPU correction, 534 cells | `9c2a6d0b54469dbd1f14833b54e7f4d0758a4242a25673ecdd2d13ff32fd64f3` |
| DALI and Pillow-AllSIMD correction, 297 cells | `32825567918e566a56b11057388185be4085e381f4194495529fe29e0a8897f8` |

The three manifests contain complementary implementation cells. Together
they cover the complete 1,323-cell matrix. The generator validates each cell
against its own immutable manifest, then checks that the combined set has no
missing or duplicate implementation, recipe, and seed combinations. All three
runs use the same recipes, dataset, machine configuration, and package locks.

The GCS root is `gs://imagenet_validation/augmentation-benchmark`.
Manifests and cells are under `runs/<run_id>/`. The source archive is
`code/<code_archive_sha256>/source.tar.gz`.

Place each run's manifest, cells, and extracted recipe catalog in separate
directories. Run from the repository root:

```bash
run_args=(--run primary/run.json --cells primary/cells --recipes primary/rgb.yaml
          --supplemental-run gpu/run.json gpu/cells gpu/rgb.yaml
          --supplemental-run dali-pillow/run.json dali-pillow/cells dali-pillow/rgb.yaml)
uv run python -m paper.generate_results "${run_args[@]}"
uv run python -m paper.generate_readme "${run_args[@]}"
```

The second command also needs `pdflatex` with TikZ and `preview`, and
Poppler's `pdftocairo`. Both commands use the same TikZ drawings and plot
data. LaTeX embeds the drawings as vectors in the paper. The README and
website use PNGs from `docs/generated/`; the export lists each image's
caption, dimensions, and checksum. The website converts those PNGs to WebP.

The [interim November report](https://github.com/albumentations-team/benchmark/blob/863c9a7eb4bd13eedf3f3abbe2b5ff735e76f31c/paper/README.md#regenerate-the-results)
has its own commands and source manifests at that revision.

The generators validate each cell against its source manifest and require
complete coverage of the report matrix. Each observation retains its source
run and cell IDs. Throughput ratios use shared recipe IDs and seed medians.
The CSV includes GPU memory baselines and peaks for each seed. The paper
generator builds the coverage table and appendix from the versioned
[catalog census](../catalog/operations.yaml). The census versions describe
the reviewed APIs. Execution versions come from the recorded cells.
