# RGB input-pipeline preprint

The [current manuscript](main.tex) compares eight JPEG-to-CUDA paths, including
AlbumentationsX 2.6.3 and Pillow-AllSIMD in fast mode. The
[September preprint](https://arxiv.org/abs/2609.06635) reports the earlier
seven-path experiment. The [results CSV](generated/recipe-results.csv) retains
all three seed observations for each reported implementation and recipe.

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
| Full RGB | `1e7c6db29a2cc0da95eee8f9fce0806f5eb4c5e0c541e1defc784f2f56864d68` |
| AX-only | `175d077e5ee85e9785067111573615b3fa7c6bf80641dabf4f5dd72f3a5dae81` |

The report uses AX observations from the AX-only run because automatic OS
package updates overlapped the AX measurements in the full run. Both runs
used the same recipes, dataset, machine configuration, and package locks.

The GCS root is `gs://imagenet_validation/augmentation-benchmark`.
Manifests and cells are under `runs/<run_id>/`. The source archive is
`code/<code_archive_sha256>/source.tar.gz`.

Place each run's manifest, cells, and extracted recipe catalog in separate
directories. Run from the repository root:

```bash
run_args=(--run full/run.json --cells full/cells --recipes full/rgb.yaml
          --ax-run ax/run.json ax/cells ax/rgb.yaml)
uv run python -m paper.generate_results "${run_args[@]}"
uv run python -m paper.generate_readme "${run_args[@]}"
```

The second command also needs `pdflatex` with TikZ and `preview`, and
Poppler's `pdftocairo`.

The [interim November report](https://github.com/albumentations-team/benchmark/blob/863c9a7eb4bd13eedf3f3abbe2b5ff735e76f31c/paper/README.md#regenerate-the-results)
has its own commands and source manifests at that revision.

The generators validate each cell against its source manifest and require
complete coverage of the report matrix. Each observation retains its source
run and cell IDs. Throughput ratios use shared recipe IDs and seed medians.
The CSV includes GPU memory baselines and peaks for each seed. The paper
generator builds the coverage table and appendix from the versioned
[catalog census](../catalog/operations.yaml). The census versions describe
the reviewed APIs. Execution versions come from the recorded cells.
