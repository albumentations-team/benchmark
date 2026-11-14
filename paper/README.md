# RGB input-pipeline preprint

The [current manuscript](main.tex) compares eight JPEG-to-CUDA paths, including
AlbumentationsX 2.6.2 and Pillow-AllSIMD in fast mode. The
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
| Full RGB | `75bc062fe770cfec3d857f401e16366deb01f6582c8645e952930e3cd3d788fb` |
| AX 2.6.2 | `a3c904c6349eeb0a7d4b6f6a82f741c43d132ea7adb410b0281cea13a9e444c5` |
| Flip and shear source 1 | `040576ceab887e161ce7d029c247ba4d1cf387723a48bf9d5af87f0f19bbe239` |
| Flip and shear source 2 | `e568c8b543fc699a27793130f9a96a8d34eee7aa94a8a59c24bc2f30d7c1a9f4` |
| RGB expansion source 1 | `603c8be58828089a1f53863addf73063bd39f0237a2ba1c7cb6a2fc28649a464` |
| RGB expansion source 2 | `6a307927437af3821afdab05cb7830a7ec33f0e38143bf8fc25163fdd5c85543` |
| RGB expansion source 3 | `8035391a6cb73aa9f28dbacb87b059674f40e345b87c1630cf7f235edae17867` |

The GCS root is `gs://imagenet_validation/augmentation-benchmark`.
Manifests and cells are under `runs/<run_id>/`. The source archive is
`code/<code_archive_sha256>/source.tar.gz`.

Place each run's manifest, cells, and extracted recipe catalog in separate
directories. Use the expanded catalog in the repository checkout as the
report catalog. Run from the repository root:

```bash
run_args=(--run full/run.json --cells full/cells --recipes full/rgb.yaml
          --ax-run ax/run.json ax/cells ax/rgb.yaml
          --correction-run correction1/run.json correction1/cells correction1/rgb.yaml
          --correction-run correction2/run.json correction2/cells correction2/rgb.yaml
          --extension-run expansion1/run.json expansion1/cells expansion1/rgb.yaml
          --extension-run expansion2/run.json expansion2/cells expansion2/rgb.yaml
          --extension-run expansion3/run.json expansion3/cells expansion3/rgb.yaml)
uv run python -m paper.generate_results "${run_args[@]}"
uv run python -m paper.generate_readme "${run_args[@]}"
```

The second command also needs `pdflatex` with TikZ and `preview`, and
Poppler's `pdftocairo`. To select a self-contained full run for the README,
use its manifest, cells, and catalog.

The generators validate each cell against its source manifest and require
complete coverage of the report matrix. Each observation retains its source
run and cell IDs. Throughput ratios use shared recipe IDs and seed medians.
The CSV includes GPU memory baselines and peaks for each seed. The paper
generator builds the coverage table and appendix from the versioned
[catalog census](../catalog/operations.yaml). The census versions describe
the reviewed APIs. Execution versions come from the recorded cells.
