# RGB input-pipeline preprint

This paper compares seven implementations from five libraries on a fixed RGB
JPEG-to-CUDA workload. It reports throughput and peak process GPU memory from
one execution per seed. A separate census describes correspondences within a
selected AlbumentationsX transformation catalog.

Read [the manuscript](main.tex), inspect the
[complete results table](generated/recipe-results.csv), or build the PDF. The
[greenfield plan](greenfield-plan.md) records the design of this rewrite.
The main text is followed by References, then a new page headed Appendices.

## Build the PDF

From the repository root, with a LaTeX distribution and `latexmk` installed:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `paper/build/main.pdf`. Generated `.tex` inputs are included in
the repository, so building the paper requires neither cloud access nor a GPU.
The manuscript uses standard TeX packages including TikZ, booktabs, longtable,
xcolor, microtype, xurl, seqsplit, float, and hyperref.

## Regenerate the reported data

The execution results belong to immutable run
`3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c`, measured
with source commit `5fc35f6fdd177c286cbc4f5e39d1520576d6464a` and code-archive
SHA-256 `61238619ace0dc471bc5df7a4f165e15052ffb67046ceae07c26fd7f370eac6d`.
There are 759 cells: 253 implementation–recipe pairs with three seeds each.

Obtain the existing cell JSON files from the run's `cells/` prefix. This is an
artifact location; bucket permissions determine access:

```text
gs://imagenet_validation/augmentation-benchmark/runs/3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c/cells/
```

Then run from the repository root:

```bash
uv sync --group dev
uv run python paper/generate_results.py --cells /path/to/cells
```

The default output directory is `paper/generated`; `--output /path/to/output`
allows an independent regeneration. The generator checks record schemas,
filename/cell identity, run identity, the complete frozen matrix, output shape,
and measured item count. Changes to inputs that determine this run, or missing
or foreign cells, prevent generation. Throughput is calculated from each cell's
completed-item count and duration. No augmentation is executed.

The generated files contain:

- `common-heatmap.tex` and `common-memory-bars.tex`: per-recipe throughput
  ratios and median peak GPU memory on the same 11 recipes across all seven
  implementations, with CPU and GPU paths kept separate.
- `common-mean-bars.tex`: arithmetic means of the 11 per-recipe throughput
  ratios, sorted by value, with AX at 1× and all seven paths shown separately.
- `pairwise-kornia.tex`, `pairwise-torchvision.tex`, `pairwise-pillow.tex`, and
  `pairwise-dali.tex`: compact mean-ratio bar charts. Kornia and TorchVision
  each show AX, CPU, and GPU on the recipes supported by all three paths
  (46 and 25 respectively). Pillow and DALI each show two paths, on 26 and 22
  recipes. Each chart gives every path the same recipe set.
- `recipe-results.csv`: all 253 measured pairs, with full recipe definitions,
  cell IDs, versions, durations, completed-image counts, throughputs, GPU peaks,
  and memory polling intervals for seeds 137, 138, and 139. The boolean columns
  identify common-set membership and the competing path selected for a pairwise
  comparison. AX is the reference, so its `selected_pairwise_path` is false.
- `recipes.tex`, `recipe-results.tex`, and `versions.tex`: the full recipe
  catalog, per-pair median/minimum/maximum across seeds, and recorded versions.
- `coverage.tex`, `coverage-mapping.tex`, and `coverage-metrics.tex`: the separate
  coverage census, its API correspondences, counts, and source SHA-256.
- `metrics.tex`: aggregate values used in the prose.

## Calculation and interpretation

Each implementation–recipe result is the median of its three seed observations.
Throughput bar charts show arithmetic means of per-recipe ratios: the path's
throughput divided by AX throughput for that recipe. CPU and GPU are separate
bars, and every bar in a chart uses the exact same supported recipe set. The
common chart uses 11 recipes; the pairwise charts use 46/25/26/22 for
Kornia/TorchVision/Pillow/DALI. These different sets cannot rank the libraries.

The memory chart shows the median of per-recipe peak-memory medians, in MiB.
The appendix and CSV retain the individual seed observations.

Recipe-level win counts consider all measured recipes, including those supported
by only one CPU/GPU path: 51/26/26/22. AX wins only when it exceeds every available
competing path. The CSV's selected path is the faster competitor for that recipe;
an exact tie selects CPU. Paired memory differences use this same selected path.

The frozen adapters do not establish numerical equivalence. In particular,
DALI's Crop includes a short-side resize, and its Affine mapping omits the
catalog's rotation and shear. The paper describes these mappings and the
CPU/GPU execution choices explicitly. Results apply to those implementations
at the fixed settings; model training and optimal library configurations were
not measured.

Coverage uses the maintained mapping in
[`catalog/operations.yaml`](../catalog/operations.yaml). It counts selected AX
RGB 2D entries in `geometry`, `pixel`, and `dropout_or_multi_image`: 118 AX
entries, with 46 Kornia, 38 TorchVision, and 23 Pillow correspondences. A
competing API can match several AX entries. The census starts from AX 2.3.7;
the execution run used AX 2.4.2. DALI participates in execution only. Coverage
entries and the 57 complete execution recipes are distinct units.
