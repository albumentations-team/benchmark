# Good Plots For Benchmark Papers

This guide maps public benchmark claims to figures for the augmentation benchmark. Use it when writing website copy,
README summaries, the manuscript, or deciding which plots belong in a short result page versus deeper supporting docs.

## Source Principles

- Start from the claim, not the chart type. Amar, Eagan, and Stasko frame visualization around analytic tasks such as retrieving values, finding extrema, sorting, correlating, clustering, characterizing distribution, finding anomalies, and comparing subsets.
- Prefer encodings people read accurately. Cleveland and McGill's graphical perception work argues for using the highest-accuracy perceptual tasks available: position on a common scale before length, angle, area, or color-only encodings.
- For benchmark papers, make scope and assumptions visible. Weber et al. emphasize defining benchmark purpose and scope before evaluation; DAWNBench and MLPerf show why a proxy metric such as minibatch throughput can fail to support end-to-end claims.
- Do not reduce a benchmark to a leaderboard when the story is a tradeoff. Wiesenfarth et al. show that challenge rankings can be sensitive to design choices and that tables or simple metric plots can hide case-level behavior.

## Claim-To-Plot Matrix

| Paper claim | Main plot | Supporting detail | Avoid |
|---|---|---|---|
| One library is fastest within one clearly defined regime | Sorted dot plot or horizontal bar chart with common x-axis | Table with median, IQR/CI, hardware, mode, library versions | Unqualified global ranking across micro, CPU DataLoader, GPU DataLoader, and DALI |
| Benchmark regime changes the conclusion | Grouped or stacked winner-count bars by regime | Small multiples of per-regime sorted throughput | One merged leaderboard that hides regime labels |
| GPU micro speed does not predict DataLoader throughput | Paired scatter plot or slopegraph from micro to DataLoader for matching transforms | Ratio distribution and outlier labels | Separate unrelated bar charts that force visual matching by memory |
| CPU pipeline remains competitive against GPU pipelines | Box, violin, or jittered strip plot of `GPU / AlbumentationsX CPU` ratios with a reference line at `1.0` | Per-library win counts and median ratio | Only showing winning rows or only showing means |
| Coverage breadth differs across libraries and regimes | Coverage-versus-throughput scatter or small multiples with a common recipe denominator | Unsupported/early-stopped supplement and row-level reason table | Treating missing rows as zero throughput in winner counts or medians |
| Transform support differs by library | Coverage heatmap or stacked bars: full, early-stopped, unsupported, absent/not reported | Row-level reason table | Throughput-only plot without a coverage denominator |
| Which transforms each library misses in production | Transform-by-library support/performance matrix with CPU and GPU columns where both exist | Long CSV/markdown appendix generated from row-level results | Only aggregate coverage counts when the claim depends on specific missing transforms |
| Memory or operational cost matters | Scatter plot of throughput versus peak GPU memory, with library color | Table of peak/reserved memory and batch size | Ranking by speed alone when memory is part of the claim |
| Repeated runs show stability or noise | Dot/interval plot with every run visible and median marked | Coefficient of variation table | Bar chart with no variance or run count |
| Method performance varies across transforms | Jittered dot plot, beeswarm, or small multiples by transform group | Appendix pivot table | Pie chart, 3D chart, or area chart for quantitative comparison |
| Multiple algorithms over many datasets or tasks need statistical comparison | Critical-difference diagram only when the design matches Demsar-style repeated-method comparisons | Friedman/post-hoc test details and assumptions | CD diagram for one dataset, one hardware setting, or non-independent rows |

## Rules For This Project

1. Every figure caption must state the benchmark regime, metric, transform set, dataset size, hardware class, and whether unsupported or early-stopped rows are included.
2. Use throughput plots only inside one measurement scope. Do not compare augmentation-only micro rows directly against DataLoader rows without making the scope difference the explicit claim.
3. Show denominators for support claims: `48/57 GPU DataLoader recipes measured` or `22/57 GPU DataLoader recipes measured` is stronger and safer than saying a library "has broad support" without a denominator.
4. Prefer ratios when the claim is comparative and paired, for example `GPU pipeline / AlbumentationsX CPU pipeline` for the same transform.
5. Keep row-level provenance available. Main figures should summarize, but the appendix or generated CSV must let readers trace each plotted point to `all_results.csv`.
6. Use color for library identity, not for quantitative magnitude when position can encode the metric. Keep library colors stable across all figures.
7. Avoid pie charts, donut charts, 3D effects, color-only heatmaps for exact values, and dual y-axes in the paper.
8. Use log scale only when order-of-magnitude differences are the point; label it explicitly in the axis and caption.
9. If a figure supports a negative claim, show the failed or missing cases. Unsupported and early-stopped rows are evidence, not cleanup.
10. If the plot changes a paper conclusion, promote it to the main text; if it only diagnoses mechanism or resource cost, put it in the appendix.

## Current Figure Roles

The generator writes manuscript-local copies under `_internal/paper/figures/` and README-facing copies under
`docs/benchmark_figures/`.

| Figure | Role |
|---|---|
| `_internal/paper/figures/abstract_claims.png` | Main text overview figure tying the abstract claims to winner counts, CPU medians, GPU ratios, and coverage breadth. |
| `_internal/paper/figures/coverage_vs_throughput.png` | Main text figure for coverage breadth versus measured-row throughput. Missing or unsupported rows reduce coverage, not throughput. |
| `_internal/paper/figures/winner_counts.png` | Main text figure for the claim that winners change by benchmark regime. |
| `_internal/paper/figures/gpu_vs_albumentationsx_cpu_boxplot.png` | Main text figure for paired GPU-vs-CPU DataLoader competitiveness. |
| `_internal/paper/figures/coverage_by_regime.png` | Main text or near-main figure to prevent overclaiming library coverage. |
| `_internal/paper/figures/dataloader_library_medians.png` | Supporting high-level throughput summary; pair it with coverage and regime caveats. |
| `_internal/paper/figures/gpu_memory_vs_throughput.png` | Appendix figure unless the manuscript makes memory an explicit operational-cost claim. |
| `_internal/paper/generated/production_support_matrix.md` | Appendix table for transform-level production DataLoader support and throughput. |

## References

- Cleveland, W. S. and McGill, R. "Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods." JASA, 1984. https://doi.org/10.1080/01621459.1984.10478080
- Cleveland, W. S. and McGill, R. "An Experiment in Graphical Perception." International Journal of Man-Machine Studies, 1986. https://doi.org/10.1016/S0020-7373(86)80019-0
- Heer, J. and Bostock, M. "Crowdsourcing Graphical Perception." CHI, 2010. https://idl.uw.edu/papers/crowdsourcing-graphical-perception
- Amar, R., Eagan, J., and Stasko, J. "Low-Level Components of Analytic Activity in Information Visualization." InfoVis, 2005. https://doi.org/10.1109/INFVIS.2005.1532136
- Heer, J., Bostock, M., and Ogievetsky, V. "A Tour through the Visualization Zoo." Communications of the ACM, 2010. https://idl.uw.edu/papers/visualization-zoo
- Weber, L. M. et al. "Essential Guidelines for Computational Method Benchmarking." Genome Biology, 2019. https://doi.org/10.1186/s13059-019-1738-8
- Coleman, C. A. et al. "DAWNBench: An End-to-End Deep Learning Benchmark and Competition." NeurIPS ML Systems Workshop, 2017. https://dawn.cs.stanford.edu/publications/dawnbench/dawnbench-end-end-deep-learning-benchmark-and-competition
- Mattson, P. et al. "MLPerf Training Benchmark." MLSys, 2020. https://proceedings.mlsys.org/paper_files/paper/2020/hash/411e39b117e885341f25efb8912945f7-Abstract.html
- Wiesenfarth, M. et al. "Methods and Open-Source Toolkit for Analyzing and Visualizing Challenge Results." Scientific Reports, 2021. https://doi.org/10.1038/s41598-021-82017-6
- Demsar, J. "Statistical Comparisons of Classifiers over Multiple Data Sets." JMLR, 2006. https://jmlr.org/papers/v7/demsar06a.html
