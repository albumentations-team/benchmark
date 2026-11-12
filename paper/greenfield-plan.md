# Greenfield plan for the RGB JPEG-to-GPU benchmark paper

September 6, 2026. This plan describes the original seven-path study.
See the [paper README](README.md) for the current report and its source runs.

## 1. What the new paper should deliver

Write a new empirical paper in English for ML engineers choosing an image pipeline for a model.
After reading it, they should understand:

1. What the comparison helps them decide and which conditions are fixed.
2. What each measured path does, including decoding and work on the CPU and GPU.
3. How throughput and GPU memory differ on the recipes they need.
4. How much of the selected transformation catalog each library covers.
5. Which findings apply to their project and which questions need separate measurements.

On the fixed RGB workload, DALI is faster than AX on their measured intersection, and AX is
faster than the selected Pillow, TorchVision, and Kornia paths on most shared recipes. Show the
size of those differences, the exceptions, and the corresponding memory use. The catalog of
supported transformations is a separate factor in choosing a library.

The measurements, code from the measured run, configuration, and catalogs determine the new
paper's structure. Use the existing `main.tex`, wording, and figures as a checklist of claims
to verify again.

The deliverables are a new paper, a compiled PDF, tables and figures generated from the data,
and one results table for independent analysis. This plan does not itself start the rewrite or
any new measurements.

## 2. Data and scope

Use the existing immutable run:

```text
run_id: 3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c
commit: 5fc35f6fdd177c286cbc4f5e39d1520576d6464a
```

Use the local configuration and `paper/generated/metrics.tex` to check these counts.

| Item | Count | What it measures |
| --- | ---: | --- |
| Libraries | 5 | AlbumentationsX, Pillow, TorchVision, Kornia, DALI |
| Implementations | 7 | Separate CPU and GPU paths for TorchVision and Kornia |
| Recipes | 57 | Full sequences for preparing an RGB batch |
| Measured implementation–recipe pairs | 253 | The supported matrix for this run |
| Cells | 759 | 253 pairs × 3 seeds |
| Common intersection | 11 | Recipes measured for all seven implementations |
| Intersections with AX | 51 / 26 / 26 / 22 | Kornia / TorchVision / Pillow / DALI |
| Coverage | 118 / 46 / 38 / 23 | AX catalog entries and matches in Kornia / TorchVision / Pillow |

These are checkpoints for preparing the paper. Before regenerating results, run the existing
validator against the full set of raw cells. Writing this plan does not validate the source
JSON.

Use this run for both throughput and memory. Coverage is based on the AX 2.3.7 catalog. The
execution benchmark used AX 2.4.2. State both versions alongside their respective results and
keep the census pinned to its original release.

The scope excludes new GPU runs, batch size or worker tuning, training measurements, and new
families. A change to the measured implementation or workload requires a separate decision and a
new run. Correcting the description of existing code leaves the original results intact.

## 3. Order of presentation

### 3.1. What choice the experiment helps readers make

Open with the application's need for the next ready batch and explain how libraries distribute
preparation across the decoder, CPU, and GPU. Frame the question in terms of images delivered
per second and GPU memory used by the process.

The experiment stops at the model input. Training time and how GPU augmentation shares the
device with a model require separate measurements.

Explain why the JPEG-to-GPU boundary matters. It includes all the work an application must do to
hand images to the model, so each path keeps its own decoder and placement of operations.

### 3.2. What one test takes in and returns

Define a recipe with two examples before listing the libraries:

```text
RandomCrop224 → Normalize → ToTensor
RandomCrop224 → Equalize → Normalize → ToTensor
```

Show the common input, a JPEG at its original dimensions from the selected file list, and the
common output, an RGB CUDA float16 BCHW batch of 256 × 3 × 224 × 224. Explain each dimension
once. In the second example, throughput covers the entire path including Equalize. There is no
isolated Equalize measurement.

Recipe stage names describe the intended transformation. The path descriptions show where each
library creates the tensor or changes layout and dtype. Each library can produce the required
output without literally calling `ToTensor`.

### 3.3. Why these conditions were chosen and how paths differ

Explain the decisions in the table in Section 4. Then give a compact experiment summary and an
implementation table. Readers should know the batch size, worker count, and where operations run
before seeing the first number.

Explain how closely the recipes match. Name the parameters that are aligned, the public API
differences that remain, and what the output validator checks. Matching shape, dtype, and device
establishes the output contract. Pixel values and random draws can still differ.

### 3.4. Results on one common workload

Lead with the 11 recipes supported by all seven implementations. This lets readers compare all
paths on the same rows without selecting a CPU or GPU winner in advance.

Show per-recipe throughput first, followed by GPU memory for the same recipes. Each paragraph
should name an observation and point to the figure or table that supports it. Introduce
aggregated win counts after readers have seen the measurements.

### 3.5. What the other measured recipes add

The all-library intersection covers 11 of 57 recipes. Use the broader intersection between each
library and AX to show what the common set leaves out. These intersections overlap and differ in
composition.

Define how to select the faster CPU or GPU path for each recipe. This is a choice between the
two measured implementations based on the observed results. It establishes neither automatic
tuning nor a global performance maximum.

Show four compact bar charts of mean ratios for AX versus Kornia, TorchVision, Pillow, and DALI.
Average CPU and GPU paths separately on the same recipe set. Put the full per-recipe results in
the appendix and a machine-readable table. Each comparison covers its own recipe set and
supports conclusions about that set.

### 3.6. Available transformations and how to use the results

After the measurements, include a short standalone coverage section. Count entries in the
selected AX RGB 2D catalog with a match in another library. One competing API can match several
entries. This unit measures matches to the AX catalog, and leaves the competing library's total
API size and capabilities outside the count.

Explain the relationship to the 57 recipes. One class can produce several recipes, and some
classes were outside the execution benchmark. DALI participates only in execution, so it has no
census row.

End with a sequence readers can follow. Check that the transformations they need are supported,
find the measured recipes, compare throughput and memory, and then check the full pipeline with
their model. A new composition of measured transformations needs its own measurement.

### Abstract and appendices

Write the abstract last. Include the question, measurement boundary, scale of the experiment,
main results, and scope. Report both measured metrics and give the sizes of intersections with
their corresponding win counts.

The appendices contain versions, hardware details, the full recipe list and parameters, API
details, results by implementation and seed, and reproduction instructions. Include run history,
resume behavior, and diagnostics only where they affect how results can be checked. Start the
appendices on a new page after References and put an explicit Appendices heading before sections
A–E.

## 4. Explain decisions through the experimental conditions

| Decision | What to explain | Limit on the conclusion |
| --- | --- | --- |
| JPEG → ready CUDA batch | The application needs the full preparation path, including reading, decoding, collation, transfer, and normalization | Throughput describes the full path. The experiment cannot isolate a kernel's contribution |
| Native reader and decoder | The libraries prepare data differently, and each keeps its own path within the shared input and output boundary | The measurements describe the specific integrations. Kernel timings require separate measurements |
| RGB and ImageNet JPEG | The workload uses the first 10,000 sorted files, three channels, and the original dimensions | Results apply to this workload. Other RGB data, medical images, and video need separate evidence |
| Prewarm the files | A full initial read reduces the difference between the first cold pass and subsequent reads | The measurement includes filesystem calls and decoding with a warm cache. Physical disk bandwidth is outside scope |
| One file order per seed | All implementations receive the same permutation without replacement | The shared seed sets file order. Libraries can still draw different augmentation parameters |
| 224 × 224, float16, batch 256 | These are the output contract and fixed operating point. State that Resize produces 224 × 224 output | Establishing an optimum would require a separate search |
| 15 workers and prefetch 2 | These settings are fixed for DataLoader paths. Also report the number of DALI threads and its native queue | Different architectures can have different queue sizes at the same numeric prefetch setting |
| CPU stages of GPU paths | Each implementation needs to turn inputs of different sizes into samples that can be batched | The constraint applies to the measured DataLoader path. Other GPU implementations can handle batching differently |
| GPU normalization | The contract puts final normalization on the GPU for every implementation | This benchmark does not compare CPU and GPU normalization |
| Per-image randomness and calls | Report what each implementation samples and how it executes the transformations | When the reason for per-image execution is unknown, describe the observed calls without assigning the choice to an API limitation |
| One warm-up batch and 32 measured batches | Throughput is 8,192 images divided by the elapsed time. The window starts and ends with synchronization | The window includes asynchronous preparation and queuing. Illustrate the possible overlap without claiming sequential batch processing or confirmed saturation |

Report a value as an experimental condition when the configuration only establishes that it was
fixed. Explaining why the batch size is exactly 256 requires evidence. Treat that value as a
fixed choice unless there is a record of tuning.

Describe a reproducible recipe selection rule. Use complete paths from the frozen catalog that
AX and at least one competitor support. State the limitations of this selection. It covers a
particular catalog and is neither exhaustive nor a random sample of transformations.

## 5. Calculation and interpretation rules

### Throughput and implementation selection

1. For each implementation–recipe pair, take the median throughput across the three seeds.
2. Keep all seven implementations in separate columns for the common intersection.
3. To count wins across all measured TorchVision and Kornia recipes, choose the path with the
   higher median throughput for each recipe. Use the CPU path to break exact ties
   deterministically. Record the chosen path in the data.
4. For each recipe, take GPU memory from the same selected path. An independent memory minimum
   could come from a different implementation.
5. Use the same direction for relative throughput throughout the paper,
   `selected path throughput / AX throughput` for one recipe. A ratio of `1×` means equal
   throughput. A ratio of `2×` means the competing path delivers twice as many images per
   second. Label the direction on each figure.
6. For the chart of 11 shared recipes, use the arithmetic mean of the per-recipe ratios for each
   of the seven implementations. The separate charts show the mean for AX and for each CPU or
   GPU path on their own. Use three bars each for Kornia and TorchVision and two each for Pillow
   and DALI. Use 46/25/26/22 shared recipes, respectively, across all bars within a chart. AX is
   `1×`, and each recipe has equal weight.
7. For each pair, calculate the median of its per-recipe ratios. When reversing the direction,
   recalculate the ratios from the data to avoid inverting a rounded headline value.

Count wins by strict comparison of medians and show exact ties separately. The three seeds
support a descriptive count. They do not justify a statistical significance claim or an
arbitrary tolerance for ties.

### GPU memory and variation

Use the peak process GPU memory reported by NVML. The monitor runs in the same pass as the
throughput measurement, with a window that starts before pipeline construction. It measures GPU
memory used by the whole process and its infrastructure. CPU RAM is outside scope, and the
result cannot isolate the increase due to one transform or the memory reported by the tensor
allocator.

State the polling interval from the raw records. Short peaks between polls can go unobserved, so
the data cannot establish byte-level precision or the maximum trainable batch size.

For each recipe, take the median GPU memory across the three seeds. For pairwise comparisons,
compute `selected path memory − AX memory` for each recipe and then take the median of those
differences. Subtracting the overall medians would answer a different question.

Report two kinds of variation:

- **Between seeds of one recipe:** retain all three values and their min–max range. This
  describes observed repeatability without estimating a confidence interval.
- **Between recipes:** use the median and the 25th and 75th percentiles to show differences
  across workloads. Use the `inclusive` quartile method throughout, as in the generator. These
  percentiles describe the recipes, so they cannot establish measurement error or the
  reliability of a ranking.

Give each recipe equal weight in the summary. This describes the selected set. A project that
applies different transformations or uses them at different frequencies needs its own throughput
estimate.

## 6. Figures and tables

Renumber the figures and tables at build time. Keep existing objects only if they serve the new
paper. Use TikZ or PGFPlots for the diagram and heatmap, with captions, legends, numbers, and
tables in LaTeX text. Check readability at the final page width and in grayscale. Align numeric
table cells to the right and wrap long explanations at word boundaries. Keep text at a readable
size without shrinking the entire object with `resizebox`.

### Figure: input, output, and the two measurement windows

Use one diagram to show the JPEG, library processing, and the model-ready CUDA batch. Below it,
draw separate timelines for throughput after warm-up and memory from construction through final
synchronization. Show prefetch and possible overlap between stages. Block widths are schematic
and carry no measured durations.

Use a separate table for the seven paths. Include the decoder, work before H2D, transfer dtype,
work after H2D, and whether each operation runs per image or per batch. Describe DALI's native
graph in its own terms, including the mixed decoder.

Check the table against the frozen code. In particular, TorchVision GPU converts the batch to
float16 after H2D and calls the remaining transforms one image at a time. Label CPU and GPU
paths by where the main transformations run. CPU paths also normalize on the GPU.

### Table: experiment summary

Include data and access order, cache state, output, batch size, workers and threads, prefetch
policy, warm-up, the measurement window, seeds, and machine, CPU, and GPU. Keep the short,
essential values in the main text. Put full versions, hashes, and environment settings in an
appendix linked directly from the table.

### Figure: throughput on the 11 shared recipes

Use a heatmap with 11 recipe rows and six columns for the competing paths. Print each path's
throughput ratio to AX in its cell. Add a column on the right with AX's absolute throughput in
images/s so readers can see the scale. This shows all seven implementations without an AX column
of identical `1×` ratios.

Map `log2(ratio)` to color, with a neutral color at `1×` and symmetric bounds around zero that
cover every value. Print the ratios themselves as ordinary numbers. Use a palette that works
without relying on red and green and retain the full range without clipping.

Order recipes by structure. Show the basic shape-forming recipes first, followed by recipes with
an additional transformation. Explain common stages once and link abbreviated labels to the full
recipe IDs in the appendix. Place the CPU and GPU columns for each library next to each other.
Keep the order independent of the results for AX.

The caption should explain that each cell shows the ratio of throughput medians across three
seeds for one recipe. A value above `1×` means the competing path is faster. Color conveys the
size of the difference and carries no statistical significance information. Put all absolute
values in the appendix and the data table.

### Figure: mean ratios on the common set

Add a horizontal bar chart for the seven implementations on the same 11 recipes, sorted by mean
ratio. Use the arithmetic mean of per-recipe `throughput / AX throughput` ratios. Calculate the
median across three seeds for each recipe before dividing, and give every recipe equal weight.
Taking the ratio of separately averaged throughputs would give recipes different weights.

Start the bars at zero. Mark AX = `1×` with a dashed line. Label every bar with its ratio, use
full CPU and GPU labels, and indicate that higher is faster. Keep the per-recipe heatmap nearby
so readers can see what the average summarizes and judge whether the recipes fit their training
pipeline.

### Figure: GPU memory on the same 11 recipes

Show seven horizontal bars, one per implementation. Each value is the median of the per-recipe
medians of peak process GPU memory in MiB. Start the axis at zero, label it “Lower is better,”
and put the values at the ends of the bars. Sort by increasing memory use. Retain individual
results and variation across seeds in the appendix and CSV.

### Four figures: AX versus Kornia, TorchVision, Pillow, and DALI

Use a compact mean-ratio bar chart for each comparison. Plot AX, Kornia CPU, and Kornia GPU on
46 common recipes. Plot AX, TorchVision CPU, and TorchVision GPU on 25. Plot AX and Pillow CPU
on 26, and AX and DALI GPU on 22. Each bar is the arithmetic mean of per-recipe ratios of path
throughput to AX throughput. Mark AX = `1×`, use a linear scale starting at zero, and print the
values at the ends of the bars.

Each bar represents one fixed CPU or GPU path. Use the same recipes for all bars within a chart
and state their count in the caption. Recipe sets can differ between charts. Give win counts
across all measured recipes (51/26/26/22) in a separate paragraph. Keep full per-recipe results
in the appendix and CSV, which replace the per-recipe figures for these pairwise comparisons.

### Table: coverage of the selected catalog

Use four rows for AX, Kornia, TorchVision, and Pillow. Report the number of matched entries and
the census version. Name AX as the source catalog in the caption. Four counts fit in a table and
give readers the comparison they need.

For verification, include the mapping from each AX entry to the competing API and the limits of
that match in the appendix or the accompanying catalog. DALI has no role in this count. Explain
the difference between coverage and execution with one example of a class that produces several
recipes.

## 7. Limits and research context

The existing results support claims about the observed throughput and memory of particular paths
on the specified recipe sets. Recheck the main comparisons. Verify DALI's advantage on its
measured intersection, AX's advantage on most shared recipes with the other libraries, and the
exceptions and sizes of the differences.

These data leave the following questions open:

- Epoch time, accuracy, and convergence.
- How efficiently augmentation and a model can share the GPU.
- The causal contribution of the decoder, transfer dtype, lookup tables, or any single kernel.
- Optimal batch size or worker count, a universal ranking of libraries, and results for other
  channel counts, resolutions, video, or volumes.
- Equivalence of pixels, distributions of random parameters, or APIs.
- How the full AX catalog compares with the full capabilities of other libraries.

Use a short related-work section with primary sources to place this experiment among kernel
benchmarks, input pipeline measurements, and full training measurements. Explain the relevant
differences rather than listing five library descriptions. Cite the AlbumentationsX paper for AX
and verify the bibliography during the rewrite.

Briefly disclose the author's relationship with AX. Discuss dependencies and optimization
history only where they describe the actual measured path. Explaining the measured speedup
through those details would require separate measurements.

## 8. Workflow and checks

The author carries out the rewrite in sequence in the current repository. Use one manuscript and
the existing generator and benchmark framework.

| Step | Work | Completion check |
| --- | --- | --- |
| 1. Gather evidence | Obtain the raw cells for this run, check them with the existing validator, and read the frozen code, config, and versioned census | All 759 expected cells have both metrics, versions are known, and the path table matches the measured code |
| 2. Prepare the numbers | Extend the existing `generate_results.py` only as needed for the selected material | The common set has 11 recipes and pairwise sets have 51/26/26/22. Ratios, selected paths, seed ranges, and memory differences are correct |
| 3. Generate the material | Generate the diagram, heatmap, mean throughput and memory charts, four pairwise charts, and three tables described above. Put full results in the appendix and one `recipe-results.csv` | Every number traces to raw cells or a versioned census. The CSV has 253 implementation–recipe rows and both metrics for each of the three seeds |
| 4. Write the paper | Write methods and results first, then conclusions and limits, then the opening and abstract | Each section answers a reader question. Readers know the relevant conditions before seeing results |
| 5. Build and review the PDF | Inspect every page at its final width. Read the whole paper again and aloud | Numbers and labels are legible at normal size, links work, and comparison sets, colors, and intervals are clear |
| 6. Finish the replacement | Remove superseded text and generated plot assets, then update the paper README and build command | There is one paper, one generation path, and a reproducible build. Raw evidence and other contributors' changes are preserved |

Obtain valid data for the new material first, then replace the old text and assets in one pass.
Maintain one version of the generator and manuscript. Preserve immutable evidence during asset
cleanup.

Changes to the Python generator require an `unslop-code` review limited to the diff and the
relevant existing checks. Add a check only for substantive new logic, such as taking memory from
the selected path or setting the direction of the median ratio. The Markdown plan requires no
code tests.

The work is limited to reading, locally aggregating 759 records, and compiling the PDF. It
requires no new paid compute. If the raw evidence is unavailable or fails validation, stop
preparing numerical results and report the specific problem. The figures in the old PDF cannot
substitute for the source data.

## 9. Paper readiness criteria

- After the first page, readers can name the question, input, output, and scope.
- The paper defines recipe, implementation, cache state, batch size, workers, hardware, and both
  measurement windows before presenting the first result.
- The methods distinguish conditions of the experiment, limits of a particular implementation,
  and confirmed API constraints.
- Each CPU and GPU path matches the frozen code, including the H2D dtype and how transforms are
  called. All paths normalize on the GPU.
- Catalog entries, recipes, implementation–recipe pairs, and seeds are distinct units.
- The paper distinguishes the common intersection from pairwise intersections, a median of
  ratios from a ratio of medians, differences across recipes from variation across repeats, and
  process peak from marginal memory.
- Throughput and GPU memory appear together and come from one run.
- Readers can inspect every measured recipe. Unsupported operations are excluded from speed
  comparisons and win counts.
- The abstract, figures, and captions tell the same story even when read independently.
- Claims about DALI, AX, and exceptions match the data. Any effect on training remains a
  question to measure.
- Benchmark numbers come from the data. Explanations of parameter choices and causes rely on
  evidence, and results appear where they answer the reader's question.

## Sources for carrying out the plan

- [Execution contract](../docs/benchmark_execution_contract.md).
- [Measured run](README.md).
- [Generator and frozen run identity](generate_results.py).
- [Local checkpoints and summaries](generated/metrics.tex).
- [RGB configuration](../configs/families/rgb.yaml).
- [Recipe catalog](../catalog/recipes/rgb.yaml).
- [Coverage census](../catalog/operations.yaml).
- [Timing window](../src/augbench/measurement_window.py).
- [GPU memory monitor](../src/augbench/nvml_memory.py).

Read these files at the measured run's commit before making claims about execution. Check
coverage against its own catalog version. The current checkout may contain changes outside the
published measurements.
