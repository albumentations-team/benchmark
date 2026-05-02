---
name: documentation-generator
description: Updates benchmark documentation with latest results including README tables, speedup plots, and library metadata. Use when updating documentation, generating comparison tables, or when the user mentions update_docs.sh or documentation generation.
---

# Documentation Generator

Automate updating benchmark documentation with latest results.

## Quick Update

```bash
# Update all documentation
./tools/update_docs.sh

# Update with custom paths
./tools/update_docs.sh \
  --image-results output/ \
  --video-results output_videos/ \
  --docs-dir docs/
```

## What Gets Updated

### Architecture / Policy Docs
- `docs/benchmark_architecture.md` - Control-plane and runner architecture.
- `docs/benchmark_scope.md` - Paper benchmark scope, transform selection, pipeline recipes, and architecture source of truth.
- `.cursor/skills/benchmark-runner/SKILL.md` - Agent-facing benchmark execution policy.
- `.cursor/skills/paper-benchmark-execution/SKILL.md` - Agent-facing paper run policy.

### Image Benchmarks
- `docs/images/README.md` - Detailed results table
- `docs/images/images_speedup_analysis.webp` - Speedup visualization
- `docs/images/images_speedups.csv` - Raw speedup data
- `README.md` - Main speedup summary

### Video Benchmarks
- `docs/videos/README.md` - Detailed results table
- `docs/videos/videos_speedup_analysis.webp` - Speedup visualization
- `docs/videos/videos_speedups.csv` - Raw speedup data
- `README.md` - Main speedup summary

## Manual Documentation Steps

### 1. Generate Comparison Tables

**Image benchmarks**:
```bash
python tools/compare_results.py \
  --results-dir output/ \
  --update-readme docs/images/README.md
```

**Video benchmarks**:
```bash
python tools/compare_video_results.py \
  --results-dir output_videos/ \
  --update-readme docs/videos/README.md
```

### 2. Generate Speedup Plots

**Image benchmarks**:
```bash
python tools/generate_speedup_plots.py \
  --results-dir output/ \
  --output-dir docs/images \
  --type images \
  --reference-library albumentationsx
```

**Video benchmarks**:
```bash
python tools/generate_speedup_plots.py \
  --results-dir output_videos/ \
  --output-dir docs/videos \
  --type videos \
  --reference-library albumentationsx
```

### 3. Update Main README

The script automatically updates speedup summaries between markers:
- `<!-- IMAGE_SPEEDUP_SUMMARY_START -->` ... `<!-- IMAGE_SPEEDUP_SUMMARY_END -->`
- `<!-- VIDEO_SPEEDUP_SUMMARY_START -->` ... `<!-- VIDEO_SPEEDUP_SUMMARY_END -->`

Manual update if needed:
```python
import pandas as pd

df = pd.read_csv('docs/images/images_speedups.csv', index_col=0)
median = df['albumentationsx'].median()
max_val = df['albumentationsx'].max()
max_transform = df['albumentationsx'].idxmax()

summary = f"AlbumentationsX is generally the fastest library for image augmentation, "
summary += f"with a median speedup of {median:.1f}× compared to other libraries. "
summary += f"For some transforms, the speedup can be as high as {max_val:.1f}× ({max_transform})."
```

## Library Metadata

Create metadata files for new libraries:

**Image**: `docs/images/{library}_metadata.yaml`
**Video**: `docs/videos/{library}_metadata.yaml`

```yaml
library_name: LibraryName
version: "1.2.3"
description: Brief description of the library
documentation: https://library.readthedocs.io
repository: https://github.com/org/library
```

## Documentation Structure

```
docs/
├── images/
│   ├── README.md                      # Detailed benchmark results
│   ├── images_speedup_analysis.webp   # Main visualization
│   ├── images_speedups.csv            # Speedup data
│   ├── albumentationsx_metadata.yaml  # Library info
│   └── ...
└── videos/
    ├── README.md
    ├── videos_speedup_analysis.webp
    ├── videos_speedups.csv
    └── ...metadata.yaml files
```

## Comparison Tools

### compare_results.py (images)
```bash
python tools/compare_results.py --results-dir output/
```

Output format:
```markdown
| Transform | albumentationsx | torchvision | kornia |
|-----------|-----------------|--------|-------------|--------|-------|
| HorizontalFlip | 1234 ± 45 | 567 ± 23 | ... | ... | ... |
```

### compare_video_results.py (videos)
```bash
python tools/compare_video_results.py --results-dir output_videos/
```

Includes CPU vs GPU comparisons.

### generate_speedup_plots.py
```bash
python tools/generate_speedup_plots.py \
  --results-dir output/ \
  --output-dir docs/images \
  --type images \
  --reference-library albumentationsx
```

Generates:
- Speedup bar chart
- CSV with speedup factors
- Statistical summary

## Validation

After updating documentation:

1. **Check markdown syntax**:
```bash
# Tables should render correctly
# Links should be valid
```

2. **Verify images**:
```bash
ls -lh docs/images/*.webp
ls -lh docs/videos/*.webp
```

3. **Check CSV data**:
```python
import pandas as pd
df = pd.read_csv('docs/images/images_speedups.csv', index_col=0)
print(df.head())
print(f"Shape: {df.shape}")
```

4. **Validate README markers**:
```bash
grep -n "IMAGE_SPEEDUP_SUMMARY" README.md
grep -n "VIDEO_SPEEDUP_SUMMARY" README.md
```

## Workflow

Complete documentation update workflow:

```bash
# 1. Run benchmarks (if needed)
python -m benchmark.cli run \
  --scenario image-rgb \
  --mode micro \
  --data-dir /path/to/imagenet/val \
  --output output/rgb_micro \
  --num-items 2000

# 2. Update all documentation
./tools/update_docs.sh

# 3. Review changes
git diff README.md
git diff docs/

# 4. Commit if satisfied
git add README.md docs/
git commit -m "docs: update benchmark results"
```

## Benchmark Policy Notes

Keep README guidance aligned with these policies:

- Benchmark architecture docs should say that `benchmark/matrix.py` owns scenario/library/mode support, `benchmark/policy.py`
  owns media defaults and slow-skip thresholds, `benchmark/jobs.py` owns command construction, and
  `benchmark/orchestrator.py` owns backend dispatch.
- If the benchmark matrix changes, update `docs/benchmark_architecture.md`, `docs/benchmark_scope.md`, and the relevant
  skill docs in the same change.
- Cloud benchmark docs should show `--gcp-gcs-data-uri` pointing at one dataset archive/object, not a directory of individual images.
- Micro benchmark docs should state that media is preloaded once per library and reused across transform measurements.
- Pyperf docs should mention per-transform subprocess isolation, media-cache reuse, lazy transform construction, and slow-transform preflight/early-stop behavior.
- Benchmark policy docs should mention lazy output materialization: micro timing should force returned outputs to contiguous memory, including contiguous NumPy conversion for Pillow/PIL `Image.Image` outputs. Checksums belong only in diagnostics.
- Benchmark policy docs should state that library tables include only direct transform support. Missing transforms should remain unsupported instead of being recreated with benchmark-side helper code.
- Environment docs should mention joined environments and cached dependency installs, including the detached GCP venv cache.
- Local rerun examples should include `--no-refresh-requirements` when dependency versions are intentionally fixed.

## Troubleshooting

**Missing speedup summary in README**:
- Check CSV file exists: `docs/images/images_speedups.csv`
- Verify markers in README.md
- Run update_docs.sh again

**Plot generation fails**:
- Ensure matplotlib, seaborn installed: `pip install -r requirements-dev.txt`
- Check result files are valid JSON
- Verify all libraries have results

**Table formatting issues**:
- Check all result files have same transform names
- Verify no special characters in transform names
- Ensure consistent JSON structure
