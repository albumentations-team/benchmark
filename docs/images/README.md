# Image Benchmark Results

## Aggregated Summary

![Image Speedup Analysis](images_speedup_analysis.webp)

*Median speedup vs reference and win rate per library. Single CPU thread.*

See [images_summary.md](images_summary.md) for the full table (paper-ready).

## Regenerate

```bash
python -m tools.generate_speedup_plots --results-dir output/ --output-dir docs/images --type images
# With LaTeX table:
python -m tools.generate_speedup_plots -r output/ -o docs/images --type images --latex
```

Or update all docs: `./tools/update_docs.sh`
