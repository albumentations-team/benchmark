# Benchmark preprint

This is the paper for this repository's RGB disk-to-GPU benchmark. It is not the
AlbumentationsX system paper in `albumentations-ai-docs`.

The manuscript reports immutable RGB run
`3f8e2e315710528399b8e82e2359ab85c58c809644595b68a92fb9d83492cc8c`.
All 759 production cells passed schema, identity, and frozen-matrix validation.
Tables and plots are generated from those raw records rather than entered by
hand.

Generate the paper inputs after downloading the run's `cells/` directory:

```bash
uv run python paper/generate_results.py \
  --cells /path/to/3f8e2e-cells \
  --output paper/generated
```

Build locally with:

```bash
latexmk -pdf main.tex
```

The generated `.dat` and `.tex` files are committed so the paper remains
buildable without cloud access. The generator refuses incomplete, foreign, or
mixed-run cell directories.
