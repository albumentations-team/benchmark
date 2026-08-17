# Benchmark preprint

This is the paper for this repository's RGB disk-to-GPU benchmark. It is not the
AlbumentationsX system paper in `albumentations-ai-docs`.

The manuscript describes only the current production contract. Final tables and
figures may be inserted only after every cell in one immutable RGB run has passed
validation. They must be generated from the raw cell records, not entered by hand.

Build locally with:

```bash
latexmk -pdf main.tex
```

The committed draft has no result figure dependencies, so it remains buildable
while the production matrix is running.
