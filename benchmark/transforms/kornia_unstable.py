"""Kornia transforms excluded from video DataLoader/pipeline recipes.

Keep this list narrow. Rows should stay in the benchmark and be recorded as unsupported at
runtime unless a transform is proven to crash the worker process or poison the CUDA context.
"""

from __future__ import annotations

KORNIA_BENCHMARK_EXCLUDED_NAMES: frozenset[str] = frozenset()
