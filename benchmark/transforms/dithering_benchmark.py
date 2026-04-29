"""Shared benchmark contract for ``Dithering`` across AlbumentationsX and Pillow.

Keep all libraries and tests aligned on the same parameter dict. Change values only here
(and in any intentional negative tests that assert non-matching params).
"""

from __future__ import annotations

from typing import Any

DITHERING_BENCHMARK_PARAMS: dict[str, Any] = {
    "method": "error_diffusion",
    "n_colors": 2,
    "color_mode": "grayscale",
    "error_diffusion_algorithm": "floyd_steinberg",
}


def dithering_params_match_benchmark_contract(params: dict[str, Any]) -> bool:
    """True iff ``params`` matches :data:`DITHERING_BENCHMARK_PARAMS` on every required key.

    Extra keys in ``params`` are ignored. Missing or unequal values yield False.
    """
    return all(k in params and params[k] == v for k, v in DITHERING_BENCHMARK_PARAMS.items())


def albumentations_dithering_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Kwargs for ``A.Dithering`` from spec params; raises ``KeyError`` if any contract key is missing.

    Uses the same required keys as :data:`DITHERING_BENCHMARK_PARAMS` (direct indexing, like other transforms).
    """
    return {key: params[key] for key in DITHERING_BENCHMARK_PARAMS}
