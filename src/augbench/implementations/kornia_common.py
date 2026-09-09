from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def force_per_sample_randomness(module: nn.Module) -> nn.Module:
    for child in module.modules():
        if hasattr(child, "same_on_batch"):
            child.same_on_batch = False
    return module
