from __future__ import annotations

from typing import Any

from torch import nn


def set_same_on_batch(module: nn.Module, *, value: bool) -> None:
    for child in module.modules():
        if hasattr(child, "same_on_batch"):
            child.same_on_batch = value


def force_per_sample_randomness(module: nn.Module) -> nn.Module:
    set_same_on_batch(module, value=False)
    return module


class SplitRecipe(nn.Module):
    def __init__(self, cpu_transform: nn.Module, gpu_transform: nn.Module) -> None:
        super().__init__()
        self.cpu_transform = cpu_transform
        self.gpu_transform = gpu_transform

    def forward(self, sample: Any) -> Any:
        return self.gpu_transform(self.cpu_transform(sample))
