"""Models for the training workload."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def build_model(name: str, device: str) -> torch.nn.Module:
    """Build an untrained torchvision classifier in channels-last layout on ``device``."""
    import torch
    from torchvision import models

    constructors = {"resnet18": models.resnet18, "resnet50": models.resnet50}
    return constructors[name](weights=None).to(device).to(memory_format=torch.channels_last)
