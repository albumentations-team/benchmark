from __future__ import annotations

import torch
from torch import nn

_BATCH_DIMENSIONS = 4


class GpuBatchNormalize(nn.Module):
    """Convert a collated RGB batch to normalized float16 BCHW on CUDA.

    CPU pipelines retain uint8 until the pinned H2D copy. Kornia CPU augments
    in float32, then casts the completed host batch to float16 before H2D.
    Both representations use this single GPU normalization step.
    """

    def __init__(
        self,
        *,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> None:
        super().__init__()
        if not mean or len(mean) != len(std):
            raise ValueError("normalization requires matching non-empty mean and std values")
        self._channels = len(mean)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float16).view(1, self._channels, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float16).view(1, self._channels, 1, 1))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.device.type != "cuda":
            raise RuntimeError("GPU normalization requires a CUDA batch")
        if batch.ndim != _BATCH_DIMENSIONS:
            raise ValueError(f"deferred normalization requires a rank-four batch, got {tuple(batch.shape)}")
        if batch.shape[1] != self._channels:
            raise ValueError(f"BCHW batch must have {self._channels} channels, got {tuple(batch.shape)}")
        if batch.dtype == torch.uint8:
            batch = batch.to(dtype=torch.float16).div_(255.0)
        elif batch.is_floating_point():
            batch = batch.to(dtype=torch.float16)
        else:
            raise TypeError(f"GPU normalization requires uint8 or floating input, got {batch.dtype}")
        return batch.sub_(self.mean).div_(self.std)
