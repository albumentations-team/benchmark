"""One forward, backward, and optimizer step on a ready CUDA batch."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from augbench.training.model import build_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    import torch

    from augbench.run_config import TrainingConfig


class TrainSession:
    """Own the model for a whole worker process and reuse its compiled graph across cells."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        batch_size: int,
        device: str = "cuda",
        model_factory: Callable[[str, str], torch.nn.Module] = build_model,
    ) -> None:
        import torch

        self._config = config
        self._batch_size = batch_size
        self._device = device
        self._model = model_factory(config.model, device)
        self._model.train()
        self._initial = {name: value.detach().clone() for name, value in self._model.state_dict().items()}
        self._runner: Callable[[torch.Tensor], torch.Tensor] = (
            torch.compile(self._model, dynamic=False) if config.compile else self._model
        )
        self._optimizer: torch.optim.Optimizer | None = None
        self._labels: torch.Tensor | None = None
        self._loss: torch.Tensor | None = None

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def model_name(self) -> Literal["resnet18", "resnet50"]:
        return self._config.model

    @property
    def labels(self) -> torch.Tensor:
        if self._labels is None:
            raise RuntimeError("reset() must run before the first step")
        return self._labels

    def reset(self, seed: int) -> None:
        """Restore the initial weights in place, then draw the seed's labels and a fresh optimizer."""
        import torch

        self._model.load_state_dict(self._initial)
        generator = torch.Generator(device=self._device).manual_seed(seed)
        self._labels = torch.randint(
            0, self._config.label_classes, (self._batch_size,), generator=generator, device=self._device
        )
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._config.learning_rate,
            momentum=self._config.momentum,
            weight_decay=self._config.weight_decay,
            fused=self._device == "cuda",
        )
        self._loss = None

    def step(self, batch: torch.Tensor) -> int:
        """Train on ``batch`` (normalized, BCHW) and return its size."""
        import torch
        from torch.nn import functional

        optimizer = self._optimizer
        if optimizer is None:
            raise RuntimeError("reset() must run before the first step")
        images = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self._device, dtype=torch.bfloat16):
            logits = self._runner(images)
            loss = functional.cross_entropy(logits, self.labels[: images.shape[0]])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self._loss = loss.detach()
        return int(images.shape[0])

    def loss_is_finite(self) -> bool:
        import torch

        if self._loss is None:
            raise RuntimeError("no step has run since reset()")
        return bool(torch.isfinite(self._loss))

    def optimizer_state_size(self) -> int:
        return 0 if self._optimizer is None else len(self._optimizer.state)

    def ceiling(self, images: torch.Tensor, *, steps: int, warmup: int) -> float:
        """Images per second of the step alone on one resident batch; the warm-up covers compilation."""
        for _ in range(warmup):
            self.step(images)
        _synchronize(self._device)
        started = time.perf_counter()
        for _ in range(steps):
            self.step(images)
        _synchronize(self._device)
        return steps * int(images.shape[0]) / (time.perf_counter() - started)


def _synchronize(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
