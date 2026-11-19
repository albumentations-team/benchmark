"""A transform invocation copied from the frozen recipe catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformSpec:
    """One non-terminal recipe stage in the implementation's native API."""

    name: str
    params: dict[str, Any]


def parameter_range(value: Any) -> tuple[Any, Any]:
    if isinstance(value, (list, tuple)):
        low, high = value
        return low, high
    return value, value


def odd_kernel_sizes(value: Any) -> range:
    low, high = parameter_range(value)
    sizes = range(max(3, low | 1), high + 1, 2)
    if not sizes:
        raise ValueError(f"kernel range {value!r} must contain an odd size of at least 3")
    return sizes
