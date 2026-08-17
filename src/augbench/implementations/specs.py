"""A transform invocation copied from the frozen recipe catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformSpec:
    """One non-terminal recipe stage in the implementation's native API."""

    name: str
    params: dict[str, Any]
