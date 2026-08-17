from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AdapterRegistration:
    """One RGB implementation and the module that builds its adapter."""

    implementation_id: str
    adapter_kind: Literal["python-module", "dali"]
    module: str
