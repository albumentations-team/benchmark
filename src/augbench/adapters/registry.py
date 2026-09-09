from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

from augbench.adapters.models import AdapterRegistration

if TYPE_CHECKING:
    from augbench.adapters.runtime import AugmentationAdapter


class AdapterRegistry:
    def __init__(self, registrations: tuple[AdapterRegistration, ...]) -> None:
        self._registrations = {registration.implementation_id: registration for registration in registrations}
        if len(self._registrations) != len(registrations):
            raise ValueError("adapter registry contains duplicate implementation IDs")

    def get(self, implementation_id: str) -> AdapterRegistration:
        try:
            return self._registrations[implementation_id]
        except KeyError as error:
            raise ValueError(f"unknown implementation {implementation_id!r}") from error

    def all(self) -> tuple[AdapterRegistration, ...]:
        return tuple(self._registrations[implementation_id] for implementation_id in sorted(self._registrations))

    def load(
        self,
        implementation_id: str,
    ) -> AugmentationAdapter:
        registration = self.get(implementation_id)
        module = importlib.import_module(registration.module)
        factory = module.__dict__.get("create_adapter")
        if not callable(factory):
            raise TypeError(f"adapter module {module.__name__!r} does not export create_adapter")
        adapter = cast("AugmentationAdapter", factory(implementation_id))
        if adapter.implementation_id != implementation_id:
            raise ValueError(f"adapter module {module.__name__!r} returned the wrong identity")
        return adapter


def default_adapter_registry() -> AdapterRegistry:
    return AdapterRegistry(
        (
            AdapterRegistration(
                implementation_id="albumentationsx_cpu",
                adapter_kind="python-module",
                module="augbench.adapters.albumentationsx.rgb",
            ),
            AdapterRegistration(
                implementation_id="pillow_cpu",
                adapter_kind="python-module",
                module="augbench.adapters.pillow.rgb",
            ),
            AdapterRegistration(
                implementation_id="torchvision_cpu",
                adapter_kind="python-module",
                module="augbench.adapters.torchvision.rgb",
            ),
            AdapterRegistration(
                implementation_id="torchvision_gpu",
                adapter_kind="python-module",
                module="augbench.adapters.torchvision.rgb",
            ),
            AdapterRegistration(
                implementation_id="kornia_cpu",
                adapter_kind="python-module",
                module="augbench.adapters.kornia.rgb",
            ),
            AdapterRegistration(
                implementation_id="kornia_gpu",
                adapter_kind="python-module",
                module="augbench.adapters.kornia.rgb",
            ),
            AdapterRegistration(
                implementation_id="dali_gpu",
                adapter_kind="dali",
                module="augbench.adapters.dali.rgb",
            ),
        ),
    )
