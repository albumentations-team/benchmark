from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from augbench.recipes.runtime import UnsupportedRecipeError

if TYPE_CHECKING:
    from collections.abc import Callable

    from augbench.recipes.models import RecipeSpec

Placement = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class RecipeRuntime:
    implementation_id: str
    recipe_id: str
    placement: Placement
    sample_transform: Callable[[Any], Any] | None
    host_batch_transform: Callable[[Any], Any] | None
    batch_transform: Callable[[Any], Any] | None


class AugmentationAdapter(Protocol):
    implementation_id: str
    placement: Placement

    def load_source(self, source: Any) -> Any: ...

    def build_recipe(self, recipe: RecipeSpec) -> RecipeRuntime: ...

    def metadata(self) -> dict[str, str]: ...


@dataclass(frozen=True)
class PythonModuleAdapterConfig:
    implementation_id: str
    recipe_module: str
    placement: Placement
    source_id: str
    unsupported_operation_ids: frozenset[str] = frozenset()


class PythonModuleAdapter:
    implementation_id: str
    placement: Placement

    def __init__(
        self,
        *,
        config: PythonModuleAdapterConfig,
        load_source: Callable[[Any], Any],
    ) -> None:
        self.implementation_id = config.implementation_id
        self.placement = config.placement
        self._recipe_module = config.recipe_module
        self._load_source = load_source
        self._source_id = config.source_id
        self._unsupported_operation_ids = config.unsupported_operation_ids

    def load_source(self, source: Any) -> Any:
        return self._load_source(source)

    def build_recipe(self, recipe: RecipeSpec) -> RecipeRuntime:
        if self.implementation_id not in recipe.supported_implementations:
            raise UnsupportedRecipeError(
                f"recipe {recipe.recipe_id!r} is unsupported by {self.implementation_id!r}",
            )
        unsupported = sorted(
            {stage.operation_id for stage in recipe.stages} & self._unsupported_operation_ids,
        )
        if unsupported:
            operation_ids = ", ".join(unsupported)
            raise UnsupportedRecipeError(
                f"operations {operation_ids} are unsupported by {self.implementation_id!r}",
            )
        module = importlib.import_module(self._recipe_module)
        build_recipe = getattr(module, "build_recipe", None)
        if not callable(build_recipe):
            raise TypeError(f"recipe module {module.__name__!r} must define build_recipe(recipe, implementation_id)")
        transform = build_recipe(recipe, self.implementation_id)
        call_transform = cast("Callable[[Any, Any], Any]", module.__dict__["__call__"])
        sample_transform: Callable[[Any], Any] | None
        host_batch_transform: Callable[[Any], Any] | None = None
        batch_transform: Callable[[Any], Any] | None

        cpu_transform = getattr(transform, "cpu_transform", None)
        gpu_transform = getattr(transform, "gpu_transform", None)
        if self.placement == "cpu":
            if getattr(transform, "defer_batch_to_gpu", False):
                if cpu_transform is None or gpu_transform is None:
                    raise TypeError("a deferred CPU recipe requires both cpu_transform and gpu_transform")
                sample_transform = partial(call_transform, cpu_transform)
                host_batch_transform = _optional_host_batch_transform(transform)
                batch_transform = _LazyDeviceTransform(gpu_transform)
            else:
                sample_transform = partial(call_transform, transform)
                batch_transform = None
        else:
            sample_transform = None if cpu_transform is None else partial(call_transform, cpu_transform)
            host_batch_transform = _optional_host_batch_transform(transform)
            device_transform = transform if gpu_transform is None else gpu_transform
            batch_transform = _LazyDeviceTransform(device_transform)

        return RecipeRuntime(
            implementation_id=self.implementation_id,
            recipe_id=recipe.recipe_id,
            placement=self.placement,
            sample_transform=sample_transform,
            host_batch_transform=host_batch_transform,
            batch_transform=batch_transform,
        )

    def metadata(self) -> dict[str, str]:
        return {
            "adapter": type(self).__name__,
            "recipe_module": self._recipe_module,
            "placement": self.placement,
            "source_id": self._source_id,
            "unsupported_operation_ids": ",".join(sorted(self._unsupported_operation_ids)),
        }


class _LazyDeviceTransform:
    def __init__(self, transform: Any) -> None:
        self._transform: Callable[[Any], Any] = cast("Callable[[Any], Any]", transform)
        self._moved = False

    def __call__(self, batch: Any) -> Any:
        if not self._moved:
            move = getattr(self._transform, "to", None)
            if callable(move):
                moved = move("cuda")
                if not callable(moved):
                    raise TypeError("GPU recipe transform must remain callable after moving to CUDA")
                self._transform = moved
            self._moved = True
        return self._transform(batch)


def _optional_host_batch_transform(transform: Any) -> Callable[[Any], Any] | None:
    host_batch_transform = getattr(transform, "host_batch_transform", None)
    if host_batch_transform is not None and not callable(host_batch_transform):
        raise TypeError("host_batch_transform must be callable")
    return cast("Callable[[Any], Any] | None", host_batch_transform)
