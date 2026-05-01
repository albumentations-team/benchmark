from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def load_from_python_file(specs_file: Path) -> tuple[str, Callable[[Any, Any], Any], list[dict[str, Any]]]:
    """Load library name, __call__ function, and transforms from a Python spec file."""
    spec = importlib.util.spec_from_file_location("custom_transforms", specs_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load from {specs_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "LIBRARY"):
        raise ValueError(f"Python file {specs_file} must define LIBRARY string")

    if "__call__" not in module.__dict__:
        raise TypeError(f"Python file {specs_file} must define __call__ function")

    call_attr = module.__dict__["__call__"]
    if not callable(call_attr):
        raise TypeError("__call__ must be a callable function")

    if not hasattr(module, "TRANSFORMS"):
        raise ValueError(f"Python file {specs_file} must define TRANSFORMS list")

    transforms = module.TRANSFORMS
    for i, transform in enumerate(transforms):
        if not isinstance(transform, dict):
            raise TypeError(f"TRANSFORMS[{i}] must be a dictionary")

        required_keys = {"name", "transform"}
        missing = required_keys - transform.keys()
        if missing:
            raise ValueError(f"TRANSFORMS[{i}] missing keys: {missing}")

    return str(module.LIBRARY), call_attr, transforms
