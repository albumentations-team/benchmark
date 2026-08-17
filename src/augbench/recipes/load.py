from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from augbench.recipes.models import RecipeCatalog

if TYPE_CHECKING:
    from pathlib import Path


def load_recipe_catalog(path: Path) -> RecipeCatalog:
    with path.open(encoding="utf-8") as stream:
        return RecipeCatalog.model_validate(yaml.safe_load(stream))
