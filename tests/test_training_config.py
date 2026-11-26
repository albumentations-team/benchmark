import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from augbench.frozen_run import build_frozen_run
from augbench.matrix import build_matrix
from augbench.recipe_contract import validate_rgb_recipes
from augbench.recipes.load import load_recipe_catalog
from augbench.run_config import FamilyRunConfig, TrainingConfig, load_family_config

ROOT = Path(__file__).parents[1]


def _training() -> dict[str, Any]:
    return {
        "model": "resnet18",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "ceiling_warmup_steps": 15,
        "ceiling_steps": 50,
    }


def _rgb_payload() -> dict[str, object]:
    return load_family_config(ROOT / "configs/families/rgb.yaml").model_dump(mode="json")


def test_rgb_run_identity_inputs_gain_no_training_key() -> None:
    frozen = build_frozen_run(repository_root=ROOT, git_commit="a" * 40, code_archive_sha256="b" * 64)
    published = json.loads((ROOT / "docs/generated/run.json").read_text(encoding="utf-8"))

    assert "training" not in frozen.run.family_config
    assert set(frozen.run.family_config) == set(published["family_config"])


def test_training_family_requires_a_training_section() -> None:
    payload = {**_rgb_payload(), "family": "imagenet_train", "recipes": "catalog/recipes/imagenet_train.yaml"}

    with pytest.raises(ValidationError, match="training"):
        FamilyRunConfig.model_validate(payload)
    assert FamilyRunConfig.model_validate({**payload, "training": _training()}).training == TrainingConfig(
        **_training()
    )


def test_input_families_reject_a_training_section() -> None:
    with pytest.raises(ValidationError, match="training"):
        FamilyRunConfig.model_validate({**_rgb_payload(), "training": _training()})


def test_training_section_is_strict() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(**{**_training(), "model": "vgg16"})
    with pytest.raises(ValidationError):
        TrainingConfig(**{**_training(), "ceiling_steps": 0})


@pytest.mark.parametrize(("name", "batch_size"), [("resnet18", 256), ("resnet50", 128)])
def test_training_configs_build_a_complete_matrix_that_fits_the_dataset(name: str, batch_size: int) -> None:
    config = load_family_config(ROOT / f"configs/families/train-{name}.yaml")
    recipes = load_recipe_catalog(ROOT / config.recipes)
    validate_rgb_recipes(config, recipes)
    cells = build_matrix(run_id="a" * 64, config=config, recipes=recipes)

    assert config.training is not None
    assert config.training.model == name
    assert config.execution.batch_size == batch_size
    assert config.dataset.item_count == 50_000
    assert config.execution.required_items <= config.dataset.item_count
    assert len(cells) == 5 * (8 + 6 + 6 + 8)
    assert {cell.family for cell in cells} == {"imagenet_train"}


def test_training_recipes_reuse_the_rgb_catalog_stage_definitions() -> None:
    rgb = {recipe.recipe_id: recipe for recipe in load_recipe_catalog(ROOT / "catalog/recipes/rgb.yaml").recipes}
    ladder = load_recipe_catalog(ROOT / "catalog/recipes/imagenet_train.yaml").recipes
    donors = {
        "RandomResizedCrop": rgb["RandomResizedCrop"],
        "HorizontalFlip": rgb["RandomCrop224+HorizontalFlip"],
        "ColorJitter": rgb["RandomCrop224+ColorJitter"],
        "Erasing": rgb["RandomCrop224+Erasing"],
    }
    for recipe in ladder:
        for stage in recipe.stages:
            if stage.operation_id == "RandAugment":
                continue  # the one operation that has no RGB-catalog donor
            donor = next(
                s
                for s in donors.get(stage.operation_id, rgb["Resize224"]).stages
                if s.operation_id == stage.operation_id
            )
            assert stage == donor, (recipe.recipe_id, stage.operation_id)
