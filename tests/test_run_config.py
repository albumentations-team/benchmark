from pathlib import Path

import pytest

from augbench.run_config import load_family_config, load_gcp_config, source_order

ROOT = Path(__file__).parents[1]


def test_rgb_config_defines_the_entire_runnable_contract() -> None:
    config = load_family_config(ROOT / "configs" / "families" / "rgb.yaml")

    assert config.family == "rgb"
    assert config.output.shape == (3, 224, 224)
    assert config.output.dtype == "float16"
    assert config.normalization == "gpu"
    assert config.output.device == "cuda"
    assert config.output.layout == "BCHW"
    assert config.dataset.item_count == 10_000
    assert config.dataset.archive_member_prefix == "val/"
    assert config.dataset.archive_member_suffix == ".JPEG"
    assert config.execution.batch_size == 256
    assert config.execution.warmup_batches == 1
    assert config.execution.measured_batches == 32
    assert config.execution.seeds == (137, 138, 139)
    assert config.execution.num_workers == 15
    assert config.execution.prefetch_factor == 2
    assert config.execution.prewarm_dataset
    assert config.recipes == Path("catalog/recipes/rgb.yaml")
    assert config.implementations == (
        "albumentationsx_cpu",
        "pillow_cpu",
        "torchvision_cpu",
        "torchvision_gpu",
        "kornia_cpu",
        "kornia_gpu",
        "dali_gpu",
    )


def test_gcp_config_is_limited_to_one_standard_l4_vm() -> None:
    config = load_gcp_config(ROOT / "configs" / "cloud" / "gcp-l4.yaml")

    assert config.project == "albumentations"
    assert config.zones == "all"
    assert config.machine_type == "g2-standard-16"
    assert config.accelerator == "nvidia-l4"
    assert config.provisioning_model == "STANDARD"


def test_each_seed_has_a_stable_random_access_order() -> None:
    first = source_order(item_count=10_000, required_items=8448, seed=137)
    second = source_order(item_count=10_000, required_items=8448, seed=137)

    assert first == second
    assert len(first) == 8448
    assert len(set(first)) == len(first)
    assert first != tuple(range(8448))
    assert first != source_order(item_count=10_000, required_items=8448, seed=138)


def test_config_rejects_a_measurement_window_larger_than_the_dataset() -> None:
    with pytest.raises(ValueError, match="dataset"):
        source_order(item_count=32, required_items=33, seed=137)
