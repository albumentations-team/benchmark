import io
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from augbench.adapters.registry import default_adapter_registry
from augbench.adapters.sources import SourceRef
from augbench.datasets.prepare_grayscale import prepare_grayscale_archive
from augbench.frozen_run import build_frozen_run
from augbench.recipes.load import load_recipe_catalog
from augbench.run_config import FamilyRunConfig, load_family_config

ROOT = Path(__file__).parents[1]


def test_grayscale_archive_contains_deterministic_one_channel_jpegs(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tar"
    exif = Image.Exif()
    exif[274] = 6
    with io.BytesIO() as buffer, tarfile.open(source, "w") as archive:
        Image.new("RGB", (240, 256), (123, 44, 173)).save(buffer, format="JPEG", exif=exif)
        member = tarfile.TarInfo("val/example.JPEG")
        member.size = buffer.tell()
        buffer.seek(0)
        archive.addfile(member, buffer)

    output = tmp_path / "gray.tar"
    first = prepare_grayscale_archive(source, output)
    second = prepare_grayscale_archive(source, tmp_path / "repeat.tar")
    assert first["file_count"] == 1
    assert first["sha256"] == second["sha256"]
    with tarfile.open(output) as archive:
        stream = archive.extractfile("val/example.JPEG")
        assert stream is not None
        with stream, Image.open(io.BytesIO(stream.read())) as image:
            assert image.format == "JPEG"
            assert image.mode == "L"
            assert image.size == (240, 256)
            assert not image.getexif()


def test_grayscale_colorize_has_a_separate_frozen_matrix() -> None:
    stock = load_family_config(ROOT / "configs/families/grayscale.yaml")
    archive = {"archive_uri": "gs://bucket/gray.tar", "archive_sha256": "c" * 64}
    config = FamilyRunConfig.model_validate(
        {**stock.model_dump(), "dataset": {**stock.dataset.model_dump(), **archive}}
    )
    frozen = build_frozen_run(repository_root=ROOT, git_commit="a" * 40, code_archive_sha256="b" * 64, config=config)
    assert config.implementations == (
        "albumentationsx_cpu",
        "pillow_cpu",
        "kornia_cpu",
        "kornia_gpu",
        "dali_gpu",
        "pillow_allsimd_cpu",
    )
    assert config.execution == load_family_config(ROOT / "configs/families/rgb.yaml").execution
    assert config.jpeg_channels == 1
    assert config.output.shape == (3, 224, 224)
    assert {cell.family for cell in frozen.cells} == {"grayscale"}
    assert {cell.recipe_id for cell in frozen.cells} == {"RandomCrop224+Colorize"}
    assert len(frozen.cells) == 30
    assert frozen.request.dataset_archive_uri == "gs://bucket/gray.tar"


@pytest.mark.parametrize(
    ("implementation", "dependency"),
    [
        ("albumentationsx_cpu", "simplejpeg"),
        ("pillow_cpu", "torch"),
        ("kornia_cpu", "kornia"),
    ],
)
def test_colorize_uses_real_grayscale_jpeg_input(tmp_path: Path, implementation: str, dependency: str) -> None:
    pytest.importorskip(dependency)
    if implementation == "albumentationsx_cpu":
        pytest.importorskip("albumentations")
    path = tmp_path / "gray.JPEG"
    Image.new("L", (248, 272)).save(path, format="JPEG")
    adapter = default_adapter_registry().load(implementation)
    (recipe,) = load_recipe_catalog(ROOT / "catalog/recipes/grayscale.yaml").recipes
    sample = adapter.load_source(SourceRef(path, channels=1))
    transform = adapter.build_recipe(recipe).sample_transform
    assert transform is not None
    color = transform(sample)
    if implementation == "albumentationsx_cpu":
        assert sample.shape == (272, 248, 1)
        assert color.shape == (224, 224, 3)
        assert color[0, 0].tolist() == [8, 24, 64]
    else:
        assert color.shape == (3, 224, 224)
        assert color[:, 0, 0].tolist() == pytest.approx(
            [8, 24, 64] if implementation == "pillow_cpu" else [8 / 255, 24 / 255, 64 / 255]
        )
