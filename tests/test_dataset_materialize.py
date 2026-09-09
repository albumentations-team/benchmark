import io
import tarfile
from pathlib import Path

from augbench.dataset_materialize import materialize_rgb_dataset
from augbench.run_config import DatasetConfig


def test_materialize_selects_sorted_jpegs_and_reuses_a_complete_cache(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        for name, payload in (("val/b.JPEG", b"b"), ("val/a.JPEG", b"a"), ("val/skip.png", b"skip")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    dataset = DatasetConfig(
        archive_uri="gs://bucket/source.tar",
        archive_sha256="a" * 64,
        archive_member_prefix="val/",
        archive_member_suffix=".JPEG",
        item_count=1,
        selection="first-sorted",
        access_order="seeded-random-without-replacement",
    )

    first = materialize_rgb_dataset(archive=archive, dataset=dataset, cache_root=tmp_path / "cache")
    second = materialize_rgb_dataset(archive=archive, dataset=dataset, cache_root=tmp_path / "cache")

    assert first.files == (first.root / "val/a.JPEG",)
    assert first.files[0].read_bytes() == b"a"
    assert not first.root.joinpath("val/b.JPEG").exists()
    assert not first.cache_hit
    assert second.cache_hit
