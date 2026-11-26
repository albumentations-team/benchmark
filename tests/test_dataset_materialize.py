import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from augbench.dataset_materialize import materialize_rgb_dataset
from augbench.run_config import DatasetConfig


def test_materialize_selects_sorted_jpegs_and_reuses_a_complete_cache(tmp_path: Path) -> None:
    pixels = Image.new("RGB", (48, 32), color=(41, 79, 137))
    exif = Image.Exif()
    exif[274] = 6
    with io.BytesIO() as clean, io.BytesIO() as tagged:
        pixels.save(clean, format="JPEG")
        pixels.save(tagged, format="JPEG", exif=exif)
        xmp = b"http://ns.adobe.com/xap/1.0/\x00<xmpmeta/>"
        app1 = b"\xff\xe1" + (len(xmp) + 2).to_bytes(2, "big") + xmp
        expected = clean.getvalue()[:2] + app1 + clean.getvalue()[2:]
        source_bytes = tagged.getvalue()[:2] + app1 + tagged.getvalue()[2:]

    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        for name, payload in (("val/b.JPEG", expected), ("val/a.JPEG", source_bytes), ("val/skip.png", b"skip")):
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

    cache = tmp_path / "cache"
    old_key = hashlib.sha256(
        json.dumps(dataset.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    old_file = cache / old_key / "val" / "a.JPEG"
    old_file.parent.mkdir(parents=True)
    old_file.write_bytes(source_bytes)

    first = materialize_rgb_dataset(archive=archive, dataset=dataset, cache_root=cache)
    second = materialize_rgb_dataset(archive=archive, dataset=dataset, cache_root=cache)

    assert first.files == (first.root / "val/a.JPEG",)
    assert first.root != old_file.parent.parent
    assert first.files[0].read_bytes() == expected
    with Image.open(first.files[0]) as image:
        assert image.size == (48, 32)
        assert not image.getexif()
    assert not first.root.joinpath("val/b.JPEG").exists()
    assert not first.cache_hit
    assert second.cache_hit


@pytest.mark.parametrize(
    ("selection", "expected"),
    [("first-sorted", ("a.JPEG", "b.JPEG")), ("first-sorted-non-cmyk", ("b.JPEG", "c.JPEG"))],
)
def test_shared_selection_excludes_cmyk_without_reencoding_other_jpegs(
    tmp_path: Path, selection: str, expected: tuple[str, ...]
) -> None:
    payloads = {}
    for name, mode in (("a.JPEG", "CMYK"), ("b.JPEG", "RGB"), ("c.JPEG", "L")):
        stream = io.BytesIO()
        Image.new(mode, (48, 32)).save(stream, format="JPEG")
        payloads[name] = stream.getvalue()
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as source:
        for name, payload in payloads.items():
            member = tarfile.TarInfo("val/" + name)
            member.size = len(payload)
            source.addfile(member, io.BytesIO(payload))
    config = DatasetConfig.model_validate(
        {
            "archive_uri": "gs://bucket/source.tar",
            "archive_sha256": "a" * 64,
            "archive_member_prefix": "val/",
            "archive_member_suffix": ".JPEG",
            "item_count": 2,
            "selection": selection,
            "access_order": "seeded-random-without-replacement",
        }
    )
    dataset = materialize_rgb_dataset(archive=archive, dataset=config, cache_root=tmp_path / "cache")
    assert tuple(path.name for path in dataset.files) == expected
    assert all(path.read_bytes() == payloads[path.name] for path in dataset.files)
