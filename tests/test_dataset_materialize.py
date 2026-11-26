import hashlib
import io
import json
import tarfile
from pathlib import Path

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


def test_materialize_converts_a_cmyk_jpeg_to_rgb_and_leaves_other_files_untouched(tmp_path: Path) -> None:
    with io.BytesIO() as rgb, io.BytesIO() as gray, io.BytesIO() as cmyk:
        Image.new("RGB", (16, 16), color=(10, 200, 30)).save(rgb, format="JPEG")
        Image.new("L", (16, 16), color=90).save(gray, format="JPEG")
        Image.new("CMYK", (16, 16), color=(0, 255, 255, 0)).save(cmyk, format="JPEG")
        payloads = {"val/a.JPEG": rgb.getvalue(), "val/b.JPEG": gray.getvalue(), "val/c.JPEG": cmyk.getvalue()}
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    dataset = DatasetConfig(
        archive_uri="gs://bucket/source.tar",
        archive_sha256="a" * 64,
        archive_member_prefix="val/",
        archive_member_suffix=".JPEG",
        item_count=3,
        selection="first-sorted",
        access_order="seeded-random-without-replacement",
    )

    staged = materialize_rgb_dataset(archive=archive, dataset=dataset, cache_root=tmp_path / "cache")

    modes = {path.name: Image.open(path).mode for path in staged.files}
    assert modes == {"a.JPEG": "RGB", "b.JPEG": "L", "c.JPEG": "RGB"}
    assert staged.files[0].read_bytes() == payloads["val/a.JPEG"]
    assert staged.files[1].read_bytes() == payloads["val/b.JPEG"]
    assert Image.open(staged.files[2]).convert("RGB").getpixel((8, 8))[0] > 200  # magenta plus yellow ink makes red
