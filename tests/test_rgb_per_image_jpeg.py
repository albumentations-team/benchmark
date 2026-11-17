from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_gpu_jpeg_reader_keeps_stored_orientation(tmp_path: Path) -> None:
    image = pytest.importorskip("PIL.Image")
    from augbench.adapters.sources import SourceRef, load_jpeg_bytes

    pixels = image.new("RGB", (48, 32), color=(41, 79, 137))
    reference = tmp_path / "reference.JPEG"
    source = tmp_path / "rotated.JPEG"
    exif = image.Exif()
    exif[274] = 6
    pixels.save(reference)
    pixels.save(source, exif=exif)

    assert load_jpeg_bytes(SourceRef(source)) == reference.read_bytes()


def test_jpeg_bytes_collate_in_worker(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    image = pytest.importorskip("PIL.Image")
    from torch.utils.data import DataLoader

    from augbench.adapters.sources import SourceRef, load_jpeg_bytes

    source = tmp_path / "sample.JPEG"
    image.new("RGB", (48, 32), color=(41, 79, 137)).save(source)
    encoded = load_jpeg_bytes(SourceRef(source))
    batches = DataLoader([encoded] * 16, batch_size=16, num_workers=1, collate_fn=list)
    batch = next(iter(batches))

    assert len(batch) == 16
    assert all(isinstance(sample, bytes) and sample == encoded for sample in batch)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_torchvision_jpeg_preserves_sample_order(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    io = pytest.importorskip("torchvision.io")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    codec = pytest.importorskip("torchcodec.decoders")
    from augbench.implementations import torchvision_pipeline_impl as pipeline

    generator = torch.Generator().manual_seed(137)
    image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8, generator=generator).to(device)
    batch = image.repeat(4, 1, 1, 1)
    qualities = (31, 89, 31, 65)
    picks = iter(qualities)
    monkeypatch.setattr(pipeline.random, "randint", lambda _low, _high: next(picks))
    expected = torch.stack(
        [codec.decode_jpeg(io.encode_jpeg(image, quality).cpu(), device=device) for quality in qualities]
    )
    actual = pipeline._jpeg_round_trip(batch, (30, 90))
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual[0], actual[1])
