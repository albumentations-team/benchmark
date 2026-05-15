from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from benchmark.media.loaders import BenchmarkMediaLoader

if TYPE_CHECKING:
    from pathlib import Path


def test_video_micro_loader_decodes_fixed_length_tensor_clips(tmp_path: Path, monkeypatch: Any) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")
    seen: dict[str, Any] = {}

    class FakeDecodedClip:
        frames = np.zeros((4, 5, 6, 3), dtype=np.uint8)

    def fake_decode_video(decoder: str, path: Path, clip_length: int) -> FakeDecodedClip:
        seen["decoder"] = decoder
        seen["path"] = path
        seen["clip_length"] = clip_length
        return FakeDecodedClip()

    monkeypatch.setattr("benchmark.media.loaders.decode_video", fake_decode_video)
    loader = BenchmarkMediaLoader(
        library="kornia",
        data_dir=tmp_path,
        media="video",
        num_items=1,
        clip_length=4,
    )

    [clip] = loader.load()

    assert seen == {"decoder": "opencv", "path": video_path, "clip_length": 4}
    assert tuple(clip.shape) == (4, 3, 5, 6)
    assert str(clip.dtype).endswith("float32")
    device = getattr(clip, "device", None)
    assert getattr(device, "type", device) == "cpu"


def test_torchvision_video_micro_loader_keeps_uint8_clips(tmp_path: Path, monkeypatch: Any) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")

    class FakeDecodedClip:
        frames = np.zeros((4, 5, 6, 3), dtype=np.uint8)

    def fake_decode_video(_decoder: str, _path: Path, _clip_length: int) -> FakeDecodedClip:
        return FakeDecodedClip()

    monkeypatch.setattr("benchmark.media.loaders.decode_video", fake_decode_video)
    loader = BenchmarkMediaLoader(
        library="torchvision",
        data_dir=tmp_path,
        media="video",
        num_items=1,
        clip_length=4,
    )

    [clip] = loader.load()

    assert tuple(clip.shape) == (4, 3, 5, 6)
    assert str(clip.dtype).endswith("uint8")


def test_image_loader_scans_past_invalid_files_to_requested_count(tmp_path: Path) -> None:
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "gray.png")
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    for idx in range(2):
        Image.fromarray(np.full((4, 4, 3), idx, dtype=np.uint8)).save(tmp_path / f"valid_{idx}.png")

    loader = BenchmarkMediaLoader(
        library="albumentationsx",
        data_dir=tmp_path,
        media="image",
        num_items=2,
    )

    images = loader.load()

    assert len(images) == 2
    assert all(image.shape == (4, 4, 3) for image in images)
