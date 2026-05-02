from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

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
    assert str(clip.dtype).endswith("float16")


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
