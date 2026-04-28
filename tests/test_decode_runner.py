from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmark.decode_runner import VideoDecodeRunner

if TYPE_CHECKING:
    from pathlib import Path


def _write_test_video(path: Path, *, num_frames: int = 6) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (16, 16),
    )
    try:
        for i in range(num_frames):
            frame = np.full((16, 16, 3), i * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_decode_runner_writes_opencv_results(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    video_dir.mkdir()
    _write_test_video(video_dir / "sample.mp4")

    runner = VideoDecodeRunner(
        data_dir=video_dir,
        decoders=["opencv"],
        output_dir=output_dir,
        num_items=1,
        num_runs=1,
        clip_length=4,
    )
    result = runner.run()

    assert result["results"]["opencv"]["supported"] is True
    assert result["results"]["opencv"]["clip_length"] == 4
    assert (output_dir / "opencv_decode_results.json").exists()
    assert (output_dir / "video_decode_results.json").exists()
