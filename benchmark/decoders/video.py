from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class DecoderUnavailableError(RuntimeError):
    """Raised when an optional decoder dependency is not installed."""


@dataclass(frozen=True)
class DecodedClip:
    frames: Any
    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"


VideoDecoder = Callable[..., DecodedClip]


def _uniform_indices(frame_count: int, clip_length: int) -> list[int]:
    if frame_count <= 0:
        return list(range(clip_length))
    if frame_count >= clip_length:
        return np.linspace(0, frame_count - 1, clip_length, dtype=np.int64).tolist()
    return [min(i, frame_count - 1) for i in range(clip_length)]


def _ensure_clip_length(frames: list[np.ndarray], clip_length: int) -> list[np.ndarray]:
    if not frames:
        raise ValueError("No frames decoded")
    while len(frames) < clip_length:
        frames.append(frames[-1].copy())
    return frames[:clip_length]


def _decode_opencv(path: Path, clip_length: int) -> DecodedClip:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    for idx in _uniform_indices(frame_count, clip_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    clip = np.ascontiguousarray(np.stack(_ensure_clip_length(frames, clip_length), axis=0))
    return DecodedClip(frames=clip, shape=tuple(clip.shape), dtype=str(clip.dtype))


def _decode_pyav(path: Path, clip_length: int) -> DecodedClip:
    try:
        import av
    except ImportError as e:
        raise DecoderUnavailableError("PyAV is not installed") from e

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)
        wanted = set(_uniform_indices(total_frames, clip_length))
        frames: list[np.ndarray] = []
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted or total_frames <= 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= clip_length:
                    break

    clip = np.ascontiguousarray(np.stack(_ensure_clip_length(frames, clip_length), axis=0))
    return DecodedClip(frames=clip, shape=tuple(clip.shape), dtype=str(clip.dtype))


def _decode_decord(path: Path, clip_length: int) -> DecodedClip:
    try:
        from decord import VideoReader
    except ImportError as e:
        raise DecoderUnavailableError("Decord is not installed") from e

    vr = VideoReader(str(path))
    indices = _uniform_indices(len(vr), clip_length)
    clip = vr.get_batch(indices).asnumpy()
    clip = np.ascontiguousarray(clip)
    return DecodedClip(frames=clip, shape=tuple(clip.shape), dtype=str(clip.dtype))


def _decode_torchvision(path: Path, clip_length: int) -> DecodedClip:
    try:
        import torch
        import torchvision
    except ImportError as e:
        raise DecoderUnavailableError("torchvision is not installed") from e

    video, _, _ = torchvision.io.read_video(str(path), output_format="TCHW")
    if video.numel() == 0:
        raise ValueError(f"No frames decoded from {path}")
    indices = torch.tensor(_uniform_indices(int(video.shape[0]), clip_length), dtype=torch.long)
    clip = video.index_select(0, indices).contiguous()
    return DecodedClip(frames=clip, shape=tuple(int(x) for x in clip.shape), dtype=str(clip.dtype))


def _decode_torchcodec(path: Path, clip_length: int) -> DecodedClip:
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise DecoderUnavailableError("TorchCodec is not installed") from e

    decoder = VideoDecoder(str(path))
    frame_count = len(decoder)
    frames = decoder.get_frames_at(indices=_uniform_indices(frame_count, clip_length)).data
    return DecodedClip(frames=frames, shape=tuple(int(x) for x in frames.shape), dtype=str(frames.dtype))


def _decode_dali(path: Path, clip_length: int) -> DecodedClip:
    try:
        from nvidia.dali import fn, pipeline_def, types
    except ImportError as e:
        raise DecoderUnavailableError("NVIDIA DALI is not installed") from e

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe() -> Any:
        return fn.readers.video(
            device="gpu",
            filenames=[str(path)],
            sequence_length=clip_length,
            random_shuffle=False,
            dtype=types.UINT8,
        )

    pipeline = pipe()
    pipeline.build()
    (clip_batch,) = pipeline.run()
    clip = clip_batch.as_cpu().as_array()[0]
    return DecodedClip(frames=clip, shape=tuple(int(x) for x in clip.shape), dtype=str(clip.dtype), device="gpu")


_DECODERS: dict[str, VideoDecoder] = {
    "opencv": _decode_opencv,
    "pyav": _decode_pyav,
    "decord": _decode_decord,
    "torchvision": _decode_torchvision,
    "torchcodec": _decode_torchcodec,
    "dali": _decode_dali,
}


def get_video_decoder_names() -> tuple[str, ...]:
    return tuple(_DECODERS)


def decode_video(decoder: str, path: Path, clip_length: int) -> DecodedClip:
    try:
        decode_fn = _DECODERS[decoder]
    except KeyError as e:
        msg = f"Unknown decoder {decoder!r}. Available decoders: {list(_DECODERS)}"
        raise ValueError(msg) from e
    return decode_fn(path, clip_length)
