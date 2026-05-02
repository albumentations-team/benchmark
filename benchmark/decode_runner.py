from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmark.decoders import DecoderUnavailableError, decode_video
from benchmark.results import build_metadata, summarize_runs, unsupported_result, write_results
from benchmark.term import tqdm_kwargs

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


class VideoDecodeRunner:
    def __init__(
        self,
        *,
        data_dir: Path,
        decoders: list[str],
        output_dir: Path,
        num_items: int | None = None,
        num_runs: int = 5,
        clip_length: int = 16,
        scenario: str = "video-decode-16f",
        min_time: float = 0.0,
    ) -> None:
        self.data_dir = data_dir
        self.decoders = decoders
        self.output_dir = output_dir
        self.num_items = num_items
        self.num_runs = num_runs
        self.clip_length = clip_length
        self.scenario = scenario
        self.min_time = min_time

    def _video_paths(self) -> list[Path]:
        paths = sorted(path for path in self.data_dir.rglob("*") if path.suffix.lower() in _VIDEO_EXTENSIONS)
        if self.num_items is not None:
            paths = paths[: self.num_items]
        if not paths:
            raise ValueError(f"No video files found in {self.data_dir}")
        return paths

    def _run_decoder(self, decoder: str, paths: list[Path]) -> dict[str, Any]:
        throughputs: list[float] = []
        frame_throughputs: list[float] = []
        times: list[float] = []
        sample_shape: tuple[int, ...] | None = None
        sample_dtype: str | None = None
        sample_device: str | None = None

        try:
            sample = decode_video(decoder, paths[0], self.clip_length)
            sample_shape = sample.shape
            sample_dtype = sample.dtype
            sample_device = sample.device
        except DecoderUnavailableError as e:
            return unsupported_result(str(e))
        except Exception as e:
            return unsupported_result(f"Decoder validation failed: {type(e).__name__}: {e}")

        for _ in tqdm(range(self.num_runs), desc=f"Decoding ({decoder})", leave=False, **tqdm_kwargs()):
            start = time.perf_counter()
            decoded = 0
            while True:
                for path in paths:
                    _ = decode_video(decoder, path, self.clip_length)
                    decoded += 1
                if time.perf_counter() - start >= self.min_time:
                    break
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            throughputs.append(decoded / elapsed)
            frame_throughputs.append(decoded * self.clip_length / elapsed)

        result = summarize_runs(throughputs, times)
        result.update(
            {
                "frame_throughputs": frame_throughputs,
                "median_frame_throughput": summarize_runs(frame_throughputs, times)["median_throughput"],
                "clip_length": self.clip_length,
                "sample_shape": sample_shape,
                "sample_dtype": sample_dtype,
                "sample_device": sample_device,
            },
        )
        return result

    def run(self) -> dict[str, Any]:
        paths = self._video_paths()
        logger.info("Running video decode benchmark for %d videos, %d frames/clip", len(paths), self.clip_length)

        results: dict[str, Any] = {}
        for decoder in self.decoders:
            logger.info("Benchmarking decoder: %s", decoder)
            results[decoder] = self._run_decoder(decoder, paths)

            payload = {
                "metadata": build_metadata(
                    scenario=self.scenario,
                    mode="decode",
                    decoder=decoder,
                    benchmark_params={
                        "num_videos": len(paths),
                        "num_runs": self.num_runs,
                        "clip_length": self.clip_length,
                        "min_time": self.min_time,
                        "decode_included": True,
                        "frame_sampling": "uniform",
                    },
                    timing_backend="perf_counter",
                    measurement_scope="decode_only",
                    data_source="disk",
                    data_dir=self.data_dir,
                    media="video",
                    includes_decode=True,
                    includes_collate=False,
                    includes_gpu_transfer=decoder == "dali",
                    includes_dataloader_workers=False,
                    repo_root=Path(__file__).parent.parent,
                ),
                "results": {decoder: results[decoder]},
            }
            write_results(self.output_dir / f"{decoder}_decode_results.json", payload)

        full_payload = {
            "metadata": build_metadata(
                scenario=self.scenario,
                mode="decode",
                benchmark_params={
                    "num_videos": len(paths),
                    "num_runs": self.num_runs,
                    "clip_length": self.clip_length,
                    "min_time": self.min_time,
                    "decode_included": True,
                    "frame_sampling": "uniform",
                },
                timing_backend="perf_counter",
                measurement_scope="decode_only",
                data_source="disk",
                data_dir=self.data_dir,
                media="video",
                includes_decode=True,
                includes_collate=False,
                includes_gpu_transfer=any(decoder == "dali" for decoder in self.decoders),
                includes_dataloader_workers=False,
                repo_root=Path(__file__).parent.parent,
            ),
            "results": results,
        }
        write_results(self.output_dir / "video_decode_results.json", full_payload)
        return full_payload
