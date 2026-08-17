"""Low-overhead process-wide CUDA-memory measurement via NVML."""

from __future__ import annotations

import os
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

from augbench.run_records import GpuMemory

if TYPE_CHECKING:
    from collections.abc import Callable


class NvmlProcessMemoryMonitor:
    """Poll current-process GPU memory without spawning ``nvidia-smi``."""

    def __init__(self, *, sample: Callable[[], float | None] | None = None, poll_interval_ms: int = 50) -> None:
        if poll_interval_ms < 1:
            raise ValueError("poll_interval_ms must be positive")
        self._owned_sampler: _NvmlProcessSampler | None = None
        if sample is None:
            self._owned_sampler = _NvmlProcessSampler()
            self._sample_process_memory: Callable[[], float | None] = self._owned_sampler
        else:
            self._sample_process_memory = sample
        self._poll_interval_ms = poll_interval_ms
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._peak_mib = 0.0
        self._valid_samples = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU memory monitor is already running")
        self._record_sample()
        self._thread = Thread(target=self._run, name="augbench-nvml-memory", daemon=True)
        self._thread.start()

    def stop(self) -> GpuMemory:
        if self._thread is None:
            raise RuntimeError("GPU memory monitor was not started")
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._poll_interval_ms / 250.0))
        self._record_sample()
        try:
            with self._lock:
                return GpuMemory(
                    peak_mib=self._peak_mib,
                    poll_interval_ms=self._poll_interval_ms,
                    valid_samples=self._valid_samples,
                )
        finally:
            if self._owned_sampler is not None:
                self._owned_sampler.close()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval_ms / 1000.0):
            self._record_sample()

    def _record_sample(self) -> None:
        value = self._sample_process_memory()
        if value is None:
            return
        with self._lock:
            self._valid_samples += 1
            self._peak_mib = max(self._peak_mib, value)


class _NvmlProcessSampler:
    """Hold one NVML session for the complete benchmark cell."""

    def __init__(self) -> None:
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError("NVML Python bindings are required for production GPU-memory measurement") from error
        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def __call__(self) -> float:
        processes: list[Any] = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        used_bytes = sum(
            int(process.usedGpuMemory)
            for process in processes
            if int(process.pid) == os.getpid() and int(process.usedGpuMemory) >= 0
        )
        return used_bytes / (1024 * 1024)

    def close(self) -> None:
        self._pynvml.nvmlShutdown()
