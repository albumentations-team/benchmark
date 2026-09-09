from threading import Event

from augbench.nvml_memory import NvmlProcessMemoryMonitor


def test_nvml_monitor_reports_the_peak_from_the_same_execution_pass() -> None:
    reached_peak = Event()
    values = iter((128.0, 512.0, 256.0))

    def sample() -> float | None:
        value = next(values, 256.0)
        if value == 512.0:
            reached_peak.set()
        return value

    monitor = NvmlProcessMemoryMonitor(sample=sample, poll_interval_ms=1)
    monitor.start()
    assert reached_peak.wait(timeout=1)
    memory = monitor.stop()

    assert memory.measurement == "nvml_process_memory"
    assert memory.peak_mib == 512.0
    assert memory.valid_samples >= 2
