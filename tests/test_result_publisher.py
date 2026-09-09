from augbench.result_publisher import ResultPublisher
from augbench.run_records import CellKey, GpuMemory, OutputObservation, ResultRecord, Throughput


class _ObjectStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        if key in self.values:
            return False
        self.values[key] = payload
        return True

    def read(self, key: str) -> bytes:
        return self.values[key]


def _result() -> ResultRecord:
    cell = CellKey(run_id="a" * 64, family="rgb", implementation="pillow_cpu", recipe_id="Resize", seed=137)
    return ResultRecord(
        run_id=cell.run_id,
        cell=cell,
        status="ok",
        throughput=Throughput(completed_items=8192, duration_seconds=10),
        gpu_memory=GpuMemory(peak_mib=512, poll_interval_ms=50, valid_samples=3),
        output=OutputObservation(shape=(256, 3, 224, 224)),
    )


def test_result_publisher_uses_one_conditional_gcs_object_per_cell() -> None:
    remote = _ObjectStore()
    result = _result()
    publisher = ResultPublisher(remote)

    assert publisher.publish(result)
    assert not publisher.publish(result)
    assert set(remote.values) == {f"runs/{result.run_id}/cells/{result.cell_id}.json"}
