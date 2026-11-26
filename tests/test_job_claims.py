from dataclasses import dataclass, field

from augbench.job_claims import claim_job, claim_key, release_stale_claims


@dataclass
class _Store:
    objects: dict[str, bytes] = field(default_factory=dict)

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        if key in self.objects:
            return False
        self.objects[key] = payload
        return True

    def read(self, key: str) -> bytes:
        return self.objects[key]

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(key for key in self.objects if key.startswith(prefix))

    def delete(self, key: str) -> None:
        del self.objects[key]


def test_exactly_one_vm_wins_a_job() -> None:
    store = _Store()
    first = claim_job(store=store, run_id="r", implementation="pillow_cpu", seed=137, instance="vm-1")
    second = claim_job(store=store, run_id="r", implementation="pillow_cpu", seed=137, instance="vm-2")
    other_seed = claim_job(store=store, run_id="r", implementation="pillow_cpu", seed=138, instance="vm-2")

    assert (first, second, other_seed) == (True, False, True)
    assert b"vm-1" in store.objects[claim_key("r", "pillow_cpu", 137)]


def test_only_claims_of_vanished_vms_are_released() -> None:
    store = _Store()
    claim_job(store=store, run_id="r", implementation="a", seed=1, instance="alive")
    claim_job(store=store, run_id="r", implementation="a", seed=2, instance="gone")

    released = release_stale_claims(store=store, run_id="r", active_instances={"alive"})

    assert released == 1
    assert list(store.objects) == [claim_key("r", "a", 1)]
