"""GCP instance configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GCPInstanceConfig:
    """Configuration for a GCP Compute Engine instance used to run benchmarks.

    Attributes:
        project:        GCP project ID.
        zone:           Compute Engine zone, e.g. "us-central1-a".
        machine_type:   Machine type, e.g. "n1-standard-8" or "a2-highgpu-1g".
        image_family:   Boot disk image family (default: deep learning VM with CUDA).
        image_project:  Project that owns the image family.
        disk_size_gb:   Boot disk size in GB.
        accelerator_type: Optional GPU accelerator type, e.g. "nvidia-tesla-t4".
        accelerator_count: Number of GPUs (0 for CPU-only).
        preemptible:    Whether to use a preemptible (spot) instance.
        tags:           Network tags applied to the instance.
        scopes:         OAuth scopes granted to the instance service account.
    """

    project: str
    zone: str
    machine_type: str = "n1-standard-8"
    image_family: str = "pytorch-latest-cpu"
    image_project: str = "deeplearning-platform-release"
    disk_size_gb: int = 100
    accelerator_type: str | None = None
    accelerator_count: int = 0
    preemptible: bool = True
    tags: list[str] = field(default_factory=lambda: ["benchmark"])
    scopes: list[str] = field(default_factory=lambda: ["https://www.googleapis.com/auth/cloud-platform"])

    @property
    def instance_name(self) -> str:
        return f"benchmark-{self.machine_type.replace('/', '-')}"

    @classmethod
    def cpu(cls, project: str, zone: str = "us-central1-a", machine_type: str = "n1-standard-8") -> GCPInstanceConfig:
        """Convenience factory for a CPU-only benchmark instance."""
        return cls(
            project=project,
            zone=zone,
            machine_type=machine_type,
            image_family="pytorch-latest-cpu",
            accelerator_type=None,
            accelerator_count=0,
        )

    @classmethod
    def gpu_t4(cls, project: str, zone: str = "us-central1-a") -> GCPInstanceConfig:
        """Convenience factory for an NVIDIA T4 GPU benchmark instance."""
        return cls(
            project=project,
            zone=zone,
            machine_type="n1-standard-4",
            image_family="pytorch-latest-gpu",
            accelerator_type="nvidia-tesla-t4",
            accelerator_count=1,
        )

    @classmethod
    def gpu_a100(cls, project: str, zone: str = "us-central1-a") -> GCPInstanceConfig:
        """Convenience factory for an NVIDIA A100 GPU benchmark instance."""
        return cls(
            project=project,
            zone=zone,
            machine_type="a2-highgpu-1g",
            image_family="pytorch-latest-gpu",
            accelerator_type="nvidia-tesla-a100",
            accelerator_count=1,
        )
