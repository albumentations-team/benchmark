from __future__ import annotations

VM_WORKDIR = "/root/benchmark-work"
VM_REPODIR = f"{VM_WORKDIR}/repo"
VM_DATADIR = "/root/benchmark-data"
VM_RESULTS = f"{VM_WORKDIR}/results"


def staged_data_dir_for_gcs_uri(gcs_uri: str | None) -> str:
    """Return the dataset path the detached GCP startup script will pass to benchmark runners."""
    uri = (gcs_uri or "").rstrip("/")
    if not uri.startswith("gs://"):
        return VM_DATADIR
    base = uri.rsplit("/", 1)[-1].lower()
    if base.endswith(".tar") and base.startswith("val"):
        return f"{VM_DATADIR}/val"
    if base in {"val", "train", "test"}:
        return f"{VM_DATADIR}/{base}"
    return VM_DATADIR
