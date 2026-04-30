from __future__ import annotations

import os
from contextlib import suppress
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

ThreadPolicy = Literal["micro-single", "pipeline-default", "pipeline-single-worker"]

_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def apply_thread_policy(policy: ThreadPolicy) -> None:
    """Apply benchmark thread policy outside transform specs."""
    if policy == "pipeline-default":
        return

    for name, value in _SINGLE_THREAD_ENV.items():
        os.environ[name] = value

    try:
        import cv2

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except ImportError:
        pass

    try:
        import torch

        torch.set_num_threads(1)
        # PyTorch only allows changing interop threads before parallel work starts.
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
    except ImportError:
        pass


def worker_init_for_policy(policy: ThreadPolicy) -> Callable[[int], None] | None:
    if policy == "pipeline-default":
        return None

    def _init_worker(_worker_id: int) -> None:
        apply_thread_policy(policy)

    return _init_worker
