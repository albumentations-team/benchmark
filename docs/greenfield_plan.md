# Benchmark architecture

The [execution contract](benchmark_execution_contract.md) defines the RGB workload and
what a new benchmark family needs before it can run.

- [`frozen_run.py`](../src/augbench/frozen_run.py) hashes the workload, source,
  dataset, and environment locks into the run identity.
- [`gcp_controller.py`](../src/augbench/gcp_controller.py) provisions or reuses a VM.
  [`guest_worker.py`](../src/augbench/guest_worker.py) runs the pending cells and
  publishes each result to GCS.
- [`cell_executor.py`](../src/augbench/cell_executor.py) owns the timing window and
  output checks. The adapters own decoding and recipe execution.
- [`training/`](../src/augbench/training/step.py) holds the training step. `cell_executor.py` feeds each
  ready batch to it when the family config has a `training` section.
- The [report generators](../paper/README.md#regenerate-the-results) validate the
  complete matrix and preserve the source run and cell IDs.
