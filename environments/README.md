# Benchmark environment

`environments/rgb/requirements.in` declares the base production environment.
`environments/rgb/lock.txt` pins Linux x86-64 packages with hashes, including the
PyPI build of PyTorch 2.15.0 with CUDA 13.2 and Triton 3.9.0. AX is built from the
source archive at the commit pinned in `requirements.in`. The archive hash
identifies the exact pre-release source even when its package metadata still
reports the previous release version.

Pillow-AllSIMD uses the same base packages. Bootstrap verifies
`environments/rgb/pillow-allsimd.lock.txt` and installs its wheel into a
separate directory. Only the AllSIMD worker prepends that directory to
`PYTHONPATH`. All other workers import Pillow's `PIL`. The AllSIMD worker
requires fast mode and records the active SIMD tier.

The archive omits the relocatable venv's Python links. Every VM restores the
matching managed Python and relinks the environment before it executes a cell.

Before a VM measures a cell, the bootstrap restores a cached environment whose
key is the SHA-256 of `lock.txt`, or builds that environment once and stores the
cache. Environment setup is outside the timed boundary. Both lock hashes are
part of the immutable `RunRecord`, alongside the code archive, dataset archive and selection rule, recipe
catalog, and hardware configuration.

Changing the lock creates a new run. Local development may use a separate
environment; it is not production evidence.
