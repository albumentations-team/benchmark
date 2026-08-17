# Benchmark environment

`environments/rgb/requirements.in` declares the sole production worker
environment. `environments/rgb/lock.txt` is its Linux x86-64 lock with hashes.

Before a VM measures a cell, the bootstrap restores a cached environment whose
key is the SHA-256 of `lock.txt`, or builds that environment once and stores the
cache. The locked environment is outside the timed boundary. Its hash is part of
the immutable `RunRecord`, alongside the code archive, dataset archive and selection rule, recipe
catalog, and hardware configuration.

Changing the lock creates a new run. Local development may use a separate
environment; it is not production evidence.
