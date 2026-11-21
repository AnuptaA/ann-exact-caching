# Environments

| File | Environment | Used for |
|---|---|---|
| `core.txt` | `.venv-core` (Python 3.11) | `simulations/approximate_hits/cachecore/build.sh`, online simulator on the C++ core, timing runs |
| `ann.txt` | `.venv-ann` | real Aker through the shim, DiskANN index builds and searches, throughput |
| `prep.txt` | `.venv` | dataset preparation, embeddings, offline evaluations, plots |
| `harness.txt` | Q1 venv | `harness/` and `workloads/q1/` (see `requirements-q1.lock` for hashes) |

Only the versions recorded by the advisor (core, ann) are pinned; the rest are unpinned because none were recorded.
Create one with `python3.11 -m venv .venv-core && .venv-core/bin/pip install -r envs/core.txt`.
