# Vector Cache Research

Research code on caching the results of top-K vector search for vector
databases. The current direction is approximate cache hits with a retention
target: a cache hit is allowed when the mean fraction of the true top-K that
the cached answer returns stays at or above a user target.

- Approximate-hits pipeline and scripts: `simulations/approximate_hits/README.md`
- Problem framing, decisions and results: `docs/approximate_cache_hits_context.md`
- Research question 1 (Aker and Proximity versus the other systems on shared workloads): `workloads/q1/README.md` and `harness/README.md`

## Layout

- `simulations/approximate_hits/`: the approximate-hit pipeline (greedy population, retention predictors, calibrated thresholds, online simulator, Python Aker and QVCache baselines). One CLI: `python -m simulations.approximate_hits <populate|eval|online|report> ...`.
- `utils/base/`: exact nearest neighbors (`knn`), vector file readers (`io`), logging. `utils/cache/`: the library behind the pipeline.
- `workloads/q1/`: Q1 data generation (datasets, universe of interpolated query groups, Zipf streams, ground truth, recall@10).
- `harness/`: Q1 generation pipeline (`make q1-generate`), pgvector backend and answer tables, runner, Aker and Proximity replay drivers (`harness/cpp`), deploy scripts.
- `third_party/aker/`: the vendored Aker C++ library with local patches (`PATCHES.md`).
- `datasets/`: the 50K ESCI embedding sample builder used by `populate esci`.
- `tests/`: unit tests, one folder per area (`approximate_hits`, `base`, `cache`, `workloads`, `harness`).
- `stale/`: retired exact-caching code and its tests (see `stale/README.md`).

## Getting started

```
python -m venv venv && venv/bin/pip install -r requirements.txt
make test
make help    # pipeline targets, including the q1-* targets
```
