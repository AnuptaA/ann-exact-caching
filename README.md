# Vector Cache Research

Research code on caching the results of top-K vector search for vector
databases. The current direction is approximate cache hits with a retention
target: a cache hit is allowed when the mean fraction of the true top-K that
the cached answer returns stays at or above a user target.

- Pipeline and scripts: `simulations/approximate_hits/README.md`
- Problem framing, decisions and results: `docs/approximate_cache_hits_context.md`

## Layout

- `simulations/approximate_hits/`: the approximate-hit pipeline (greedy population, retention predictors, calibrated thresholds, online simulator, Aker and QVCache baselines).
- `utils/base/`: distance metrics, brute-force main-memory search, cache entries, perturbation helpers.
- `datasets/`: data loading and benchmark preparation for SIFT, ESCI and the Aker workloads.
- `tests/`: unit tests (`tests/approximate_hits/` and `tests/utils`, `tests/dataloaders`, `tests/generation`).
- `stale/`: retired exact-caching code (see `stale/README.md`).

## Getting started

```
python -m venv venv && venv/bin/pip install -r requirements.txt
make test
```
