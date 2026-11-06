# Vector Cache Research

Research code on caching the results of top-K vector search for vector
databases. The current direction is approximate cache hits with a retention
target: a cache hit is allowed when the mean fraction of the true top-K that
the cached answer returns stays at or above a user target. That work lives on
branch `jd` and is being merged into this layout.

## Layout

- `utils/base/`: distance metrics, brute-force main-memory search, cache entries, perturbation helpers.
- `datasets/`: data loading and benchmark preparation for SIFT and ESCI.
- `tests/`: unit tests for the code above.
- `stale/`: retired exact-caching code (see `stale/README.md`).

## Getting started

```
python -m venv venv && venv/bin/pip install -r requirements.txt
make test
make sift-prepare   # or esci-prepare
```
