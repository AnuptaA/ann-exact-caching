# stale/

Old leftover from the exact-caching phase of this project (CIG/HGG hit
policies, cache population and selection, synthetic workload, PDF reports).
It is no longer maintained or compared against. It is self-contained: run
anything from inside this directory, for example `cd stale && python -m pytest tests -q`.

The docs under `stale/docs/` predate the final deletions and still mention a few
removed files (overlap-threshold sweep, intersection-fraction calibration,
`debug_distance_ratio.py`). They are in git tag `archive/main-pre-cleanup`.
