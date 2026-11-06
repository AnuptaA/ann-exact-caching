# Approximate Cache Hits with a Retention Target

Cache hits are allowed even when the cached answer is not exactly the full
search result, as long as the mean *retention* (fraction of the full search's
top-k returned by the cache) over cache hits stays at or above a user target
T. Problem framing, decisions, and results so far:
`docs/approximate_cache_hits_context.md`.

All offline numbers are best-case in one respect: caches are populated from
training queries whose exact neighbors are known.

## Layout

Reusable logic is in `utils/cache/` (no command-line code, no global state);
`simulations/approximate_hits/` is one command-line program that writes JSON
and figures into `results/`:

    PYTHONPATH=. python -m simulations.approximate_hits <group> <command> ...

Groups: `populate`, `eval`, `online`, `report`. Every command has `--help`.
`--sift-dir` and `--aker-dir` (before the group) point at other data
directories. The Makefile wraps the common commands; `make help` lists them.

| module (`utils/`) | contents |
|---|---|
| `base/knn.py` | squared and pairwise distances, exact top-k (blocked for large databases), `knn_dists` |
| `cache/workloads.py` | `RunData`, `load_run_data`; SIFT/ESCI, simZipf and Aker-trace loaders |
| `cache/population.py` | hits policy, `GreedyPopulator`, `greedy_entries` |
| `cache/entries.py` | entry bookkeeping, `build_entry_tables`, `state_tables` |
| `cache/features.py`, `shell.py` | the 7-feature inputs: best-fitting balls and the shell-sample intruder count |
| `cache/cache.py` | online cache state (ref-counted entries, shell sample, features) |
| `cache/models.py` | logistic, ridge and gradient-boosted models |
| `cache/calibration.py` | threshold selection with a retention guarantee, Hoeffding bound |
| `cache/baselines.py` | Aker and QVCache baselines for the online simulator |
| `cache/metrics.py`, `parallel.py` | hit-rate helpers; sharing read-only inputs with forked workers |
| `cache/legacy_features.py` | the earlier 12 + 8 features behind `calibrated.json` / `predictors.json` |

## Pipeline

1. **Greedy population (oracle retention).** Candidate entries are the top-K
   balls of candidate-generating queries for several K; greedy adds the entry
   with the largest (additional scoring-query hits) / (additional cached
   vectors). Commands (write `results.json` and plots into `--out`):
   - `populate sift`, `populate esci`: real queries on SIFT or ESCI.
   - `populate synthetic`: Aker-style simZipf workloads on SIFT.
   - `populate trace`: Aker request traces and the SPACEV1B query log.
2. **Admission without the oracle.**
   - Current model (7 inputs): `eval core` (knobs), `eval simple`
     (model comparison), `eval shell-ablation`, `eval generalize`
     (across caches and k), `eval final` (every checkpoint, memory-fair
     baseline), `eval shell-size`, `eval shell-budget`.
   - Earlier 12/20-feature model: `eval calibrated`, `eval predictors`
     (their JSON feeds `report figures`), `eval transfer`,
     `eval calibration`, `eval certificates` (provable lower
     bounds), `eval aker` (Aker's per-entry rule offline).
   - Removed exploratory scripts are in git tag `archive/pre-modularization`.
3. **Online simulation.** `online prepare` (ground truth and streams),
   `online run` (static / periodic / triggered / LRU plus the `aker` and
   `qvcache` baselines), `online summary`.
4. **Summaries and figures.** `report` commands: `figures` (writes
   `results/summary/`), `calibrated`, `certificates`, `master`, `trends`,
   `feature-diagram`, and the tables `tables`, `aker`, `calibration`,
   `predictors`, `transfer`. `--results-dir` redirects the results folder.

## Data

- SIFT1M: `datasets/sift/raw_sift1m/` (TEXMEX corpus).
- ESCI: built with `datasets/esci/build_embedding_subsample.py`.
- Aker workloads and SPACEV1B query files: expected under `datasets/aker/`
  (the download script referenced in earlier notes is not in the repo).

Data and generated results are gitignored except `results/summary/`.

## Tests

```
make test            # or: python -m pytest tests/
```
