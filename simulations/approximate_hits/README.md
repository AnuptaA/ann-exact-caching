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
`simulations/approximate_hits/` holds thin drivers that write JSON and figures
into `results/`. Run drivers from the repo root with `PYTHONPATH=.` (the
Makefile does this); `make help` lists the targets.

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
   vectors). Drivers (write `results.json` and plots into `--out`):
   - `run_e2e_oracle_sift.py`: SIFT or ESCI (`--dataset`) with real queries.
   - `run_e2e_oracle_synthetic.py`: Aker-style simZipf workloads on SIFT.
   - `run_e2e_oracle_trace.py`: Aker request traces and the SPACEV1B query log.
2. **Admission without the oracle.**
   - Current model (7 inputs): `eval_core.py` (knobs), `eval_simple.py`
     (model comparison), `eval_shell_ablation.py`, `eval_generalize.py`
     (across caches and k), `eval_final.py` (every checkpoint, memory-fair
     baseline), `eval_shell_size.py`, `eval_shell_budget.py`.
   - Earlier 12/20-feature model: `eval_calibrated.py`, `eval_predictors.py`
     (their JSON feeds `make_summary_figures.py`), `eval_threshold_transfer.py`,
     `eval_calibration_methods.py`, `eval_certificates.py` (provable lower
     bounds), `eval_aker_thresholds.py` (Aker's per-entry rule offline).
   - Removed exploratory scripts are in git tag `archive/pre-modularization`.
3. **Online simulation.** `online_prepare.py` (ground truth and streams),
   `online_sim.py` (static / periodic / triggered / LRU plus the `aker` and
   `qvcache` baselines), `online_summary.py`.
4. **Summaries and figures.** `summarize*.py`, `plot_*.py`,
   `make_summary_figures.py` (writes `results/summary/`),
   `plot_feature_diagram.py`.

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
