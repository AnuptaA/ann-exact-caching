# Approximate Cache Hits with a Retention Target

Cache hits are allowed even when the cached answer is not exactly the full
search result, as long as the mean *retention* (fraction of the full search's
top-k returned by the cache) over cache hits stays at or above a user target
T. Problem framing, decisions, and results so far:
`docs/approximate_cache_hits_context.md`.

All numbers are best-case in one respect: caches are populated offline from
training queries whose exact neighbors are known.

## Pipeline

1. **Greedy population (oracle retention).** Candidate entries are the top-K
   balls of candidate-generating queries for several K. Greedy adds the entry
   with the largest (additional scoring-query hits) / (additional cached
   vectors) under the hits-only average-retention policy.
   - `greedy_population.py`: policy (`hits_from_histograms`, `online_hits`)
     and the incremental `GreedyPopulator`.
   - Drivers (write `results.json` and plots into `--out`):
     - `run_e2e_oracle_sift.py`: SIFT or ESCI (`--dataset`) with real queries.
     - `run_e2e_oracle_synthetic.py`: Aker-style simZipf workloads on SIFT.
     - `run_e2e_oracle_trace.py`: Aker request traces (SPACEV, TriviaQA on
       SPHERE) and the SPACEV1B real query log; time-ordered splits; saves
       exact neighbors to `topk.npz`.
2. **Admission without the oracle.**
   - `eval_certificates.py`: provable retention lower bounds (CIG, partial
     CIG, neighbor-distance profiles).
   - `eval_calibrated.py`: predictor + calibrated threshold (default: normal
     approximation with fixed-order testing, `threshold_selection.py`).
   - `eval_predictors.py`: compares predictors (ridge, gradient-boosted trees,
     ...) on fit / dev / cal / test splits. Needs scikit-learn (repo `.venv`).
   - `eval_calibration_methods.py`: validates threshold-selection procedures
     by repeated calibration draws.
   - `eval_radius_baselines.py`: LSH / global-mean radius estimation vs direct
     retention prediction, end to end (`.venv`).
3. **Summaries and figures.** `summarize*.py`, `plot_*.py`,
   `make_summary_figures.py` (writes `results/summary/`, including
   `fig4_aker.png` for the Aker traces).
   `run_eval_queue.py` runs an evaluation script on run directories as their
   greedy results appear.

## Data

- SIFT1M: `datasets/sift/raw_sift1m/` (TEXMEX corpus).
- ESCI: built with `datasets/esci/build_embedding_subsample.py` (`.venv`).
- Aker workloads and SPACEV1B query files: `datasets/aker/download.sh`.

Data and generated results are gitignored except `results/summary/`.

## Tests

```
python -m pytest tests/approximate_hits/
```
