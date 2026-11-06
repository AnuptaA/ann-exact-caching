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
   - `eval_aker_thresholds.py`: replays Aker's density-aware per-entry
     threshold rule on our offline caches (trace runs).
   - `eval_threshold_transfer.py`: does a calibrated threshold transfer across
     caches (size / target / perturbation)? Needs `.venv`.
   - Current model (7 inputs: relative offset and relative size for the 3
     best-fitting cache balls, plus the intruder count from a shell sample of
     uncached vectors just outside each ball), all `.venv`:
     - `eval_core.py`: oracle vs ball stats / shell sample, with knobs (number
       of balls, sample fraction, shell radius, bits per dimension).
     - `eval_simple.py`: GBT vs linear / logistic / no-fitting scores.
     - `eval_shell_ablation.py`: which shell features matter; uncalibrated
       model quality.
     - `eval_generalize.py`: transfer across cache states and across k.
   - Exploratory (radius prediction, training focus, gap decomposition,
     samples, older feature sets): `eval_radius_scores.py`,
     `eval_radius_models.py`, `eval_focused.py`, `eval_gap.py`,
     `eval_ablation.py`, `eval_features_v2.py`.
   - Final model at scale: `eval_final.py` (every checkpoint, memory-fair
     baseline, calibration-size sweep), `eval_shell_size.py` (larger shells,
     calibration-size study), `eval_shell_budget.py` (cache / shell split of a
     memory budget chosen on dev).
4. **Online simulation.** `online_prepare.py` (ground truth and query
   streams), `online_sim.py` (static / periodic / triggered / LRU, warm-up,
   audits, recalibration, drift alarms, shadow rebuilds), `online_baselines.py`
   (Aker, QVCache), `online_summary.py`.
3. **Summaries and figures.** `summarize*.py`, `plot_*.py`,
   `make_summary_figures.py` (writes `results/summary/`, including
   `fig4_aker.png` for the Aker traces), `plot_feature_diagram.py` (2-D
   illustration of the 7 model inputs, `results/summary/feature_diagram.png`).
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
