# Codebase Overview

Repo-wide map: what lives where, and how the pieces connect. Detailed
algorithmic design lives in `docs/POPULATION_DESIGN.md`; resolved gotchas
worth remembering live in `docs/KNOWN_ISSUES.md`.

## Two research threads

- **Hit policy**: given an already-populated cache, which policy (CIG, HGG,
  or the probabilistic overlap-threshold prototype) decides whether a query
  hits, and how well does each do.
- **Population**: given a query budget, which candidate cache entries should
  be materialized to maximize certified hit coverage under that budget.

Both share the same certification core in `utils/base/` and the same
simulator (`simulations/core/simulate.py`).

## `utils/base/` (core library)

Everything here is importable as `utils.base.<name>`, re-exported from
`utils/base/__init__.py`:

- `distance_metrics.py`: `euclidean_distance`, `cosine_distance`,
  `angular_distance` (scalar, single pair), `pairwise_angular_distance`
  (matrix, pre-normalized vectors), `get_distance_function`.
- `main_memory.py`: `MainMemory` (brute-force top-k over the full base set,
  used as ground truth and by the on-the-fly hit-policy path), `range_search`.
- `kv_cache.py`: `KVCache` / `CacheEntry`, the cache data structure both
  pipelines populate and query.
- `lemmas.py`: the exact certification algorithms `lemma1_circular_inclusion`
  (CIG, union), `lemma2_half_gap` (HGG, union), `combined_algorithm`, plus
  `lemma1_no_union` / `lemma2_no_union` (single-entry variants). The union
  variants and `combined_algorithm` are unreachable through
  `CacheSimulator.run_query` (their dispatch branches were removed); they
  remain importable for direct use and are still exported.
- `lemmas_sigma.py`: sigma-relaxed no-union variants,
  `lemma1_no_union_sigma` / `lemma2_no_union_sigma` (threshold
  `ceil(sigma * N)` instead of the full `N`; `sigma=1.0` is behaviorally
  identical to the plain no-union lemmas), plus `fit_r_t_hat` and
  `overlap_threshold_hit_policy` (the probabilistic, uncertified hit-policy
  prototype; see `docs/POPULATION_DESIGN.md` section 9). `CacheSimulator`
  always routes `"lemma1_no_union"` / `"lemma2_no_union"` through these sigma
  variants, never through the plain versions in `lemmas.py`.
- `population.py`: `CandidateEntry`, `propose_candidates` (generic
  alpha-scaling proposal), `propose_candidates_ball` (the nested-sphere
  family used in production, all K including K=0), `precompute_ball_invariants`.
- `selection.py`: `select_candidates`, dispatching to 4 strategies
  (`random`, `greedy_naive`, `greedy_cost_static`, `greedy_cost`); production
  uses `greedy_cost`.
- `perturbation.py`: synthetic-query perturbation helpers for generating
  test queries at controlled angular distance from a cache query.

## `datasets/`

- `dataloaders.py`: shared load/save functions for both SIFT and ESCI
  benchmark formats, plus the population-schema ground-truth loaders
  (`load_cache_ground_truth_population` / `_full`, sampled vs. full base;
  both are genuinely used by different pipelines, not redundant).
- `sift/prepare_benchmark.py`, `esci/prepare_benchmark.py`: dataset-specific
  benchmark preparation (splitting, ground-truth precomputation).
- `synthetic/generator.py`: synthetic dataset generation for the hit-policy
  synthetic workload.

## `simulations/`

### `simulations/core/`
- `simulate.py`: `CacheSimulator`, the shared engine every pipeline runs
  queries through. Dispatches `"brute"`, `"lemma1_no_union"`,
  `"lemma2_no_union"`, `"overlap_threshold"`.
- `analysis_helpers.py`: shared plotting/labeling helpers (`ALGO_LABELS`,
  `lbl`, `save_fig`, `pct_fmt`, `load_summaries`, etc.) used by both
  `hit_policy/analyze_hit_policy.py` and `synthetic/analyze_synthetic.py`.

### `simulations/hit_policy/`
- `run_hit_policy.py` / `analyze_hit_policy.py`: run and analyze the CIG/HGG
  hit-policy sweep over the SIFT/ESCI benchmark matrix.
- `run_overlap_threshold_sweep.py` / `analyze_overlap_threshold.py`: run and
  analyze the overlap-threshold prototype (see `POPULATION_DESIGN.md`
  section 9) across the same matrix plus a threshold/percentile grid. Output
  lands under `simulations/hit_policy/reports/overlap_threshold/<run_id>/`
  (gitignored).

### `simulations/population/`
- `population_runner.py`: the shared propose/select/build/simulate engine,
  analogous to `core/simulate.py` for the population thread.
- `run_population.py`: CLI driver for the production ball-proposal sweep.
- `analyze_population_proposal.py`: analyzes `run_population.py`'s output.
- `debug_common.py`: shared setup (benchmark loading, candidate proposal,
  plotting helpers) reused by the debug scripts below.
- `debug_ball_geometry.py`, `debug_sigma_sensitivity.py`,
  `debug_distance_concentration.py`, `debug_distance_ratio.py`: standalone
  diagnostics against already-prepared benchmark data, no changes to the
  production pipeline. `debug_distance_ratio.py` is a one-off (no Makefile
  target, no test); the others have both.
- `debug_intersection_calibration.py`: the intersection-fraction calibration
  study (see `POPULATION_DESIGN.md` section 9), the largest and most
  substantial of the debug scripts.

### `simulations/population/microbenchmarks/`
- `microbenchmark_containment.py`: sphere-aware candidate proposal
  containment microbenchmark, run per dataset via the Makefile's
  `microbenchmark` target.
- `containment_debug.py`, `debug_utils.py`: shared helpers for
  `microbenchmark_containment.py`'s `--debug` diagnostic mode.

### `simulations/pdf/`
- `report.py`: `PdfReportBuilder` and shared rendering primitives every
  other file here builds on.
- `compile.py`, `collect_results.py`, `collect_results_population.py`,
  `compile_proposal.py`, `collect_intersection_fraction.py`,
  `collect_overlap_threshold.py`: one thin compiler/collector per report
  type, each gathering a specific upstream producer's output into a PDF.

### `simulations/analysis/`
- `query_clustering_ratio.py`: cross-dataset query-clustering
  characterization (`d(q)/r(q)` ratio), independent of either pipeline.

### `simulations/synthetic/`
- `run_synthetic.py` / `analyze_synthetic.py`: run and analyze the
  synthetic-workload hit-policy sweep.

### `simulations/debug/`
Output-only. Contains no source files; every subdirectory here
(`population_geometry/`, `population_sigma_sensitivity/`,
`population_distance_concentration/`, `population_distance_ratio/`,
`population_intersection_calibration/`, plus timestamped run dirs) is
generated output from the debug scripts above, entirely gitignored. Don't
expect to find source code here; it moved to
`simulations/population/microbenchmarks/` for the containment-debug helpers
that used to sit in this directory.

### Other generated-output directories (gitignored, no source)
`simulations/{sift,esci,synthetic}/{raw,processed}/`,
`simulations/results/{characterization,intersection_fraction,overlap_threshold,population}/`,
`simulations/characterization/`, `simulations/population_proposal_sigma/`,
`simulations/hit_policy/reports/`, `simulations/population/reports/`,
`simulations/population/profiles/`.

## `tests/`

Mirrors `simulations/`'s structure: `tests/utils/` (core library),
`tests/population/` (+ `microbenchmarks/` subdir), `tests/hit_policy/`,
`tests/synthetic/`, `tests/dataloaders/`, `tests/generation/`,
`tests/intersection_fraction/` (the intersection-fraction calibration's
independent validation, converted from a standalone script). `tests/helpers.py`
holds shared fixtures (e.g. `make_clustered_candidate_benchmark`).

## Makefile targets (by section)

- **Synthetic**: `synthetic-generate`, `synthetic-run`, `synthetic-analyze`
- **SIFT hit-policy**: `sift-prepare`, `sift-hit-policy-run`, `sift-hit-policy-analyze`
- **SIFT population**: `sift-population-prepare`
- **ESCI hit-policy**: `esci-prepare`, `esci-hit-policy-run`, `esci-hit-policy-analyze`
- **ESCI population**: `esci-population-prepare`
- **Microbenchmark / combined population**: `microbenchmark`, `population`
- **Ball proposal sweeps**: `esci-population-ball`, `sift-population-ball`,
  `population-proposal-analyze`, `population-proposal`,
  `esci-population-ball-exact`, `sift-population-ball-exact`,
  `population-proposal-exact`
- **Population debug diagnostics**: `debug-ball-geometry`,
  `debug-sigma-sensitivity`, `debug-distance-concentration`, `debug-population`
- **Combined hit-policy analysis**: `analyze-hit-policy`
- **Intersection-fraction / overlap-threshold PDF collection**:
  `collect-intersection-fraction`, `collect-overlap-threshold`
- **Characterization**: `dataset-characterize`, `full-pipeline`
- **Tests**: `test`, `test-utils`, `test-synthetic`, `test-sift`, `test-esci`,
  `test-hit-policy`, `test-population`, `test-containment`, `test-proposal`,
  `test-dataloaders`, `test-generation`, `test-sigma`
- **Utilities/cleanup**: `count-datasets`, `clean-cache`, `clobber`,
  `clobber-synthetic`, `clobber-sift`, `clobber-esci`

Note: there is no Makefile target for running `run_overlap_threshold_sweep.py`,
`analyze_overlap_threshold.py`, or `debug_intersection_calibration.py`
directly, only their downstream PDF collectors
(`collect-intersection-fraction`, `collect-overlap-threshold`). Run those
scripts directly when iterating on that work.

## Generated-output layout convention

Every run-producing script writes to a timestamped subdirectory
(`<output_root>/<run_id>/...`), never overwriting a previous run. PDF
collectors default to the latest run directory unless told otherwise
(`simulations/pdf/report.py::latest_run_dir`).
