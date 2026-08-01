# Codebase Overview

This is a research codebase for a vector-database caching technique with two
composable pipelines:

- **hit-policy**: given a pre-populated cache, is a new query certified as a
  cache hit under a geometric guarantee (CIG or HGG)?
- **population**: given a budget, which candidate cache entries should
  actually be materialized to maximize hit coverage?

Both pipelines are exercised against two real datasets (SIFT, ESCI) and one
synthetic, controllable workload.

## Architecture summary

```
utils/base/            core algorithms (dataset-agnostic, no I/O)
datasets/               raw-data loading + benchmark preparation (per dataset)
simulations/core/       shared simulation engine (CacheSimulator)
simulations/hit_policy/ hit-policy pipeline (run + analyze)
simulations/population/ population pipeline (propose -> select -> run + analyze)
simulations/analysis/   cross-cutting dataset characterization
simulations/debug/      containment-microbenchmark debug tooling
simulations/pdf/        shared PDF-report compiler used by every pipeline
simulations/synthetic/  synthetic-workload run + analyze (own generator)
tests/                  mirrors the simulations/ split, plus utils/dataloaders/generation
```

Both pipelines build on `simulations/core/simulate.py`'s `CacheSimulator`: the
hit-policy pipeline drives it directly against precomputed ground truth, and
the population pipeline (`population_runner.py`) hands it a `KVCache` built
from selected candidates, then drives the same `run_query` dispatch.

## `utils/base/` -- core algorithms

No I/O, no dataset knowledge -- pure geometry/algorithm code, reused by every
pipeline above it.

### Guarantees: CIG and HGG (`lemmas.py`, `lemmas_sigma.py`)

**CIG (Circular Inclusion Guarantee)**: for a cached entry `(Q, R_Q, r_Q)`, a
new query `q` and vector `v` in `R_Q` are certified correct if
`D(q, v) + D(q, Q) < r_Q` (triangle inequality argument). Costs one distance
computation per candidate vector, but can certify a hit using *any subset* of
a cache entry's vectors.

**HGG (Half-Gap Guarantee)**: if `D(q, Q) < half_gap` (half the gap between
the K-th and (K+1)-th true neighbor of `Q`), the entry's *entire* cached
top-K is guaranteed to equal `q`'s own top-K exactly. Zero per-vector distance
computation, but it's all-or-nothing on the full `K`.

**Canonical algorithms going forward** (see `docs/KNOWN_ISSUES.md` for why the
others are deprecated):

- `lemma1_no_union` (`lemmas.py`) -- CIG, single cache entry must alone supply
  the full `N` results (no accumulation across entries).
- `lemma2_no_union` (`lemmas.py`) -- HGG, same single-entry constraint.
- `lemma1_no_union_sigma` / `lemma2_no_union_sigma` (`lemmas_sigma.py`) --
  sigma-relaxed variants: certify a hit once `ceil(sigma * N)` (instead of the
  full `N`) is satisfiable from a single entry. `sigma=1.0` reproduces the
  corresponding exact function's behavior exactly (see the equivalence tests
  in `tests/utils/test_lemmas.py`). `CacheSimulator.run_query` always
  dispatches `"lemma1_no_union"`/`"lemma2_no_union"` through these sigma-aware
  versions (sigma defaults to `1.0`), so callers never need to choose between
  the sigma and non-sigma entry points directly.

**Deprecated (kept, not deleted, no longer dispatched)**: `lemma1_circular_inclusion`
and `lemma2_half_gap` (union variants -- accumulate results across multiple
cache entries), `combined_algorithm` (tries HGG then CIG per entry, unions
across entries), `lemma1_sigma` (the union-based sigma variant), and
`binary_search_last_index` (superseded by a direct `np.where` threshold mask
in the hot path). Their dispatch branches in `simulations/core/simulate.py`
and their dedicated test in `test_lemmas.py` are commented out, not removed.

### `MainMemory` / `range_search` (`main_memory.py`)

`MainMemory` wraps the "database": `M` vectors of dimension `d`, precomputes a
normalized matrix for a fast matmul-based `top_k_search`. Two code paths:
untracked (vectorized matmul) and tracked (Python loop, every call counted via
a `distance_tracker` -- used whenever the hit-policy/population pipelines need
exact distance-call counts for reporting). These two paths are cross-checked
in `tests/utils/test_main_memory.py::test_top_k_search_tracked_matches_untracked`.
Module-level `range_search(query, radius, base_vectors, metric)` is a fully
vectorized "vectors within radius" query, independently implemented but
cross-checked against `top_k_search` in the same test file.

### `KVCache` / `CacheEntry` (`kv_cache.py`)

`CacheEntry`: a cached query plus its top-K result vectors, `r_Q` (CIG
radius), `half_gap` (HGG threshold), optional `top_k_indices` (stable base
indices) and `top_k_matrix` (pre-normalized matrix enabling a single-matmul
CIG distance computation -- built by `CandidateEntry.to_cache_entry()` in the
population pipeline). `KVCache` holds a list of `CacheEntry` and supports
brute-force population from queries (`populate_from_queries`, used by the
hit-policy pipeline's on-the-fly mode and the synthetic pipeline) or direct
`add_entry` calls (used when materializing selected population candidates).

### `distance_metrics.py`

`euclidean_distance`, `cosine_distance`, `angular_distance` (the one actually
used throughout the real benchmarks -- `arccos(cosine_similarity) / pi`,
normalized to `[0, 1]`, satisfies the triangle inequality). `cosine_distance`
does *not* satisfy the triangle inequality and is not used by any research
benchmark; it exists for completeness only.

### `population.py` -- candidate proposal

Given observed cache queries and their true top-K ground truth, produces
`CandidateEntry` objects (a proposed cache radius `r_c` and the *complete* set
`R_c` of base vectors within that radius -- CIG soundness requires the
complete set, not just the original top-K; also carries `source_index`, the
observed query the candidate was proposed from, and `n`, the neighbor rank
its sphere was sized around). For full design/theory detail (the
triangle-inequality containment argument, the uniform K=0..K margin formula,
selection theory, and everything else population-specific), see
**`docs/POPULATION_DESIGN.md`**. Summary:

- `propose_candidates(alpha_values=...)` -- generic: `r_c = alpha * r_Q`.
- `propose_candidates_ball(n_max=..., K=...)` -- **the only nested-sphere
  proposal function**: `r_c` is sized to fully contain the n-th
  cache-neighbor's own K-ball, uniformly for every `K` including `0`
  (trivially a zero-radius ball). There is a single `K` parameter -- no
  separate `k_cache`/`k_test` distinction; `K_cache = K_test` always. A
  standalone `propose_candidates_point` function existed before Phase 3
  Round 12; it was exactly `propose_candidates_ball(K=0)` and was removed
  once that was fully confirmed.

### `selection.py` -- budget-constrained selection

`select_candidates(candidates, budget_B, strategy, ...)`. Four strategies:
`random`, `greedy_naive` (raw newly-covered-query count), `greedy_cost_static`
(coverage / candidate's own size, computed once), `greedy_cost` (coverage /
*actual marginal* new-vector cost, recomputed each iteration -- the strategy
used in practice; the other three exist for comparison/analysis). When
candidates have zero `R_c` overlap, `greedy_cost_static` and `greedy_cost`
provably select identically (see the equivalence test in `test_selection.py`).
`greedy_cost`'s marginal-cost recomputation has a bit-packed fast path
(`np.packbits`-based set-membership tracking, gated by
`_BITPACK_IDX_FRACTION`) that takes over from the plain fancy-index gather
once the number of already-selected indices grows large -- see
`docs/POPULATION_DESIGN.md` for the generation history.

## `datasets/` -- dataset preparation

SIFT (`.fvecs`/`.ivecs`, 1M-vector base, typically subsampled to 50K for
research benchmarks) and ESCI (Amazon product search, pre-embedded `.npy`/
`.npz`, full 1.82M-product catalog never subsampled for its hit-policy
pipeline) each have a `prepare_benchmark.py` that produces **two possible GT
schemas**:

- **"population" schema** (`cache_gt_population_K{K}.npz`) -- brute-force KNN
  against a *sampled* base (SIFT's subsample, or ESCI's fixed 50K sample).
  Used by the population pipeline for both datasets, and by SIFT's hit-policy
  pipeline (which already only ever works against a subsampled base).
- **"full"/"lean" schema** (`cache_gt_K{K}.npz`) -- vectors embedded directly
  in the file, computed against the *true full catalog*. Used by ESCI's
  hit-policy pipeline by default; available to SIFT only via an explicit
  `--full-base` prepare flag (off by default -- see the option's docstring in
  `datasets/sift/prepare_benchmark.py` for how it validates against
  `sift_groundtruth.ivecs`).

`datasets/dataloaders.py` provides the shared abstraction that lets one
`run_hit_policy.py` implementation serve both datasets despite the schema
difference: `load_cache_ground_truth_population`/`_full` and
`populate_cache_from_precomputed_population`/`_full`, wired together per
dataset via `run_hit_policy.py`'s `DATASET_CONFIG`.

## `simulations/` -- orchestration layer

- **`core/simulate.py`**: `CacheSimulator` (the shared engine both pipelines
  build on), `SimulationResult`/`QueryResult` dataclasses,
  `DistanceCalculationTracker`. `run_query(query_id, query, algorithm, ...)`
  dispatches by algorithm name; unknown algorithms and any exception raised
  during dispatch are caught and recorded on `QueryResult.error` rather than
  propagating (see `docs/KNOWN_ISSUES.md` for the implication of this for
  `hit_rate` accounting).
- **`hit_policy/`**: `run_hit_policy.py` (loads a prepared benchmark + GT,
  runs each requested algorithm against every test query, writes
  `summary.json`/`per_query.csv`/logs) and `analyze_hit_policy.py` (produces
  hit-rate/timing plots from those outputs).
- **`population/`**: `population_runner.py` (the shared engine: propose
  candidates, select under budget, materialize into a `KVCache`, simulate at
  each requested `sigma`, in both "oracle" (against the cache/observed
  queries) and "non_oracle" (against held-out test queries) modes),
  `run_population.py` (CLI for the ball-like proposal sweep, see
  `docs/POPULATION_DESIGN.md`),
  `analyze_population_proposal.py` (post-processing plots), `debug_*.py`
  (standalone read-only diagnostics against already-prepared benchmark data,
  no pipeline changes: `debug_ball_geometry.py`, `debug_trajectory.py`,
  `debug_sigma_sensitivity.py`, `debug_distance_concentration.py` (the
  latter is the only cross-dataset one, comparing SIFT and ESCI's background
  query-to-base/query-to-query angular distance concentration directly in
  one figure), `debug_vector_coverage.py` (sigma=1.0 only: compares vector
  coverage -- the fraction of test queries whose full true top-K neighbor
  set is physically cached, an upper bound independent of certification --
  against the actual CIG-certified test hit rate, sweeping real budget caps
  via `select_candidates` once per `run_population.py`'s `BUDGET_FRACTIONS`
  value against a single proposed candidate pool; not derived from a single
  unconstrained run like `debug_trajectory.py`'s budget_pct plot -- that
  approximation broke down whenever one oversized candidate's own `R_c`
  saturated most of the base in a single greedy pick), and `microbenchmarks/`
  (containment sweep only --
  `microbenchmark_containment.py`; the earlier candidate/K microbenchmark
  scripts exercised the generic alpha-scaling proposal path, unused by the
  production ball pipeline, and were removed along with their dedicated
  tests).
- **`analysis/`**: `characterize.py` and `query_clustering_ratio.py` --
  cross-workload dataset characterization, not specific to either pipeline.
- **`debug/`**: shared debug I/O helpers for the containment microbenchmarks.
- **`pdf/`**: `report.py`'s `PdfReportBuilder`/`render_sections` generic
  section-tree renderer, plus one thin compiler script per report type
  (`compile.py`, `compile_proposal.py`, `collect_results[_population].py`).
- **`synthetic/`**: a separate, fully controllable workload (own generator
  under `datasets/synthetic/`) exercising the hit-policy `CacheSimulator`
  directly -- useful for isolating perturbation-level/K/N effects that real
  datasets can't cleanly control for.

## Makefile targets (grouped)

| Group | Targets |
|---|---|
| Synthetic | `synthetic-generate`, `synthetic-run`, `synthetic-analyze` |
| SIFT hit-policy | `sift-prepare`, `sift-hit-policy-run`, `sift-hit-policy-analyze` |
| SIFT population | `sift-population-prepare` |
| ESCI hit-policy | `esci-prepare`, `esci-hit-policy-run`, `esci-hit-policy-analyze` |
| ESCI population | `esci-population-prepare` |
| Combined population | `microbenchmark` (containment microbenchmark, both datasets), `population` (microbenchmark + collected results) |
| Ball proposal sweeps | `{esci,sift}-population-ball[-exact]`, `population-proposal[-exact]`, `population-proposal-analyze` |
| Debug diagnostics | `debug-ball-geometry`, `debug-trajectory`, `debug-sigma-sensitivity`, `debug-distance-concentration`, `debug-population` (all four). `population-proposal-analyze` depends on `debug-distance-concentration` so its output is always current when the PDF is compiled. |
| Combined hit-policy analysis | `analyze-hit-policy` |
| Characterization | `dataset-characterize`, `full-pipeline` |
| Tests | `test`, `test-utils`, `test-synthetic`, `test-sift`, `test-esci`, `test-hit-policy`, `test-population`, `test-containment`, `test-proposal`, `test-dataloaders`, `test-generation`, `test-sigma` |
| Utilities/cleanup | `count-datasets`, `clean-cache`, `clobber[-synthetic\|-sift\|-esci]` |

`sift-prepare`/`esci-prepare` are shared dataset preparation used as a
prerequisite by both the hit-policy and population paths, not hit-policy
specific -- hence they aren't prefixed like the run/analyze targets.

## `tests/` directory map

| Directory | Covers |
|---|---|
| `tests/utils/` | Every module in `utils/base/`, plus `simulations/core/simulate.py`. Includes the fast-path/reference-path equivalence tests (tracked vs. untracked `top_k_search`, `range_search` vs. `top_k_search`, `_batch_distances`/`_cig_distances` fast vs. fallback, `greedy_cost` vs. `greedy_cost_static`). |
| `tests/dataloaders/` | `datasets/dataloaders.py`'s load/save/GT functions, one file per dataset. |
| `tests/generation/` | `datasets/{sift,esci}/prepare_benchmark.py`'s own logic (benchmark naming, GT construction, partial-resume) -- previously only indirectly covered via `tests/helpers.py`'s fixtures. |
| `tests/hit_policy/` | End-to-end `run_hit_policy.py` flow plus per-dataset edge cases. |
| `tests/population/` | End-to-end `population_runner.py`/`run_population.py`/`analyze_population_proposal.py`/`debug_trajectory.py`, plus `microbenchmarks/` for `microbenchmark_containment.py`. |
| `tests/synthetic/` | `datasets/synthetic/generator.py` (pytest-collected); `test_analysis.py`/`test_e2e_mini.py` are manual CLI-style smoke scripts (no `def test_*` functions, not pytest-collected -- run directly with `python`, not `pytest`). |

`tests/helpers.py` holds fixtures shared across every subdirectory:
`create_mini_benchmark[_with_population_data]` (dataset-agnostic mini SIFT/ESCI
benchmarks), `make_clustered_candidate_benchmark` (clustered synthetic data for
`utils/base/population.py`/`selection.py` tests), `check_selection_invariants`
(shared budget/monotonicity/coverage-range assertions for `select_candidates`
output), `get_temp_dir`/`cleanup_temp_dir`.

## Generated-output layout

Every pipeline follows the same convention:

```
simulations/<workload>/raw/<benchmark_or_timestamp>/
  summary.json          aggregate results per algorithm
  per_query.csv          one row per (algorithm, query)
  logs/                  per-algorithm hit/miss logs

simulations/<workload>/processed/<timestamp>/
  *.png                  plots produced by the workload's analyze script
```

`simulations/pdf/collect_results[_population].py` walk these `processed/`
trees across workloads and compile a combined report.
