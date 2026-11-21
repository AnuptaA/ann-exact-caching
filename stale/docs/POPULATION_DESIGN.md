# Population Pipeline: Design and Theory

Deep-dive reference for the population side of the codebase: given a query
budget, which candidate cache entries should be materialized to maximize
certified hit coverage. Complements `docs/CODEBASE_OVERVIEW.md` (repo-wide
map) and `docs/KNOWN_ISSUES.md` (open items).

## 1. Pipeline overview

Four stages, always in this order, implemented once in
`simulations/population/population_runner.py::run_population_benchmark`:

```
propose_candidates_*  ->  select_candidates  ->  KVCache.add_entry (xN)  ->  CacheSimulator.run_query
   (population.py)         (selection.py)         (kv_cache.py)              (simulate.py, via lemmas_sigma.py)
```

1. **Propose**: given the cache/train queries and their true top-K ground
   truth, produce a pool of `CandidateEntry` objects, each a candidate cache
   center (a query) plus a radius `r_c` and the complete set `R_c` of base
   vectors within that radius.
2. **Select**: given a memory budget `B` (in base-vector count) and a
   strategy, greedily choose a subset of candidates that fits `B` while
   maximizing certified query coverage.
3. **Build**: materialize the selected candidates into a `KVCache`.
4. **Simulate**: run `CacheSimulator.run_query` for every query in two modes
   (oracle against the cache's own queries, non_oracle against held-out test
   queries), at every requested sigma, recording hit rate and recall.

## 2. Shared certification theory (CIG, HGG, sigma relaxation)

This theory is shared with the hit-policy pipeline, but population's
proposal and selection stages are built entirely around producing candidates
that satisfy it.

### CIG (Circular Inclusion Guarantee)

For a cached entry `(Q, R_Q, r_Q)` and a new query `q`, a base vector `v` is
certified correct without touching the database if:

```
D(q, v) <= r_Q - D(Q, q)
```

By the triangle inequality, `D(Q, v) <= D(Q, q) + D(q, v)`. If the condition
holds, `D(Q, v) <= r_Q`, and since `r_Q` is the distance from `Q` to its K-th
(farthest) cached neighbor, anything within `r_Q` of `Q` is already in `R_Q`
(or closer). Implementation
(`utils/base/lemmas.py::lemma1_no_union`): an O1 gate checks `d_Q <= r_Q`
first (distances are non-negative, so no vector can satisfy the inequality
otherwise), then a direct threshold mask,
`passing = np.where(distances_to_q <= r_Q - d_Q)[0]`, replaces the older
sorted-plus-binary-search approach (`binary_search_last_index`, kept for
backward compatibility but no longer on the hot path).

### HGG (Half-Gap Guarantee)

If `D(Q, q) < half_gap` (half the gap between the K-th and (K+1)-th true
neighbor of `Q`), the entry's entire cached top-K is guaranteed to equal
`q`'s own top-K exactly (`utils/base/lemmas.py::lemma2_no_union`). Any base
vector outside `Q`'s true top-K is at least `2*half_gap` farther from `Q`
than the K-th vector; a query move of less than `half_gap` can't perturb
distances enough to let an outside vector become closer to `q` than an
inside one was to `Q`.

### No-union vs. union variants

Both guarantees above are single-cache-entry ("no-union") checks: one entry
alone must supply the full `N` results. Union variants
(`lemma1_circular_inclusion`, `lemma2_half_gap`, `combined_algorithm`)
accumulate results across multiple entries and remain importable, but
`CacheSimulator.run_query` only dispatches `"brute"`, `"lemma1_no_union"`,
`"lemma2_no_union"`, and `"overlap_threshold"`; any other algorithm string
raises `ValueError`.

### Sigma-relaxed no-union variants

`utils/base/lemmas_sigma.py` relaxes the certification threshold from the
full `N` to `ceil(sigma * N)`. `sigma=1.0` is behaviorally identical to the
plain no-union functions. `CacheSimulator.run_query` always routes
`"lemma1_no_union"` / `"lemma2_no_union"` through the sigma-aware versions,
never the plain `lemmas.py` functions, so a sigma sweep needs no branching
in the simulate call itself: it's the same code path at every sigma value.

### Containment mode (K=0)

`threshold = ceil(sigma * N)` is `0` at `N=0`. The existing structure
(O1 gate, then `len(passing) >= threshold`) already reduces correctly to
"hit iff contained" once `N=0` reaches it. The real fix was letting `N=0`
reach the check at all: `population_runner.py`'s simulate-K used to clamp to
`max(K, 1)` because a mean-recall accumulation divides by `K`. The fix is a
guard (`if K > 0: recall_sum += ...`), not a clamp, so `mean_recall` stays at
its default (0.0, meaning "not computed") for K=0 runs instead of dividing by
zero.

## 3. Candidate proposal (`utils/base/population.py`)

### `CandidateEntry` and the completeness requirement on `R_c`

CIG soundness requires the stored set to be the *complete* set of base
vectors within `r_c`, not just the candidate's nominal top-K. A candidate's
cached content must be everything within its radius, or the downstream
distance-threshold check is working against an incomplete set.

`to_cache_entry()` builds an index-only `CacheEntry` (`top_k_vectors = []`,
only `top_k_indices` set): candidates' `R_c` sets overlap heavily by
construction (nested spheres, below), so a private per-entry vector copy
would duplicate the same base vectors across many entries. Distance
computation instead goes through a shared `base_hat`/`base_vectors` matrix
passed at hit-check time. `R_c`'s indices are stored as a compact `int32`
array rather than a Python list of boxed ints for the same reason at scale.

### Generic alpha-scaling proposal

`propose_candidates(alpha_values=...)`: `r_c = alpha * r_Q`, the simplest
scheme. Coverage is checked post hoc via a CIG mask rather than constructed
to guarantee it. Still supported and tested, but not used by the production
sweep.

### The nested-sphere family: `propose_candidates_ball` (all K, including K=0)

Delegates to one shared engine, `_propose_nested_spheres`, using one uniform
formula for every K:

```
r_c(i, n) = D(query_i, neighbor_n) + margin_lookup[neighbor_n]
```

where `neighbor_n` is cache query `i`'s n-th nearest other cache-query
neighbor (`n` ranges `1..n_max`, so `n_max` nested spheres are proposed per
query). `margin_lookup[neighbor_n]` is the distance from `neighbor_n` to its
own K-th nearest base vector (its own K-ball radius), computed via
`_precompute_kth_base_dist`; at K=0 this is trivially zero, not a special
case, just the natural value of the same formula.

**Soundness (K>=1)**: the sphere of radius `r_c` centered at `Q = query_i`
is sized to fully contain `neighbor_n`'s own K-ball. For any point `x` with
`D(neighbor_n, x) <= margin_lookup[neighbor_n]`:

```
D(Q, x) <= D(Q, neighbor_n) + D(neighbor_n, x) <= D(Q, neighbor_n) + margin_lookup[neighbor_n] = r_c
```

**At K=0**: `r_c(i,n) = D(query_i, neighbor_n)` exactly, the sphere reaches
`neighbor_n` and no farther. The CIG check itself is sound for any `r_c`
regardless of construction; what changes at K=0 is that `train_r_t`/
`test_r_t` are also zero (see below), so the coverage condition collapses to
pure distance containment with no per-query margin.

### `precompute_ball_invariants`: caching K-independent work across a K-sweep

`_propose_nested_spheres` accepts an optional `invariants` parameter, filled
by `precompute_ball_invariants(...)` once and reused across every K in a
sweep. Every observed query's distance to every base vector/train
query/test query, and the cache queries' pairwise nearest-neighbor ordering,
are K-independent (only `margin_lookup` and the K-specific `train_r_t`/
`test_r_t` depend on K). The largest held array (`n_cache x n_base`) is
about 410MB in float32 at the largest real scale in this codebase (c2048,
n_base=50000), held only for the duration of the caller's K-sweep and only
by callers that opt in.

`precompute_ball_invariants` also introduces `_raw_pairwise_dists`, the
single canonical full-`(n,n)`-matrix helper both `_pairwise_cache_dists`
(sorted, diagonal masked) and `precompute_ball_invariants` derive from,
avoiding two independently-computed matrices that could disagree by a few
ULPs at a certification boundary (see `docs/KNOWN_ISSUES.md`).

### Containment mode: `train_r_t`/`test_r_t` are zero

`population_runner.py`'s `k_test_override == 0` branch zeroes both
`obs_r_Q`/`train_r_t` and `test_r_t`. `cache_load_K` stays 1 since cache
entries still need real top-1 ground truth for entry construction; that's
unrelated to `obs_r_Q`, which has no functional role beyond an informational
field in `_write_selection_log`'s debug output.

### `_precompute_kth_base_dist`'s `k >= 1` guard

`np.partition(d_batch, k-1, axis=1)[:, k-1]` at `k=0` becomes
`partition(..., -1, ...)`, which extracts the maximum distance, not zero or
any meaningful quantity. Guarded with an explicit assert rather than left to
fail silently; K=0 always routes through the `margin_lookup = zeros(...)`
branch instead.

### No dominance pruning at proposal time

All `n_max` nested spheres per query are proposed and kept (unless a
sphere's `covered_train_indices` is empty), even though a smaller sphere for
the same query is strictly dominated once a larger sphere for that query
exists in the pool. A strictly-dominated smaller sphere can never win
`select_candidates`'s newly-covered-per-marginal-cost ratio over the larger
sphere it's a strict subset of, so pruning is left implicit to selection
rather than done explicitly at proposal time. This keeps proposal a pure
"generate everything geometrically valid" stage.

## 4. Selection (`utils/base/selection.py`)

`select_candidates(candidates, budget_B, strategy, n_train_queries,
n_test_queries, seed=42) -> (selected, trajectory)`.

Budget `B` is over base-vector cache footprint (`rc_size`/`R_c`), not
candidate count. `_fits_budget` short-circuits via `rc_size` and only
computes the true marginal `R_c & ~cached_mask` when the cheap check can't
decide. A train (or test) query is "covered" once any selected candidate's
`covered_train_indices` for it is True; selection maintains a running OR of
all selected candidates' coverage masks. One trajectory entry is appended
per accepted candidate, letting downstream analysis plot coverage/hit-rate
against memory spent incrementally.

### The four strategies

1. **`random`**: fixed-seed permutation, single pass, accept iff fits
   budget.
2. **`greedy_naive`**: repeatedly picks the candidate with the most
   newly-covered train queries, ignoring cost entirely.
3. **`greedy_cost_static`**: scores `newly / static_cost`, where
   `static_cost = rc_size` is the candidate's fixed total ball size, not its
   marginal cost against the current cache.
4. **`greedy_cost`** (production, `STRATEGIES = ['greedy_cost']`): scores
   `newly / marginal`, where `marginal` is the true incremental cost
   recomputed as selection proceeds, classic weighted-set-cover-style
   greedy.

`greedy_cost_static` and `greedy_cost` provably select identically when
candidates have zero `R_c` overlap, since static and marginal cost coincide
in that case. `greedy_cost` defers zero-marginal-cost, positive-benefit
candidates into a separate list, flushed at the end sorted by benefit
against the final coverage mask, since they're free once anything covering
them is already cached.

### `greedy_cost`'s incremental delta-update

The current implementation represents `cached_mask`/`covered_train_mask`/
`covered_test_mask` as numpy bool arrays (`CandidateEntry.R_c` itself a bool
array with a cached `rc_size`), and rather than recomputing `marginal`/
`newly` for all remaining active candidates from scratch after each pick, it
computes exactly the bit positions the picked candidate newly contributed,
then subtracts, per remaining candidate, how many of those specific bits it
also has:

```python
new_base_idx = np.flatnonzero(best.R_c & ~cached_mask)
marginal = marginal - _chunked_delta_counts(active, 'R_c', new_base_idx)
```

This moves the dominant per-iteration cost from `O(|active| * n_base)` to
`O(|active| * k)` where `k` is the number of newly-set bits that iteration,
bounding total work over the whole run. `_chunked_delta_counts` processes
`active` in fixed batches (`_DELTA_CHUNK_SIZE = 6000`) to bound peak
transient memory, and has two code paths: a fancy-index gather (scales with
the number of newly-set bits) and a bit-packed popcount path (`np.packbits`
plus a 256-entry byte popcount table, roughly fixed cost per chunk). The two
cross over empirically around `idx ~= attr_len/20`; both compute the exact
same integer count, verified via `np.array_equal` in
`tests/utils/test_selection.py`. This is a numerically exact optimization,
not an approximation: `test_vectorized_matches_naive_reference`,
`test_greedy_cost_incremental_matches_reference_at_high_budget`, and
`test_greedy_cost_chunked_delta_matches_reference_across_chunk_boundary`
guard exact equivalence to full recomputation, including at chunk
boundaries.

## 5. `KVCache` internals (`utils/base/kv_cache.py`)

`CacheEntry` fields: `query` (Q), `top_k_vectors` (can be `[]` for
index-only entries), `r_Q` (CIG radius), `half_gap` (HGG threshold),
`top_k_indices` (stable base-vector indices), `top_k_matrix` (pre-stacked,
pre-normalized, enabling a single-matmul CIG distance computation, built by
`CandidateEntry.to_cache_entry()`, `None` when created via `add_entry`
directly).

Two ways entries get built: `populate_from_queries` (brute-force, used by
hit-policy's on-the-fly mode) and `add_entry` directly (the low-level
primitive population uses via `to_cache_entry()`, also used by `simulate.py`
for index-only entries with an empty `top_k_vectors` list).

`get_d_Q_batch(query)` and `get_entry_radii()` share one lazy build
(`build_query_matrix()`), invalidated together on `add_entry`, letting
`CacheSimulator.run_query` vectorize the CIG O1 gate across all cache
entries in one NumPy comparison rather than a per-entry Python loop.

## 6. Pipeline orchestration (`population_runner.py`)

Inside `run_population_benchmark()`: load benchmark and resolve K once, load
ground truth once, propose candidates once total per benchmark (candidates
don't depend on budget), then for each `(budget_B, strategy)` pair: select,
build a fresh `KVCache`, and for each sigma in the sweep, simulate oracle
then non_oracle.

**Oracle vs. non_oracle**: the only difference is the evaluation query set.
Oracle uses the same queries the cache was populated from (an upper-bound
sanity check); non_oracle uses the held-out test set (the actual metric of
interest for generalization). `analyze_population_proposal.py` only reads
non_oracle summaries for plotting; oracle results are written but not
consumed by analysis.

**Sigma sweep cost**: `select_candidates` and cache build run once per
`(budget, strategy)` pair, before the sigma loop; only the simulate stage
repeats per sigma. A 4-value sigma sweep takes roughly 2.9x the wall time of
a single-sigma run for the same config.

**Output nesting**: flat layout (`.../B{budget}/oracle/{strategy}/...`) when
only the default `sigma=1.0` is used; nests one level deeper
(`.../sigma_{value:.2f}/...`) for any other sweep. Trajectory/selection-log
files are sigma-independent and stay directly under the budget directory.

**Budget semantics**: `run_population.py::_budget_values` computes budgets
as fractions of `n_base` (the benchmark's base-vector corpus size), not the
cache/train query count.

## 7. CLI driver (`run_population.py`)

`--dataset` (required), `--n-max` (default 20), `--k-test` (default
`[0, 1, 5, 10]`), `--benchmark` (default: all research benchmarks),
`--strategies` (default `['greedy_cost']`), `--sigma` (default `[1.0]`),
`--workers` (default 1), `--seed` (default 42). `K_cache = K_test = K`
always, including K=0, with no separate clamp: `K = k_test` directly, no
special-casing needed once population's zero-margin-at-K=0 behavior above
was in place (regression-tested by
`test_e2e_ball_runner_k_test_zero_containment`).

`BUDGET_FRACTIONS` covers 15 values with finer resolution in the 0.45-0.98
range, where both datasets' hit-rate curves are most sensitive to budget.
Cache-size sweep is encoded in the benchmark names themselves (c256/c512/
c1024/c2048, 4 sizes x 2 datasets = 8 benchmarks total). One work item per
`(benchmark, k_test)` pair; `--workers > 1` uses
`concurrent.futures.ProcessPoolExecutor`, all workers sharing one run
timestamp.

## 8. Analysis (`analyze_population_proposal.py`)

`n_max` is fixed at 20 in every real run, so the analysis grid is `sigma x
K_test` (rows=sigma, columns=K_test), one combined file per metric per
dataset:

- `ball/memory_vs_hitrate.png`: hit rate vs. memory utilization, train and
  test overlaid per cell, one line per cache size.
- `ball/memory_vs_candidates_cached.png`: same grid, y=candidate count. Uses
  a shared y-axis across the global max rather than per-subplot autoscaling.
- `ball/recall_vs_sigma.png`: rows=K_test, columns=budget fraction, x=sigma,
  y=mean recall. Only produced when a real sigma sweep exists.

Directory-parsing regex: `nmax(\d+)_k(\d+)`, matching real
`run_population.py` output exactly (no separate `K_cache` component, since
`K_cache` has always equaled `K_test` in the real pipeline).
`_sigma_values_in()` detects whether a sigma sweep exists per benchmark
directory; `analyze_dataset()` is called once per dataset regardless,
looping over `(param_dir, sigma)` pairs internally.

Debug JSON is written to `{debug_dir}/{label}_ball_debug.json`, structured
as `dataset -> params -> benchmarks -> budgets -> {hit_rate, mean_recall,
memory_util, ...}`, plus sanity flags and roll-up summary blocks, for
automated auditing of a full sweep without eyeballing every plot.

`compile_proposal.py` compiles this script's PNGs (glob-based over whatever
exists in `processed_dir / 'ball'`) into a PDF; its per-dataset header page
is only rendered when a section actually has a title, since ball is the only
proposal method.

## 9. Overlap-threshold hit policy and the intersection-fraction calibration study

### Motivation

An earlier slack-augmented CIG relaxation direction
(`D(t,Q) + r_t <= r_c*(1+slack)`) was abandoned: recovering a meaningful
fraction of missed queries needed large slack, and slack caused severe
false-positive dilution (SIFT about 2x extra base vectors admitted at
slack=1.0, ESCI up to about 140x in the worst cells). A hard, deterministic
relaxed threshold isn't a usable direction.

This motivated a different question: does the actual geometric overlap
between a cached candidate's K-sphere and a test query's own K-sphere
predict whether that query is an oracle hit (its true top-K physically
present in the cache)? If so, that overlap is a candidate signal for a
probabilistic recall estimate, rather than a hard certify/reject rule.

### Two overlap metrics

For a selected candidate `c` (center `Q`, real serving radius `r_c`) and
test query `t` (radius `r_t` from ground truth, `D = D(t, Q)`):

- **`angular_fraction(c, t)`**: the fraction of `t`'s own `r_t` spherical
  cap on `S^{d-1}` (the angular sphere this codebase's distances are
  natively defined on) that lies within `c`'s `r_c` cap.
- **`solid_ball_fraction(c, t)`**: the fraction of `t`'s own `r_t` solid
  Euclidean ball in ambient `R^d` (using chord distance) that lies within
  `c`'s `r_c` solid ball.

Both are computed exactly, via closed form and 1D numerical quadrature, not
Monte Carlo. `c.r_c` (the candidate's real serving radius, not the
idealized ground-truth K-th-neighbor radius) is used, since candidates are
proposed as nested spheres and selection may pick a sub-candidate whose
radius differs from the full-K radius; what matters at serving time is the
radius of the candidate actually cached.

`oracle_hit(c, t)` and `union_covered(t)` are the exact ground-truth labels
this signal is calibrated against: whether `t`'s true top-K is contained in
`c`'s cached mask, and whether it's contained in the union of all selected
candidates, respectively.

### Implementation (`simulations/population/debug_intersection_calibration.py`)

`exact_intersection_fraction(D, r_c, r_t, d)`: the angular predictor. Fast
paths for no-overlap, full-containment, and concentric poles; otherwise a 1D
`scipy.integrate.quad` over co-latitude, using the same quadrature method
for both numerator and denominator so quadrature error cancels rather than
compounds. `solid_ball_intersection_fraction(dist, R1, R2, n)`: the
solid-ball predictor via a hyperplane-cap decomposition, using the ratio
`(R1/R2)^n` rather than absolute volume (whose `Gamma` factor overflows
float64 at real embedding dimensions).

`compare_angular_vs_solid_ball` proposes and selects real candidates at a
given budget, then for a sample of real test queries computes both
fractions against every selected candidate plus the exact `oracle_hit`
label. `run_dual_predictor_sweep` runs this across the full
(benchmark, K, budget) grid, building one setup context per `(benchmark, K)`
and reusing it across that combo's budget loop, since proposal is
budget-independent.

`bin_calibration` / `auc_score`: binned reliability-diagram data and a
manual rank-based AUC (Mann-Whitney U, no sklearn dependency, matching the
rest of this codebase). Cross-validated independently by
`tests/intersection_fraction/test_intersection_fraction_calibration.py`,
which reimplements the angular formula from scratch and checks it against
both `exact_intersection_fraction` and Monte Carlo sampling on real SIFT/ESCI
pairs.

### Statistical methods

Three complementary fits per `(dataset, benchmark, K)` cell, pooled across
budgets excluding 100% (a degenerate configuration where `greedy_cost`
admits far more heavily-overlapping candidates, skewing intersection
fraction toward 1 almost everywhere):

- **Logistic regression** (`statsmodels` `GLM(family=Binomial())`):
  `P(hit) = sigmoid(b0 + b1*x)`, covering both the binary K=1 case and the
  continuous-recall K>1 case (`n_trials=K`) via the same grouped-response
  GLM family. Reports slope, slope p-value, and McFadden's pseudo-R^2.
- **Mann-Whitney U**: the AUC already computed by `auc_score` is the
  normalized Mann-Whitney U statistic, verified to match to under 1e-9.
- **Point-biserial correlation**: a linear (not rank) correlation, tends to
  read lower than AUC given the visibly sigmoid-shaped reliability curves;
  kept for completeness.

Pair counts run from thousands to millions, so p-values are essentially
always significant regardless of effect size; AUC and pseudo-R^2 are the
numbers that actually matter.

### Results

Full production sweep (both datasets, all 4 research benchmarks each,
K in {1, 5, 10, 20}, all 14 non-degenerate budget fractions):

- **SIFT**: AUC is uniformly high and essentially flat across K
  (0.993-0.997 for every cell), pseudo-R^2 correspondingly strong
  (0.77-0.87). Intersection fraction is an excellent ranking signal for SIFT
  regardless of K.
- **ESCI**: AUC is lower and shows real K-dependence, from 0.92-0.94 at K=1
  rising to 0.96-0.97 at K=20 (roughly matching SIFT's level). Pseudo-R^2
  shows the same pattern more sharply, from 0.31-0.38 at K=1 to 0.51-0.57 at
  K=20, still below SIFT at every K but narrowing. Since ESCI is much
  higher-dimensional than SIFT (384 vs 128), a larger K producing a larger,
  less degenerate top-K set appears to make the intersection fraction a more
  reliable signal at higher dimension.
- No meaningful difference in this AUC/pseudo-R^2 trend across benchmark
  cache sizes (256/512/1024/2048) within a dataset.

### The overlap-threshold hit policy

`overlap_threshold_hit_policy` (`utils/base/lemmas_sigma.py`) turns this
finding into an actual, uncertified, probabilistic hit policy, dispatched
via `algorithm="overlap_threshold"`:

1. **Offline**: `fit_r_t_hat(train_r_t, percentile)` computes a fixed radius
   estimate from cache/training query top-K radii only, never test queries
   (a real hit policy can't know its own true top-K distance at decision
   time).
2. **Per query**: for every cache entry, scores the solid-ball intersection
   fraction between the entry's own radius `r_c` and `r_t_hat`, keeping the
   best-scoring entry.
3. **Decision**: if the best score exceeds `threshold` (default 0.9),
   reports a hit and returns that entry's nearest-N vectors re-sorted by
   true distance (entries store `R_c` in ascending base-index order, not
   distance order). Otherwise a miss, matching the no-union lemmas'
   convention (no partial credit).

Evaluated end-to-end via `simulations/hit_policy/run_overlap_threshold_sweep.py`,
which reuses `debug_common.propose_full_pool`/`select_candidates` for real
candidates and `population_runner._run_simulation` for real hit
rate/recall against real ground truth.

### Full-matrix hit-policy results

Both datasets, all 4 benchmarks each, K in {1, 5, 10, 20}, all 14
non-100%-budget fractions, `r_t_hat` at median/p95/p99, threshold grid
{0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}:

- **`r_t_hat` percentile is not a tradeoff: p50 dominates**, in every
  individual (dataset, benchmark) cell, not just pooled. A smaller assumed
  radius produces a smaller target ball that's easier for a cache entry's
  own radius to mostly contain, raising the intersection-fraction score for
  entries that already generalize well, which correlates with higher
  realized recall too, not just more (weaker) hits.
- **SIFT hit rates are much higher than ESCI's** at matched
  budget/K/benchmark size, consistent with ESCI's weaker intersection
  fraction signal (lower AUC/pseudo-R^2, especially at low K).
- **Cache size has a real effect for ESCI that SIFT doesn't show**: ESCI's
  hit rate rises fairly consistently with cache size; SIFT shows no
  comparable trend, consistent with SIFT already being close to saturated.
- **Both hit rate and recall rise monotonically with budget fraction**,
  with a visible knee around budget 0.45-0.55.
- **Oracle (union) ceiling**: SIFT's p50 hit rate sits fairly close to its
  oracle ceiling (a gap of about 0.1), but ESCI's p50 hit rate is far below
  its ceiling (a gap of about 0.39) even though the ceilings themselves are
  similar between datasets. This reframes the hit-rate gap: it isn't that
  fewer of ESCI's true neighbors are physically present in the cache, it's
  that this single-candidate policy is much worse at identifying the right
  candidate for ESCI. A policy checking overlap against multiple
  top-scoring candidates, not just the single best, would need to close
  this gap for ESCI specifically.

### Open directions

- Directly test the query-clustering mechanism hypothesized to explain the
  near-binary best-fraction split (plot `D(t, best candidate)`'s
  distribution directly to confirm bimodality).
- A statistically-calibrated, per-dataset slack on the CIG inequality,
  informed by real `r_t` statistics, is still open as a secondary direction.
- Extend the multi-candidate overlap check hinted at by the oracle-ceiling
  gap above, particularly for ESCI.

## 10. Performance characteristics

The dominant cost in a real population run is `select_candidates`
(`greedy_cost`), which scales steeply with budget size; it hasn't needed
further optimization beyond the incremental delta-update design in section
4. `propose_candidates_ball` and `select_candidates` together dominate
runtime in the intersection-fraction sweep too (about 94% combined; the
quadrature math itself is about 2%).

**Known caveat**: `budget_fraction=1.0` completes but is memory-pressure
risky on constrained machines; it's excluded by default from pooled
statistical analysis (a degenerate configuration, see section 9) but the
sweep itself still computes it since it's cheap to include. Disk
persistence of proposed candidate pools (to reuse across separate process
runs) was considered and rejected: a candidate pool measured at about 2.2GB
and 6-8s to pickle per `(benchmark, K)`, comparable to or slower than
recomputing.

## 11. Test coverage map

| File | What it validates |
|---|---|
| `tests/utils/test_proposals.py` | Sphere-sweep coverage invariants, ball geometric correctness at K=0 and K>=1, exact named-neighbor coverage (no epsilon fudge), `r_c` monotonicity in K, the K>=1 guard on `_precompute_kth_base_dist`. |
| `tests/utils/test_lemmas.py` | Certification algorithm correctness including N=0 containment-mode reduction. |
| `tests/utils/test_selection.py` | All 4 strategies' behavior and invariants, incremental/chunked `greedy_cost` correctness against a naive reference including the chunk-boundary edge case. |
| `tests/utils/test_selection_with_new_candidates.py` | Selection against realistic ball(K=0)/ball(K>=1) candidate data from a clustered synthetic benchmark. |
| `tests/population/test_population_runner.py` | `_run_simulation` sigma/hit-rate correctness, recall bounds, the K=0 recall-division guard, `_write_selection_log`'s vectorized computation against a naive reference. |
| `tests/population/test_e2e_population_proposal.py` | Full pipeline end to end: K=0/K=5 smoke tests, containment-only path, `n_max` sweep, the K=0 containment regression test, budget-utilization range, sigma output-structure nesting, alpha-proposal backward compatibility. |
| `tests/population/test_e2e_population.py` | Lower-level end-to-end chain (propose, select across all 4 strategies, build, simulate) and output-file schema. |
| `tests/population/test_analyze_proposal.py` | Directory-naming regex, expected plots created, debug JSON schema. |
| `tests/population/microbenchmarks/*` | Containment microbenchmark correctness and its distance invariants. |
| `tests/intersection_fraction/test_intersection_fraction_calibration.py` | Independent reference-formula cross-check of the intersection-fraction calibration study, against both the exact quadrature implementation and Monte Carlo, on real SIFT/ESCI data. |

## 12. Known gaps

- **Proposal function duplication (partial)**: `propose_candidates` (generic
  alpha) remains a fully separate code path from the nested-sphere family.
  They share only low-level helpers (`_cig_mask`, `_normalize`), not the
  coverage/construction logic.
- **`QueryResult.error` is effectively dead**: `run_query` sets
  `error=metadata.get('error')`, but no lemma function ever sets
  `metadata["error"]`, and there's no `try/except` in `run_query` itself.
  Real failures propagate directly out of `run_query` rather than being
  captured per-query. Worth deciding explicitly whether defensive per-query
  capture or fail-fast is the desired behavior, rather than leaving an
  unused field that looks like it does something it doesn't.
