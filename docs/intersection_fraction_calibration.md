# Intersection-fraction calibration study

## 1. Motivation

The slack-augmented CIG relaxation direction (`D(t,Q) + r_t <= r_c*(1+slack)`)
was deprecated after `simulations/population/debug_dilution_cost.py`
showed that recovering a meaningful fraction of missed queries needed
large slack, and that slack caused severe false-positive dilution (SIFT
~2x extra base vectors admitted at slack=1.0; ESCI up to ~140x IQR-upper
in the worst cells). A hard, deterministic relaxed threshold isn't a
usable direction.

This pivots to a different, softer question: does the *actual geometric
overlap* between a cached candidate's K-sphere and a test query's own
K-sphere predict whether that query is an oracle hit (its true top-K
physically present in the cache)? If so, that overlap is a candidate
signal for a future *probabilistic* recall estimate, rather than a hard
certify/reject rule. This work is entirely offline and ground-truth-only
-- no serving-time estimation is implemented or attempted here, it's a
calibration study to check whether the signal is worth pursuing further.

## 2. Two overlap metrics

For a selected candidate `c` (center `Q`, real serving radius `r_c`) and
test query `t` (radius `r_t` from ground truth, `D = D(t, Q)`), two
independent geometric overlap metrics are computed and compared:

- **`angular_fraction(c, t)`**: the fraction of `t`'s own `r_t` spherical
  cap (on `S^{d-1}`, the angular/cosine-similarity sphere this codebase's
  distances are natively defined on) that lies within `c`'s `r_c` cap.
- **`solid_ball_fraction(c, t)`**: the fraction of `t`'s own `r_t` solid
  Euclidean ball (in ambient `R^d`, using chord distance rather than
  angular distance) that lies within `c`'s `r_c` solid ball.

Both are exact (closed-form / numerical-quadrature), not Monte Carlo
estimates -- see Section 3. Comparing the two answers whether the extra
correctness of staying on the sphere (matching this codebase's actual
distance metric) matters in practice, versus the cheaper, more standard
solid-ball computation.

- **`oracle_hit(c, t)`**: exact, ground-truth, per-candidate --
  `c.R_c[test_gt_indices[t]].all()`, i.e. whether `t`'s true top-K
  neighbor indices are physically contained in `c`'s actual cached mask.
- **`union_covered(t)`**: exact, ground-truth, whole-cache -- whether
  `t`'s true top-K is physically present somewhere across the union of
  *all* selected candidates' balls. Identical definition to
  `debug_vector_coverage.py`'s `covered` label; computed independently in
  this script so it doesn't need to import from there.

`c.r_c` (the candidate's real serving radius) is used, not the idealized
ground-truth `r_q` (the candidate query's own true K-th-neighbor
distance). This was a deliberate correction during design: candidates
are proposed as nested spheres (radius = 1st, 2nd, ..., n_max-th neighbor
distance, each a separate candidate), and `select_candidates` may pick a
sub-candidate whose radius differs from `r_q`. At serving time, what
matters is the radius of the candidate actually cached, not the
theoretical full-K-sphere radius.

### Why exact, not Monte Carlo

An earlier version of this metric used Monte Carlo sampling (uniformly
sampling points within a query's cap and checking membership in the
candidate's cap). This was superseded by exact computation: a single
cap's fractional volume on `S^{d-1}` has a known closed form via the
regularized incomplete beta function (Li 2011), and the two-cap
intersection reduces to a well-behaved 1D numerical quadrature one
dimension down (see `exact_intersection_fraction`'s docstring for the
derivation). The solid-ball version has an equivalent closed form via
the same incomplete-beta machinery, using a hyperplane-cap decomposition
of the two balls. Both avoid MC's finite-sample noise and per-pair
sampling cost entirely; `sample_cap_points` (the original MC sampler) is
retained only as an independent correctness check in tests, not in the
production computation path.

An even earlier heuristic, `(r_c + r_t - D) / (2*r_t)` clipped to
`[0,1]`, was rejected before either of the above: it had the right
boundary behavior but no proof it tracked real volumetric overlap.

## 3. Implementation

All in `simulations/population/debug_intersection_calibration.py`
(standalone debug script -- reuses `debug_common.py` helpers, no changes
to the real pipeline).

**`exact_intersection_fraction(D, r_c, r_t, d, verify=False)`**: the
angular predictor. Fast exact paths for no-overlap, full-containment,
and concentric poles; otherwise a 1D `scipy.integrate.quad` over
co-latitude, using `conditional_cap_fraction` (the single-cap closed
form one dimension down) as the integrand. Both the numerator and
denominator integrals use the same quadrature method, so quadrature
error cancels between them rather than compounding -- see Section 4 for
why this matters. `verify=True` additionally recomputes via a split
integration range and warns if the two disagree by more than 1e-6; off
by default since it roughly triples the cost per call (see Section 4)
and is only useful for spot-checking numerically borderline inputs, not
production sweeps.

**`solid_ball_intersection_fraction(dist, R1, R2, n)`**: the solid-ball
predictor, via a hyperplane-cap decomposition of the two balls (using
chord distance, `chord_distance_from_angular`, to convert from this
codebase's angular units into ambient Euclidean distance). Uses the
ratio `(R1/R2)^n` rather than each ball's absolute volume, since the
absolute volume's `Gamma(n/2+1)` factor overflows float64 at real
embedding dimensions while the ratio stays well-behaved.
`solid_ball_intersection_fraction_literal` is a slower direct-volume
version, kept only as a cross-check.

**`compare_angular_vs_solid_ball(dataset, benchmark_name, K, budget_fraction, ...)`**:
proposes and selects real candidates at the given budget (reusing
`dc.propose_full_pool` + `select_candidates`, unchanged from the real
selection logic), then for a sample of real test queries computes both
`angular_fraction` and `solid_ball_fraction` against every selected
candidate (capped at `max_candidates_per_query`, subsampled if there are
more), alongside the exact `oracle_hit` label. Returns flattened
per-(candidate, query) pair arrays plus, per test query, the best (max)
fraction under each predictor independently (the two can pick different
"best" candidates) and `union_covered`.

**`run_dual_predictor_sweep(dataset, k_values, budget_fractions, ...)`**:
runs `compare_angular_vs_solid_ball` across the full (benchmark, K,
budget) grid. `propose_full_pool` is budget-independent, so this builds
one setup context per `(benchmark, K)` via `_prepare_combo_context`
(load benchmark, propose full pool) and reuses it across that
`(benchmark, K)`'s whole budget loop, rather than re-proposing on every
single budget -- see Section 4.

**`bin_calibration`** / **`auc_score`**: binned reliability-diagram data
(per-bin mean `oracle_hit`, with a minimum-pair-count-per-bin threshold
to suppress small-sample noise) and a manual rank-based AUC (Mann-Whitney
U statistic, no sklearn dependency -- none is used anywhere else in this
codebase).

**Plots** (per dataset, via `main()`):
- `dual_calibration_vs_intersection_fraction.png` -- binned reliability
  curves, rows = cache size, columns = K, one line per predictor
  (angular, solid-ball), pooled across all swept budgets.
- `dual_best_candidate_scatter_budget{N}pct.png` (one file per budget
  actually present in the swept data) -- raw, unbinned, y-jittered dot
  plot: one point per test query, x = best-candidate fraction (angular
  and solid-ball overlaid by color), y = `union_covered`, on a single
  continuous `[0,1]` x-axis.
- `angular_vs_solid_ball_scatter.png` -- direct per-pair scatter of the
  two predictors against each other, colored by `oracle_hit`, with a
  y=x reference line.
- `auc_comparison_grid.png` -- angular AUC vs. solid-ball AUC per
  budget, as connected point-pairs across the full benchmark/K grid.

**Independent validation**: `simulations/population/validate_intersection_fraction.py`
is a standalone script (not wired into tests/CI) that reimplements the
angular formula from scratch, independently of this module's own code,
and cross-checks it against `exact_intersection_fraction` and against
Monte Carlo sampling, including on real SIFT/ESCI pairs. Run directly
(`python3 simulations/population/validate_intersection_fraction.py`)
when validating a change to the core formula.

**Tests**: `tests/population/test_debug_intersection_calibration.py`,
37 tests covering: the cap sampler against a known closed-form volume
formula (still used as an independent MC cross-check, not in production
computation), its radial distribution against theory (KS test) and
tangential-direction unbiasedness, `exact_intersection_fraction`'s exact
fast paths and boundary continuity, elementary-formula cross-checks at
d=2/d=3, MC cross-check, underflow/dtype guards, the solid-ball formula
(hand-computable case, brute-force rejection sampling, literal-vs-ratio
agreement), AUC/binning correctness, and the `oracle_hit`/`union_covered`
invariants (both synthetic and on real data).

## 4. Challenges faced and how they were resolved

**OOM crash in the MC sampler (now retained only for test validation).**
An early version of `sample_cap_points` generated samples for all `M`
poles in one vectorized call, materializing several full
`(M, n_samples, d)` float64 intermediates simultaneously. While
debugging a flaky test, `n_samples` was bumped from 100 to 20000 for one
test case; at SIFT's `d=128`, `M=n_test=512`, that made each
intermediate array `512*20000*128*8 bytes ~= 10.5GB`, with peak usage
across several intermediates plausibly 40-60GB -- this crashed the
machine it ran on. Fix: `sample_cap_points` now processes poles in
bounded chunks (default 32, with a hard safety cap that shrinks further
for large `n_samples`), and downcasts the returned array to float32.
Verified directly: the exact same call that previously would have used
40-60GB now peaks at ~783MB. This sampler is no longer in the production
computation path (superseded by the exact quadrature/closed-form
predictors, Section 2), but is retained in tests as an independent
cross-check, so this fix still matters.

**Reliability-curve binning artifact.** With per-budget lines and a
fixed 20-bin resolution, cells with fewer selected candidates (e.g. ESCI
at low budgets) showed wild, non-monotonic reliability curves --
individual bins with only a handful of pairs, where a single miss swings
the bin mean wildly. Fixed two ways: `bin_calibration` gained a
`min_count` threshold (bins below it are excluded from the curve rather
than plotted as noise), and `plot_dataset_dual_reliability` pools all
swept budgets into one line per predictor per cell instead of one line
per budget -- both more data per bin and a far less cluttered legend (2
lines instead of up to 15).

**Best-candidate scatter revealed a near-binary distribution.**
`best_fraction` (max over selected candidates for a given test query)
is often close to exactly 0.0 or exactly 1.0 -- consistent with a
clustered dataset where a query's closest candidate is either in the
same tight cluster (comfortably swallowing nearly all of the query's
K-sphere) or in a different cluster entirely (essentially no overlap).
The original plot design split each K into zoomed near-0/near-1
sub-columns with x-jitter for visibility; this was later found to be a
real bug source (jitter added directly to values sitting at exactly 0 or
1 could render as apparently out-of-`[0,1]` points) and was replaced
with a single continuous `[0,1]` panel per K, y-jitter only (x is never
jittered), plus explicit `[0,1]` range assertions both at computation
time and at plot time as a permanent invariant check.

**Phantom-budget plots.** Several plotting functions defaulted to
iterating the module-level `BUDGET_FRACTIONS` (the full 15-value
production list) rather than whatever budgets were actually present in
a given sweep's data, so a reduced/partial sweep could silently produce
blank plots for budgets that were never run. Fixed with `_actual_budgets`,
which derives the real swept-budget list from the data itself; plotting
functions default to this instead of the module constant, and
`plot_dataset_dual_best_candidate_scatter` skips writing entirely for a
budget with no data anywhere in the grid.

**Redundant candidate proposal across the budget sweep.** Profiling
`run_dual_predictor_sweep` found candidate proposal
(`propose_full_pool`) and selection (`select_candidates`) dominate
runtime (~94% combined; the quadrature math above is ~2%), and that
proposal -- despite being budget-independent by its own docstring -- was
being redundantly recomputed on every single budget in the sweep instead
of once per `(benchmark, K)`. Fixed via `_prepare_combo_context`, built
once per `(benchmark, K)` and reused across that combo's budget loop.
Disk persistence of proposed candidate pools (to reuse across separate
process runs, not just within one sweep) was considered and rejected:
measured at ~2.2GB and 6-8s to pickle per `(benchmark, K)` pool (`R_c` is
an unpacked bool array of length `n_base`), which is comparable to or
slower than just recomputing -- not worth the disk footprint or format
risk for the size of win available. `select_candidates` itself (~65% of
total cost, scaling steeply with budget size) was left untouched --
production selection code used well beyond this debug module, and past
work on it hit real OOM issues.

**Reference-implementation ratio drifting past `[0,1]`.** The
independent validation script's reference formula
(`validate_intersection_fraction.py`) originally computed its numerator
via `scipy.integrate.quad` but its denominator via a separate closed
form (`gammaln`-based normalization constant). Without matching
quadrature error between numerator and denominator, the ratio could
drift slightly past the true `[0,1]` bound at extreme dimension (ESCI,
d=384): 10 real pairs disagreed with production by >1e-4, all cases
where production's quadrature correctly stayed just under 1.0 while the
reference drifted to ~1.0003-1.0009. Fixed by computing the reference's
denominator via the same `quad()` call as its numerator, mirroring
`exact_intersection_fraction`'s own num/den symmetry -- after the fix,
max disagreement on the same real pairs dropped to ~2e-16 (machine
precision).

## 5. Results so far

Full production sweep run 2026-07-30 (`simulations/debug/population_intersection_calibration/20260730_132954/`):
both datasets, all 4 research benchmarks each, `K in {1, 5, 10, 20}`,
all 14 non-degenerate budget fractions (budget=100% excluded from the
sweep entirely, not just pooled analysis -- see 5b). Numbers below are
`pooled_stats_excl_100pct` from that run's `summary.json`; see
`sift/auc_and_pseudo_r2_vs_k.png` and `esci/auc_and_pseudo_r2_vs_k.png`
(generated via `simulations/population/plot_calibration_summary.py`)
for the visual version.

- **SIFT**: AUC is uniformly high and essentially flat across K --
  0.993-0.997 for every (benchmark, K) cell, no meaningful trend with K.
  Pseudo-R^2 is correspondingly strong (0.77-0.87) and also flat.
  Practically: intersection fraction is an excellent ranking signal for
  SIFT regardless of K, and K=20 doesn't change that picture at all.
- **ESCI**: AUC is lower and shows a real K-dependence that SIFT
  doesn't -- K=1 sits at 0.92-0.94 (noticeably below SIFT's K=1), then
  rises with K, reaching 0.96-0.97 at K=20 (roughly matching SIFT's
  level). Pseudo-R^2 shows the same pattern much more sharply: 0.31-0.38
  at K=1 climbing to 0.51-0.57 at K=20 -- still well below SIFT's
  pseudo-R^2 at every K, but the gap narrows as K grows. Since ESCI is
  much higher-dimensional than SIFT (384 vs 128), this is consistent
  with the "sparse away from 0/1, curse-of-dimensionality" pattern
  flagged earlier in this doc: more neighbors (larger K) means a larger,
  less degenerate top-K set, which appears to make the intersection
  fraction a more reliable signal at higher dimension. **Does K=20
  change the picture from K=1/5/10?** Yes for ESCI specifically -- it's
  the best-performing K in every ESCI benchmark, not just "one more data
  point" -- but no for SIFT, which is already saturated by K=1.
- No meaningful difference in the AUC/pseudo-R^2 trend across benchmark
  sizes (256/512/1024/2048 cache queries) within a dataset -- the K
  effect described above is consistent across all 4 benchmarks per
  dataset.

## 5a. Statistical methods for x=intersection_fraction, y=hit/recall

`debug_intersection_calibration.py` fits three complementary checks per
(dataset, benchmark, K) cell, pooled across budgets (budget=100%
excluded -- see 5b) via `_compute_cell_stats`:

- **Logistic regression** (`fit_binomial_glm`, via `statsmodels`'
  `GLM(..., family=Binomial())`): `P(hit) = sigmoid(b0 + b1*x)`. The
  same model family covers both the binary K=1 case (`n_trials=1`) and
  the continuous-recall K>1 case (`n_trials=K`, `successes=round(recall*K)`)
  -- recall is a `Binomial(K, p)/K` proportion, so grouped-response GLM
  is the natural generalization, not a different framework. Reports
  slope, slope p-value, and McFadden's pseudo-R^2 (`1 - llf/llf_null`).
- **Mann-Whitney U** (`mannwhitney_test`, via `scipy.stats.mannwhitneyu`):
  the AUC already computed by `auc_score` *is* the normalized
  Mann-Whitney U statistic -- verified to match to <1e-9 in
  `tests/population/test_debug_intersection_calibration.py`. The test
  adds a p-value for "is this AUC distinguishable from 0.5" on top of a
  number we already compute.
- **Point-biserial correlation** (`point_biserial_test`, via
  `scipy.stats.pointbiserialr`): ordinary Pearson r between continuous
  x and binary/proportional y. Since the reliability curves are visibly
  sigmoid-shaped, point-biserial (a *linear* correlation) tends to read
  lower than AUC (a *rank* statistic) -- not a contradiction, just two
  different questions. Kept mainly for completeness; AUC/logistic
  regression are the more appropriate primary tools for this data's
  shape.

**Caveat**: pair counts run from thousands to millions, so p-values are
essentially always significant regardless of practical effect size --
treat them as a sanity floor (did we get pure noise), not as the
headline number. AUC and pseudo-R^2 are the effect-size numbers that
actually matter.

## 5b. Standing default: exclude budget=100% from pooled analysis

Budget=100% is a degenerate configuration: `select_candidates`
(`greedy_cost`) admits far more, heavily-overlapping candidate spheres
once the budget is effectively unconstrained, which skews intersection
fraction toward 1 almost everywhere and distorts any statistic pooled
across budgets. `_actual_budgets(..., exclude_full=True)` drops
`budget_fraction == 1.0` from the default budget list used by pooled
plots and `_compute_cell_stats`. This is a standing default for this
analysis area going forward, not a one-off: the sweep itself still
computes budget=100% (cheap, no reason to discard the data), and
per-budget outputs (e.g. the best-candidate scatter) still include it
-- only pooled/aggregate views exclude it by default.

## 5c. Continuous recall analysis

`compare_angular_vs_solid_ball` also computes `recall_pair` (# of a
query's true top-K neighbors present in one candidate's cache, divided
by K) alongside the existing binary `oracle_hit_pair`. At K=1 this is
identical to `oracle_hit` (0 or 1); it's only informative at K>1.
`plot_dataset_recall_reliability` mirrors the existing binary
reliability curve (`plot_dataset_dual_reliability`) but with mean
recall on the y-axis, and skips K=1 automatically.

K=20 was added to the main population pipeline (all 4 research
benchmarks, both datasets, all non-100% budgets) specifically so this
recall analysis would have a third, more granular K value beyond K=5/10
-- no new ground truth needed, `K_VALUES = [1, 5, 10, 20]` was already
prepared for every benchmark.

## 5d. Hit-policy prototype: overlap-threshold policy

`overlap_threshold_hit_policy` (`utils/base/lemmas_sigma.py`) turns the
calibration finding above into an actual (uncertified, probabilistic --
not CIG/HGG) hit policy, dispatched via `algorithm="overlap_threshold"`
in `simulations/core/simulate.py`'s `CacheSimulator`:

1. **Offline**: `fit_r_t_hat(train_r_t, percentile)` computes a single
   fixed radius estimate from *cache/training* query top-K radii only
   (never test queries -- a real hit policy can't know its own true
   top-K distance at decision time, which is exactly what this estimate
   stands in for). Tested at the median, p95, and p99 percentiles.
2. **Per query**: for every cache entry, converts the entry's own
   radius `r_c` and the fixed `r_t_hat` to chord distance and scores
   the solid-ball intersection fraction against the query-to-entry
   distance `D`. Keeps the best (highest-scoring) entry.
3. **Decision**: if the best score exceeds `threshold` (default 0.9),
   reports a hit and returns that entry's nearest-N vectors (sorted by
   true distance to the query, not raw storage order -- entries built
   via `CandidateEntry.to_cache_entry()` store the full `R_c` set in
   ascending base-index order, not distance order, so the policy must
   re-sort before truncating to N). Otherwise, a miss with no
   partial-credit accumulation, matching the no_union lemmas'
   convention.

Evaluated end-to-end via `simulations/hit_policy/run_overlap_threshold_prototype.py`,
which reuses the existing pipeline in full rather than adding new
measurement code: `debug_common.propose_full_pool`/`select_candidates`
for real proposed/selected candidates, and
`population_runner._run_simulation` (not `run_hit_policy.py`, which
hardcodes `correct_results=0` and never computes recall) for real
`hit_rate`/`mean_recall` against real ground truth.

Unit tests: `tests/utils/test_lemmas.py` (`fit_r_t_hat`'s percentile
correctness and no-test-query-input-by-construction; the policy's
input validation, empty-cache miss, near-perfect self-overlap hit, and
a regression test confirming the returned vectors are the true
nearest-N by distance, not the first N of the entry's raw index array).

## 5e. Hit-policy prototype: full-matrix results (2026-07-30)

Before the full sweep, `overlap_threshold_hit_policy`'s per-query
scoring loop was profiled and found to be a genuine (not hypothetical)
bottleneck -- a plain Python loop over cache entries, each doing scalar
chord-distance/intersection-fraction math and re-fetching
`entry.get_radius()` on every query even though it's query-invariant.
Vectorized (see `_solid_ball_intersection_fraction_array`,
`_chord_distance_from_angular_array`, and the new `precomputed_r_c`
parameter in `utils/base/lemmas_sigma.py`) -- the scoring loop itself
dropped from ~0.89s to ~0.04s per 512-query pass at a realistic scale
(K=20, 67 cache entries), a ~20x reduction for that piece. Total
per-query time is now dominated by the result-vector gathering on a hit
(sorting the entry's full R_c set down to the nearest N), a separate,
inherent cost proportional to R_c set size that vectorizing the scoring
loop doesn't address; a modest `argpartition`-based improvement was
also made there. Parity tests confirm the vectorized path is
bit-identical to the original scalar loop.

**Full sweep**: both datasets, all 4 benchmarks each, `K in {1, 5, 10,
20}`, all 14 non-100%-budget fractions, `r_t_hat` at median/p95/p99,
threshold grid `{0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}` -- 448 combos,
9408 result rows total (`simulations/hit_policy/reports/overlap_threshold/20260730_135312/`).
3 combos (all tiny-budget/small-benchmark ESCI cells that selected zero
candidates) hit a bug where an empty cache's `best_score=None` crashed
the threshold-aggregation step instead of correctly reporting a miss;
fixed in `run_overlap_threshold_sweep.py` and backfilled.

**Results** (`analyze_overlap_threshold.py`'s plots, same directory --
gridded by benchmark/cache-size (rows) x K (columns), one figure per
dataset, never pooled across benchmarks or datasets; only percentile
stays pooled as colored lines within each subplot, which is the actual
comparison being made). An earlier version of this section and its
plots incorrectly averaged together both datasets and all 4 cache-size
benchmarks at every point -- caught during review, since `budget_fraction`
isn't even on the same absolute scale across datasets (each has its own
`n_base`). Fixed; findings below are re-verified against the properly
separated data.

- **`r_t_hat` percentile is not a tradeoff here -- p50 dominates, and
  this holds in every individual (dataset, benchmark) cell**, not just
  in the pooled aggregate. At threshold=0.9, per-cell means (averaged
  over K and budget within each benchmark) show p50 strictly ahead of
  p95 and p99 on both hit rate and mean recall in all 8 benchmarks
  (e.g. SIFT c1024: hit rate 0.406/0.374/0.365, recall
  0.999/0.929/0.857 for p50/p95/p99 respectively; same ordering holds
  for every other benchmark). This runs counter to the naive
  expectation that a larger assumed radius (higher percentile) trades
  hit rate for confidence/recall -- instead, a smaller `r_t_hat` (p50)
  produces a smaller target ball that's easier for a cache entry's own
  radius to mostly contain, raising the intersection-fraction score for
  entries that already generalize well, which turns out to correlate
  with *higher* realized recall too, not just more (weaker) hits.
- **SIFT hit rates are much higher than ESCI's** at matched
  budget/K/benchmark-size (e.g. p50, threshold=0.9, averaged over K and
  budget: SIFT ranges 0.366-0.406 across its 4 benchmarks vs ESCI's
  0.058-0.099) -- consistent with 5's finding that ESCI's intersection
  fraction is a weaker signal (lower AUC/pseudo-R^2, especially at low
  K) and with the "curse of dimensionality" pattern noted earlier in
  this doc.
- **Cache size (benchmark) has a real effect for ESCI that SIFT doesn't
  show**: at p50/threshold=0.9, ESCI's hit rate rises fairly
  consistently with cache size (256->512->1024->2048: 0.058, 0.083,
  0.091, 0.099) -- more cache/training queries give the proposal step a
  more diverse candidate pool to select from at the same budget
  fraction. SIFT shows no comparable monotonic trend (0.401, 0.366,
  0.406, 0.398 across the same 4 sizes) -- consistent with SIFT already
  being close to saturated (AUC ~0.99 regardless of K per section 5),
  leaving little room for more cache data to help.
- **Both hit rate and recall rise monotonically with budget fraction**
  for every benchmark/K/percentile, with a visible knee around
  budget=0.45-0.55 where hit rate jumps -- consistent with more/larger
  cached spheres becoming available as budget grows. Recall saturates
  near 1.0 well before budget=1.0 for most cells once hit rate is above
  roughly 0.1-0.2, i.e. once the policy starts reporting hits at all,
  those hits tend to already be high-recall.
- **Oracle (union) ceiling**: added an "oracle hit rate" line to
  `hit_rate_vs_budget` -- for each budget, the fraction of test queries
  whose full top-K neighbor set is covered by the UNION of all selected
  candidates (best possible for any single-candidate policy, computed
  via `compute_oracle_hit_rate.py`, same pattern as
  `debug_vector_coverage.py`'s `vector_coverage`/
  `debug_intersection_calibration.py`'s `union_covered`). At
  threshold=0.9, pooled: **SIFT's p50 hit rate (0.393) sits fairly
  close to its oracle ceiling (0.490, gap 0.097)**, but **ESCI's p50
  hit rate (0.083) is far below its ceiling (0.469, gap 0.387) even
  though the ceilings themselves are similar between datasets**. This
  reframes the earlier "SIFT hit rates are much higher than ESCI's"
  point: it's not that fewer of ESCI's true neighbors are physically
  present in the cache (the ceilings are close), it's that the
  single-candidate `overlap_threshold_hit_policy` is much worse at
  identifying the right candidate for ESCI -- consistent with ESCI's
  weaker intersection-fraction signal (lower AUC/pseudo-R^2 per
  section 5). This is exactly the gap a smarter policy (e.g. checking
  overlap against multiple top-scoring candidates, not just the single
  best) would need to close for ESCI specifically.

## 6. Next steps

- **Broader performance pass**: this round's fixes (proposal caching,
  `verify=False` default) addressed what profiling surfaced so far, but
  a broader pass was flagged as worth doing before the full sweep grows
  much larger in scope.
- **Directly test the query-clustering mechanism** hypothesized to
  explain the near-binary `best_fraction` split -- e.g. plot the
  distribution of `D(t, best candidate)` itself to confirm it's bimodal,
  rather than only inferring the mechanism from the fraction's shape.
- **Idea B from earlier discussion** (statistically-calibrated,
  per-dataset slack on the CIG inequality, informed by real `r_t`
  statistics rather than arbitrary constants) is still open -- now
  secondary to the intersection-fraction direction, not abandoned.
- **Run the overlap-threshold prototype (5d) across the full benchmark
  matrix**, not just the single-benchmark smoke test done so far, and
  compare hit-rate/recall tradeoffs across the three `r_t_hat`
  percentiles and a range of thresholds -- the single-benchmark run so
  far is enough to confirm the mechanism works end-to-end, not enough
  to draw conclusions about where the threshold/percentile should sit
  in production.
