# Population Pipeline Profiling Baseline

Phase 3 Round 1 deliverable. Produced with `simulations/population/profile_population.py`
(the profiling harness) against synthetic, clustered data generated on the fly --
no dependency on real prepared SIFT/ESCI benchmarks. Covers the full pipeline as
scoped by the user: benchmark/ground-truth construction -> loading train (cache)
queries -> proposing candidates -> selecting candidates -> building a `KVCache`
from the selection -> running the simulator on test queries.

This is a ballpark baseline, not exhaustive benchmarking: the goal is to find out
which stages are worth optimizing next round, not to produce publication-grade
numbers. Raw data behind every number below is in
`simulations/population/profiling_grid_results.json`.

## Headline finding

**The pipeline does not currently scale past roughly `n_cache ~ 1000` in
reasonable time, and the dominant cost is *not* where it was expected to be.**
Going in, `select_candidates`' per-iteration set-difference rescan
(`docs/KNOWN_ISSUES.md` #1) was the prime hotspot suspect. In practice, at every
scale tested, **`simulate()` and `build_kv_cache()` cost far more than
`select_candidates()`** -- by one to two orders of magnitude once the selected
cache holds more than a few hundred entries.

A single `n_cache=4096` configuration (`n_base=16384`, `n_max=50`, point-like)
was run and killed after exceeding 11 minutes without finishing -- it never
completed even one `simulate()` call at that scale. That data point is excluded
from the tables below; its exclusion is itself the finding.

## Grid run

| method | n_cache | n_base | n_test | n_max | n_candidates |
|---|---|---|---|---|---|
| point | 256  | 1024 | 128 | 5  | 1,280 |
| point | 256  | 1024 | 128 | 50 | 12,800 |
| point | 1024 | 4096 | 512 | 5  | 5,120 |
| point | 1024 | 4096 | 512 | 50 | 51,200 |
| ball  | 256  | 1024 | 128 | 5  | 1,142 |
| ball  | 256  | 1024 | 128 | 50 | 12,662 |
| ball  | 1024 | 4096 | 512 | 5  | 4,575 |

(`ball`/1024/`n_max=50` was in progress when the grid was stopped early once the
scaling trend was already clear from the other seven combos; not included.)

K=5, metric=angular, strategy=greedy_cost, budget fractions [0.1, 0.3] of
`n_base`, clustered synthetic data (8 clusters, noise_scale=0.05), seed=42.

## Per-stage timing and memory

All times are wall-clock (`time.perf_counter`); memory is `tracemalloc` peak
during that stage only (not cumulative across stages).

### Stage 0-1: benchmark construction and candidate proposal

| method | n_cache | n_max | generate_benchmark | propose_candidates | propose peak MB |
|---|---|---|---|---|---|
| point | 256  | 5  | 0.06s | 0.19s | 3.2 |
| point | 256  | 50 | 0.05s | 1.10s | 60.5 |
| point | 1024 | 5  | 0.30s | 0.75s | 31.7 |
| point | 1024 | 50 | 0.31s | **8.00s** | **339.6** |
| ball  | 256  | 5  | 0.05s | 0.13s | 3.1 |
| ball  | 256  | 50 | 0.05s | 1.11s | 60.2 |
| ball  | 1024 | 5  | 0.31s | 0.82s | 30.5 |

- `propose_candidates` scales roughly linearly in `n_max` (10x n_max ~ 10x
  time and memory, comparing n_max=5 vs 50 at fixed n_cache), confirming the
  nested-sphere redesign's per-query cost model
  (`O(n_max * (n_base + n_train + n_test))`) holds in practice.
- Memory scales with `n_max * n_cache` as expected (more candidates, each
  storing its own `R_c` array) -- 339.6 MB at n_cache=1024/n_max=50 versus
  3.2 MB at n_cache=256/n_max=5, roughly the 4x (n_cache) * 10x (n_max) = 40x
  candidate-count growth (1,280 -> 51,200).
- `generate_benchmark` (brute-force ground truth via `MainMemory.top_k_search`)
  is cheap at these scales and grows roughly linearly with `(n_cache + n_test)
  * n_base` -- not a near-term concern.

### Stage 2-4: select, build cache, simulate (larger budget, fraction=0.3)

The smaller budget (fraction=0.1) never selected anything in this synthetic
setup (0 candidates fit; expected for a tight budget against clustered data
with large candidate radii) so all interesting numbers are at fraction=0.3.

| method | n_cache | n_max | budget | n_selected | select_candidates | build_kv_cache | build peak MB | **simulate** |
|---|---|---|---|---|---|---|---|---|
| point | 256  | 5  | 307  | 91    | 0.11s | 0.04s | 3.3   | 3.79s |
| point | 256  | 50 | 307  | 811   | 0.51s | 0.25s | 29.4  | 3.98s |
| point | 1024 | 5  | 1228 | 435   | 0.30s | 0.44s | 64.0  | **154.6s** |
| point | 1024 | 50 | 1228 | 6,159 | 3.15s | 6.89s | **904.6** | **141.3s** |
| ball  | 256  | 5  | 307  | 68    | 0.06s | 0.02s | 2.5   | 2.39s |
| ball  | 256  | 50 | 307  | 1,532 | 0.50s | 0.44s | 55.5  | 7.35s |
| ball  | 1024 | 5  | 1228 | 340   | 0.26s | 0.33s | 50.0  | **124.6s** |

Reading down any column at fixed `n_max`, going from n_cache=256 to 1024 (a 4x
scale-up in cache/base/test size): `select_candidates` grows ~3x, `build_kv_cache`
grows ~15-30x, and **`simulate` grows ~35-40x** (3.8s -> 154.6s). That is
worse than quadratic in the scale factor, for the stage with the least memory
footprint (2 MB peak, unlike propose/build). This means `simulate`'s cost is
almost entirely CPU time inside the per-query hit-check loop, not memory
pressure -- consistent with a cProfile deep-dive taken during harness
development (`--profile-stage simulate`), which showed `angular_distance`'s
per-call `np.linalg.norm`/`np.clip` overhead dominating `lemma1_no_union_sigma`,
called once per (test query, cached entry) pair -- i.e. **O(n_test * n_selected)
distance calls**, each paying full Python-function-call and numpy-dispatch
overhead for a single-vector distance.

### The `build_kv_cache` memory blowup, explained

At `n_cache=1024, n_max=50, B=1228`, `build_kv_cache` peaked at **904.6 MB**
building a cache of only 6,159 entries against a 4,096-vector base. Verified
directly: those 6,159 selected candidates have `rc_size` averaging **512**
(exactly `n_base/8`) with almost no variance (min 509, max 512) -- i.e. nearly
every selected candidate's sphere covers about an eighth of the entire base,
and `sum(rc_size)` across all selected candidates is **3.15 million** base-vector
references. Each reference materializes its own normalized `top_k_matrix` row
(`to_cache_entry`'s `build_matrix=True` path) -- `3.15M rows * 32 dims * 4
bytes ~ 403 MB` just for that one array class, which is the dominant share of
the measured peak.

This is a real design gap, not a synthetic-data artifact: **the memory budget
`B` only bounds the count of *distinct* base vectors ever cached
(`cached_mask`'s cardinality) -- it does not bound the redundant per-entry
storage cost when many selected candidates' spheres overlap heavily.** With
`n_max=50`, the nested-sphere redesign intentionally proposes many overlapping
radii per query; `_select_greedy_cost`'s deferred-flush step (the final loop in
`utils/base/selection.py::_select_greedy_cost`) then unconditionally accepts
*every* deferred, zero-marginal-cost candidate with any positive training
coverage benefit, with no cap on how many such candidates get added -- each
one duplicating a ~512-vector matrix copy of base vectors that were already
counted against the budget by an earlier candidate. `n_selected` (6,159) ending
up larger than `n_base` (4,096) is the visible symptom.

### Hit rates (sanity check, not a design target this round)

Hit rates land in the 20-25% range in this synthetic clustered setup at
fraction=0.3 budgets, and 0% at fraction=0.1 (nothing fit). Increasing with
budget as expected; not tuned for realism since the synthetic generator's
cluster count/noise are arbitrary. Not a signal to act on -- only confirms the
harness's simulate stage produces sane, non-degenerate results.

## Data-structure audit

(See also the comment block at the top of `simulations/population/profile_population.py`.)

| Structure | Current representation | Assessment |
|---|---|---|
| `CandidateEntry.covered_train_indices` / `covered_test_indices` | Python `set[int]` | Standout bitmask conversion candidate. `_select_greedy_cost`'s per-iteration `len(c.covered_train_indices - covered_train_set)` is a Python set-difference re-evaluated for every active candidate on every outer-loop iteration. Given `n_train_queries`/`n_test_queries` are fixed and known up front, a numpy boolean array (or bitset) indexed by query id would turn this into a vectorizable AND/popcount, and the "newly covered" check across *all* active candidates at once into a single matrix operation instead of one Python-level set op per candidate per iteration. Measured as cheaper than `simulate`/`build_kv_cache` in this round's grid, but its cost is a real, currently-hidden contributor once candidate counts exceed ~50k (n_cache=1024, n_max=50 case: 3.15s for select at that scale, growing with n_max). |
| `CandidateEntry.R_c` | numpy `int32` array | Already good. No change needed. |
| `selection.py`'s `cached_mask` | numpy `bool` array, indexed by base vector id | Already good. No change needed. |
| `KVCache.cache` | Python `list[CacheEntry]`, each with its own `top_k_matrix` (numpy array) | Not a container-type problem, but a **duplication** problem: overlapping candidates' `R_c` sets are never deduplicated against each other when building matrices, so heavily-overlapping spheres (common with `n_max` > 1, by design) multiply memory. See the `build_kv_cache` blowup above. |
| `CacheSimulator`'s per-query hit check (`lemma1_no_union_sigma` / `_cig_distances`) | Iterates `cache.get_all_entries()` (a Python list) per query, calling `angular_distance` once per (query, entry) pair | The dominant hotspot this round. Each call re-normalizes and re-computes `arccos`/`clip` from scratch rather than using any precomputed, batched matrix operation across all cache entries at once (contrast with `KVCache`'s already-existing `top_k_matrix`/O4 stacked-matrix optimization used elsewhere, per code comments in `kv_cache.py`, which this per-query loop does not appear to exploit for the top-level query-to-cache-entry check itself). |

## Prioritized optimization punch-list for next round

Ranked by measured time/memory share this round, worst first:

1. **`CacheSimulator.run_query`'s per-(test-query, cache-entry) distance-call
   overhead.** By far the largest single cost observed (up to 154.6s vs <10s
   for every other stage combined, at the same scale). Batch the per-entry
   angular-distance check across all cache entries into one matrix operation
   per query (matching the vectorized pattern already used in
   `_pairwise_cache_dists`/`_base_dists_for_query` in `population.py`), rather
   than a Python-level loop over `cache.get_all_entries()` with one
   `angular_distance` call per entry.
2. **`build_kv_cache`'s redundant per-entry matrix duplication when
   `R_c`s overlap heavily.** Measured at 904.6 MB for just 6,159 entries
   against a 4,096-vector base -- a ~100x memory amplification versus the
   budget's nominal footprint. Consider deduplicating shared base-vector
   matrix rows across cache entries (e.g. a single shared normalized base
   matrix plus per-entry index lists, rather than each entry materializing
   its own copy), or capping/rethinking the deferred-flush step in
   `_select_greedy_cost` that is the direct cause of `n_selected` exceeding
   `n_base`.
3. **`select_candidates`'s per-iteration Python `set` difference
   (`covered_train_indices - covered_train_set`)**, confirmed as a real (if
   currently smaller) cost that will dominate at higher `n_max`/candidate
   counts than tested this round (3.15s already at 51,200 candidates).
   Converting `covered_train_indices`/`covered_test_indices` to boolean numpy
   arrays is the concrete next step, per the data-structure audit above.
4. **`propose_candidates`'s memory footprint at high `n_max`** (339.6 MB at
   n_cache=1024/n_max=50) is proportional to candidate count as expected, not
   a bug -- but worth keeping in mind alongside (2): a large `n_max` multiplies
   both proposal memory and downstream cache-build duplication together.

Items (1) and (2) should come first next round -- they dominate by one to two
orders of magnitude over everything else measured, and (3) was the originally
suspected priority but was empirically smaller in this round's grid.

## Caveats

- Synthetic clustered data, not real SIFT/ESCI vectors -- absolute hit rates
  and exact timings will differ on real data, but the *relative* cost
  ordering between stages (simulate/build_kv_cache >> propose >> select >>
  generate_benchmark) should hold, since it follows from algorithmic
  complexity (loop nesting and per-call overhead), not data distribution.
- `n_cache=4096` was not reached; the scaling trend from 256->1024 (worse
  than quadratic for `simulate`) suggests it would take on the order of tens
  of minutes to hours for a single configuration under the current
  implementation, which is why it was excluded rather than run to completion.
- Budget fraction 0.1 never selected anything in this synthetic setup, so
  timing/memory at very small budgets is not represented above (those stages
  are effectively free when nothing is selected, unsurprisingly).

## After optimization (Phase 3 Round 2)

Round 2 implemented the punch-list above, but the root causes turned out more
specific than the punch-list guessed (found via code exploration before
writing any fix, not by guessing from the symptoms). Raw data:
`simulations/population/profiling_grid_results_after.json` (same seed/config
grid as the "before" data in `profiling_grid_results.json`).

**P0 -- simulate() was a dead-code problem, not a missing-feature problem.**
`CacheSimulator` always wrapped its distance function in a
`DistanceCalculationTracker` (for a `distance_calculations` call-count metric)
and always passed that tracker into `lemma1_no_union_sigma`/
`lemma2_no_union_sigma`. Every vectorized fast path already written in
`utils/base/lemmas.py` (`_cig_distances`'s matmul, `_batch_distances`'s
vectorized branch) explicitly gates on `distance_func is raw_distance_func` --
a condition that was *never true* once wrapped, so those fast paths were
silently dead in every real run. Separately, `KVCache.get_d_Q_batch()` /
`build_query_matrix()` -- a batched replacement for the per-entry
`D(query, entry.query)` gate check -- was fully implemented but never called
by anything. Per the user's direction, distance-call counting was dropped
(not preserved through the optimization; `QueryResult.distance_calculations`
is now always `None` -- it can be reintroduced later in a properly batched
form if needed) and `get_d_Q_batch` was wired into both active algorithms via
a new `precomputed_d_Q` parameter.

**P1 -- the ~900MB blowup was a pointer-vs-copy problem.**
`CandidateEntry.to_cache_entry()` now produces index-only `CacheEntry`
objects (`top_k_vectors=[]`, `top_k_matrix=None`, only `top_k_indices` set)
instead of each candidate materializing its own normalized-matrix copy of
its (frequently heavily-overlapping) `R_c` set. Distance computation and
result-vector lookup go through one shared, pre-normalized `base_hat` matrix
and the original-scale `base_vectors` array, both computed once in
`CacheSimulator.__init__` and threaded through to `lemma1_no_union_sigma`/
`lemma2_no_union_sigma`/`_cig_distances`. Scoped to the two active algorithms
only -- the deprecated union-based algorithms were left untouched, as agreed.

**P2 -- set-to-array conversion, with one internal correction.**
`covered_train_indices`/`covered_test_indices` are now fixed-size numpy
boolean arrays instead of Python `set[int]`, since the domain size
(`n_train_queries`/`n_test_queries`) is known up front and Python
set-of-boxed-ints has real per-element overhead independent of fill ratio.
The first implementation stacked all active candidates' masks into one
`(n_active, n_train)` matrix per outer-loop iteration to vectorize the
"newly covered" computation across candidates at once -- measured to *increase*
peak memory (up to ~107MB transient) with no time benefit at n_active~50k, so
it was replaced with a simple per-candidate `count_nonzero(mask & ~running)`
loop (still array-based, just not stacked). This is a reminder that
"vectorize across candidates" isn't automatically a win once the stacked
matrix itself becomes the larger allocation.

### Before / after, matching combos (budget fraction=0.3)

| method | n_cache | n_max | n_selected | select_candidates | build_kv_cache (peak MB) | simulate |
|---|---|---|---|---|---|---|
| point | 256  | 5  | 91    | 0.11s -> 0.07s | 3.3 -> 0.4     | 3.79s -> **0.05s** (76x) |
| point | 256  | 50 | 811   | 0.51s -> 0.55s | 29.4 -> 3.5    | 3.98s -> **0.20s** (19x) |
| point | 1024 | 5  | 435   | 0.30s -> 0.33s | 64.0 -> 8.6    | 154.6s -> **0.86s** (180x) |
| point | 1024 | 50 | 6,159 | 3.15s -> 3.51s | 904.6 -> 121.3 | 141.3s -> **6.60s** (21x) |
| ball  | 256  | 5  | 68    | 0.06s -> 0.06s | 2.5 -> 0.3     | 2.39s -> **0.04s** (53x) |
| ball  | 256  | 50 | 1,532 | 0.50s -> 0.57s | 55.5 -> 6.6    | 7.35s -> **0.39s** (19x) |
| ball  | 1024 | 5  | 340   | 0.26s -> 0.29s | 50.0 -> 6.7    | 124.6s -> **0.68s** (183x) |

`select_candidates` time is essentially unchanged, as expected -- P2 improved
the per-comparison constant factor (numpy array op vs Python set op), not the
`O(iterations * active_candidates)` rescan complexity in `_select_greedy_cost`
(every outer-loop iteration re-scans every still-active candidate). That
algorithmic issue was the *originally* suspected top priority and remains
unaddressed; it's just no longer the practical bottleneck at the scales
tested, since `simulate()`/`build_kv_cache` dropped by 1-2 orders of
magnitude and now sit below or near `select_candidates`' cost instead of
1-2 orders of magnitude above it.

### Remaining opportunity, noted but not implemented this round

`build_kv_cache`'s memory (121.3 MB at n_cache=1024/n_max=50/B=1228, versus a
theoretical few-MB floor for ~4,096 distinct base vectors) is now dominated by
`top_k_indices` being stored as Python lists of ints rather than numpy int32
arrays -- at ~3.15M total stored indices across 6,159 entries, Python int
list overhead (tens of bytes per boxed int) accounts for the gap. Converting
`top_k_indices` to numpy arrays would plausibly cut this further to the
low tens of MB, matching int32's 4-bytes/index floor. Not implemented this
round (found during re-profiling, out of the approved plan's scope) --
flagged here for the next round rather than expanded into silently.

### select_candidates' remaining complexity (unaddressed, for next round)

If a future round wants to address `select_candidates` directly (now the
largest single-stage cost at the scales tested, ~3.5s at 51,200 candidates),
the target is `_select_greedy_naive`/`_select_greedy_cost_static`/
`_select_greedy_cost`'s outer `while` loop re-scanning every still-active
candidate on every iteration to find the next best one -- an
`O(iterations * active_candidates)` pattern independent of representation.
A priority-queue or incremental-update approach (only re-scoring candidates
whose coverage actually changed after each selection) would address the
complexity itself, not just the per-comparison constant factor P2 improved.

## Real-benchmark-scale profiling (Phase 3 Round 3)

Everything above was synthetic, topping out at `n_base=4096`/`n_cache=1024`.
The real research benchmarks this pipeline actually serves are far larger and
had never been profiled: `n_base=50,000` (a subsample of the true SIFT1M /
ESCI ~1.82M catalogs), `n_cache` up to 2048, `n_max=20`, `K` up to 100. This
round profiled against the real, already-prepared SIFT benchmark
(`sift_b50k_c2048k100_t512n20_s42`, the largest available), via a new
`on_stage` timing callback added to `population_runner.py::run_population_benchmark`
itself (not a synthetic mock) and a driver script,
`simulations/population/profile_real_benchmark.py`.

**Methodology note on reliability**: initial attempts to monitor long-running
background profiling jobs via periodic self-scheduled check-ins were
unreliable (checks did not fire as expected, and elapsed-time tracking drifted).
Switched to direct `Bash` tool timeouts and explicit `ps`/kill checks instead,
which is why some data points below are single-shot runs rather than a full
swept grid -- the largest budget fraction was time-boxed and killed rather
than run to completion once it clearly exceeded reasonable bounds.

### First bugs found via this fresh sweep

- Three call sites still did `len(cand.covered_train_indices)` expecting the
  pre-Round-2 `set[int]` semantics; against Round 2's fixed-size boolean
  array this always returned the constant array length instead of the true
  coverage count -- silently wrong (`coverage_rate` was always exactly `1.0`).
  Fixed in `simulations/population/microbenchmarks/microbenchmark_candidates.py`,
  `microbenchmark_k.py`, and `simulations/population/population_runner.py`'s
  `_write_selection_log`. Regression-guard assertions added to
  `tests/population/microbenchmarks/test_microbenchmark.py`/
  `test_microbenchmark_k.py` so a coverage-rate-always-1.0 bug can't silently
  pass a range-only check again.
- `lemma2_no_union_sigma` branched only on `entry.top_k_vectors` (unlike
  `_cig_distances`, which checks both `top_k_matrix` and `top_k_vectors`) --
  made symmetric as a defensive fix; no known live call site was actually
  hitting the wrong branch, but it's a fragility worth closing before scaling
  further.

### Real per-stage numbers (SIFT, `point`/`ball`, `k_test=5`, `n_max=20`)

| n_cache | method | budget (fraction) | load_benchmark | load_gt | propose | propose peak MB | select | build | sim_oracle | sim_non_oracle | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256  | point | 2,500 (0.05)  | 0.30s | 0.14s | 1.70s  | 618   | 1.43s  | 0.00s | 0.69s | 0.34s | 4.76s |
| 2048 | point | 2,500 (0.05)  | 0.90s | 1.14s | 11.28s | 3,734 | 6.45s  | 0.00s | 0.60s | 0.29s | 21.4s |
| 2048 | point | 15,000 (0.3)  | --    | --    | 12.10s | 3,734 | 19.03s | 0.03s | 1.79s | 0.61s | 36.6s |
| 2048 | point | 50,000 (1.0)  | --    | --    | (not reached -- killed) | -- | **not completed in 10 min** | -- | -- | -- | -- |
| 2048 | ball  | 15,000 (0.3)  | --    | --    | 15.92s | 3,722 | 14.95s | 0.04s | 2.63s | 0.88s | 38.3s |

(`--` = not separately re-measured in that run; `propose`/`load` costs are
per-call, not cumulative -- each row except the two at `n_cache=2048` sharing
the same process was a fresh process, so `propose` was recomputed each time.)

### Headline finding: `select_candidates` doesn't just get slower at scale -- it hits a different regime

Going from budget 2,500 to 15,000 (6x), `select_candidates` grew from 6.45s
to 19.03s (~3x) -- sub-linear, in line with what the synthetic data predicted.
Extrapolating that trend to the full budget (50,000, ~3.3x more than 15,000)
would suggest roughly 40-60s. **The actual run was killed after 10 minutes
without `select_candidates` completing even once.** This is not a smooth
continuation of the smaller-budget trend -- it's a different cost regime, and
the likely mechanism is structural, not just "more work": `_fits_budget`
(`utils/base/selection.py`) rejects a candidate outright once it can't fit in
the remaining budget, which is what keeps the *active* candidate set shrinking
quickly at small budgets. At `budget_fraction=1.0`, the budget equals
`n_base` exactly, so `_fits_budget` almost never rejects anything on capacity
grounds -- the *only* thing that ever prunes the active set is a candidate's
`covered_train_indices` becoming fully redundant (`keep = newly > 0`), which
requires enough *other* candidates to have already been selected to cover
every training query that candidate would have covered. With ~40,960
candidates and 2,048 train queries, that pruning path is far slower to kick
in than budget-based rejection. The result: the active set likely stays close
to its full ~41K size for a large number of outer-loop iterations, each
re-scanning nearly all of them -- the exact `O(iterations * active_candidates)`
complexity flagged as unaddressed after Round 2, now visibly the dominant
cost at the budget size that actually matters for a real deployment (a full
memory budget, not a token slice of it).

**This should be the top priority for the next implementation round** --
ahead of any further constant-factor work -- since it's not a matter of
"make it faster," it's currently a matter of "does not finish in a
reasonable time at all" for the most realistic budget setting.

### `propose_candidates`' memory at real scale: much larger than synthetic data predicted

At `n_cache=2048`, `propose_candidates` used **3.7 GB**, against a peak of
`339.6 MB` for the largest synthetic case tested (`n_cache=1024`, `n_max=50`,
`n_base=4096`) -- roughly 11x more memory despite a smaller `n_max` (20 vs
50) and only 2x more `n_cache`. The synthetic tests never varied `n_base`
much (topped out at 4,096); the real benchmark's `n_base=50,000` (12x larger)
is the dominant factor, since each candidate's `R_c` size (and therefore its
stored `int32` index array) scales with how much of the *base* a sphere of a
given radius covers, not just with `n_cache`/`n_max`. This means the
synthetic profiling grid systematically underestimated memory pressure at
real scale by focusing scaling experiments on `n_cache`/`n_max` while holding
`n_base` nearly fixed. A future profiling round should treat `n_base` as a
first-class scaling axis, not a fixed backdrop.

Both `point` and `ball` show essentially the same order of magnitude for
`propose`/`select`/memory at matching scale (`ball` is not meaningfully worse
than `point` here) -- the earlier suspicion that `ball` might cost
substantially more than `point` is not borne out at this scale.

### ESCI and full-scale extrapolation

ESCI was not separately run this round (time-boxed out after the SIFT
`point`/`ball` data above and the killed full-budget run already consumed
the available session time) -- ESCI's research benchmarks share the same
`n_base=50,000`/`n_cache=2048`/`n_max=20`/`K` scale as SIFT's, so the same
qualitative findings (propose memory, the `select_candidates` regime change
at large budgets) are expected to carry over; only absolute hit rates would
differ due to the different underlying data distribution.

Ground-truth *generation* (as opposed to loading -- `prepare_benchmark.py`'s
brute-force KNN, not re-run this round since the benchmarks are already
prepared on disk) was estimated by extrapolating the already-measured
synthetic `generate_benchmark` scaling (linear in `(n_cache + n_test) *
n_base`, calibrated at `0.31s` for `(1024+512)*4096` units in Round 1):

| Scale | n_base | Estimated GT-generation time |
|---|---|---|
| Real 50K subsample (current) | 50,000 | ~6.3s |
| Full SIFT1M | 1,000,000 | ~126s (~2.1 min) |
| Full ESCI ~1.82M | 1,820,000 | ~230s (~3.8 min) |

At the current 50K-subsample scale, GT generation is not a bottleneck (single
digits of seconds). If the base sample size were ever increased toward the
true full-catalog scale, it would become a multi-minute one-time cost per
benchmark -- worth knowing, but dwarfed by `select_candidates`' current
non-completion at full budget, which is the real blocker.

### Updated punch list for the next round

1. **Fix `select_candidates`' large-budget regime change** (new top
   priority, supersedes the Round 2 punch list's #3). The `O(iterations *
   active_candidates)` rescan is no longer just "slower than it could be" --
   at `budget_fraction=1.0` on real data it does not complete in 10 minutes.
   Needs an algorithmic fix (e.g. incremental re-scoring of only the
   candidates whose coverage actually changed after each selection, or a
   priority-queue-based approach), not a constant-factor optimization.
2. **Re-scope memory profiling around `n_base`, not just `n_cache`/`n_max`.**
   The 3.7 GB `propose_candidates` figure at real scale was not predicted by
   any synthetic experiment run so far, because `n_base` was held small and
   fixed throughout Round 1/2's synthetic grid. `R_c`'s per-candidate size
   (and therefore total proposal memory) scales with `n_base`, independent of
   `n_cache`/`n_max`. A future profiling grid should vary `n_base` directly
   rather than holding it fixed.
3. Once (1) is fixed, re-run the full budget-fraction sweep (including 1.0)
   against real SIFT and ESCI benchmarks for both `point` and `ball` to get
   a complete, real end-to-end timing table -- this round only established
   that the current implementation cannot produce that table yet at the
   largest budget.

## Round 4: select_candidates vectorization + R_c memory fix

Follow-up round targeting the two Round 3 findings directly: `R_c` changed
from an `np.int32` index array to a boolean mask over `n_base` (unifying its
representation with `cached_mask`), and the per-iteration active-candidate
rescan in `greedy_naive`/`greedy_cost_static`/`greedy_cost`
(`utils/base/selection.py`) was vectorized. Per user direction, this was
required to produce **exactly the same selection output** as before (no
lazy-heap/CELF-style approximation) -- verified by a new cross-check test
(`tests/utils/test_selection.py::test_vectorized_matches_naive_reference`)
comparing the production implementation against a deliberately naive
reference across 5 seeds x 3 strategies x 3 budgets on ~40% overlap-density
random candidates, matching the real-data density finding from Round 3.

### The vectorization design took two iterations to get memory-safe

The first vectorized design maintained a *persistent* stacked 2D matrix per
strategy call, shrunk every iteration via boolean-mask indexing
(`matrix[keep]`) in lockstep with the candidate list. This was **measured**,
not assumed, against real data (`sift_b50k_c2048k100_t512n20_s42`,
`budget_fraction=0.05`): `select_candidates` peaked at **6.1 GB** -- worse
than the original unvectorized implementation's near-zero overhead -- while
wall time barely improved (6.45s -> 6.50s, i.e. no real speedup). Root cause:
`matrix[keep]`-style filtering creates a full copy on *every* filter step,
and there were 2-3 such steps per outer iteration (fits-check, newly>0
pruning, best-candidate removal), so multiple large transient copies could
coexist before Python's garbage collector reclaimed the previous one.

The design was reworked to never persist or incrementally shrink a stacked
matrix at all: each iteration builds a *fresh* stack from whatever the
already-Python-list-filtered working set currently is, uses it for exactly
one vectorized computation, and discards it. Two further refinements specific
to each strategy's actual data dependencies:
- `greedy_naive`/`greedy_cost_static` only need `R_c` for the budget
  fits-check (their *scores* depend on `covered_train_indices` and, for
  `cost_static`, the scalar `rc_size` -- never `R_c` itself), so the
  expensive `R_c`-touching check now only runs for the subset that fails the
  cheap `rc_size` short-circuit -- typically small at large budgets, where
  most candidates trivially fit.
- `greedy_cost` needs exact marginal cost (`R_c`-derived) for scoring
  regardless of fits-eligibility, so it's now computed once per iteration and
  reused for both the fits-check and the score, removing a redundant
  duplicate computation the original per-candidate-loop version had (compute
  once in `_fits_budget`'s fallback path, compute again for the marginal
  score line).

This brought peak memory for `select_candidates(B=2500)` down to **4.1 GB**
(from 6.1 GB) -- still far above the original's near-zero overhead, because
building even one fresh stack of all ~41,000 candidates' `R_c` masks on the
*first* iteration (before anything is filtered) is an unavoidable ~2 GB
allocation on top of the ~2.26 GB the candidates already occupy from
`propose_candidates`. This is a real, currently-unresolved memory cost of
correctness-preserving vectorization at this candidate-set scale, not fully
solved this round -- flagged in the punch list below.

### Real-scale results (SIFT, `point`, `k_test=5`)

| budget (fraction) | select_candidates (before -> after) | peak MB (before -> after) | hit rate (oracle / non-oracle) |
|---|---|---|---|
| 2,500 (0.05)  | 6.45s -> **5.23s** | 0.45 -> 4,097 | 1.27% / 0.20% (unchanged from Round 3) |
| 15,000 (0.3)  | 19.03s -> **13.24s** | 4.08 -> 4,097 | 4.64% / 1.56% (unchanged from Round 3) |
| 50,000 (1.0)  | not completed in 10 min -> **not completed in ~8.5 min** | -- | -- |

Hit rates at 2,500 and 15,000 are byte-identical to Round 3's pre-vectorization
numbers, confirming the "same output, just faster" requirement held on real
data, not just in the unit-test cross-check.

`propose_candidates` (unrelated to the select_candidates vectorization, but
measured in the same runs, from the `R_c` int32-to-bool-mask change alone):
**3,734 MB -> 2,261 MB** (~1.65x reduction) at `n_cache=2048`, matching the
predicted crossover math from Round 2's data-structure audit (int32 loses to
a plain bool array above 25% fill density; real `R_c` density measured at
~44%).

### Headline finding: budget_fraction=1.0 still does not complete

**This is the honest, disappointing result**: even after vectorization,
`budget_fraction=1.0` was killed after ~8.5 minutes without
`select_candidates` finishing -- an improvement in wall-clock *constant
factor* at smaller budgets (~20-30% faster), but not a fix to the underlying
`O(iterations * active_candidates)` **complexity class** at the budget size
that actually matters for a real deployment. The reasoning from Round 3 still
applies: at `budget_fraction=1.0`, `_fits_mask`'s capacity check almost never
rejects anything, so the active candidate set stays close to its full ~41,000
size for many iterations, and rebuilding even a *fresh, non-persistent* stack
of that size every iteration is itself expensive at real-data density (~2 GB
per rebuild for `R_c` alone) -- vectorization removed Python-loop overhead
per element, but the sheer number of (large rebuild x many iterations)
combinations at this budget size is still too costly.

### Updated punch list for the next round

1. **`select_candidates` at `budget_fraction=1.0` remains unsolved** -- this
   is still the top priority, now with two rounds of evidence that
   constant-factor fixes (Round 4) are not sufficient; the fix needs to
   change the *complexity class* itself. Two directions considered but not
   implemented, both requiring explicit user sign-off given they change
   behavior or algorithm structure:
   - A lazy-heap (CELF-style) approximation was explicitly declined this
     round due to `greedy_cost`'s benefit/cost ratio not being provably
     monotonic (see this doc's Round 4 planning context) -- but could be
     revisited if empirically shown to (almost) always agree with the exact
     algorithm on real data, trading a formal guarantee for a large
     practical speedup.
   - A fundamentally different strategy for the large-budget regime (e.g.
     batch/chunked processing of candidates to bound per-iteration memory,
     independent of whether the complexity class itself improves) was
     identified as a future direction but not implemented.
2. **The vectorized `select_candidates`' own memory floor (~4.1 GB at
   real scale)** is a new cost introduced by this round's fix, not present
   in the original implementation. Worth understanding whether this is
   tolerable given `propose_candidates` already peaks at ~2.26 GB in the same
   process (concurrent peak could approach the low single-digit GB range,
   verified safe on the 8 GB test machine but worth checking on smaller
   deployment targets).
3. Re-run the full budget-fraction sweep (including 1.0) against real SIFT
   and ESCI benchmarks for both `point` and `ball` once (1) is actually
   resolved -- still blocked, now for the second round in a row.

## Round 5: unattended benchmarking sweep

A 30-task, self-timing sweep (`simulations/population/overnight_sweep.py`,
each task run as a `subprocess.run(..., timeout=...)` so a hang is killed by
Python itself) ran unattended for ~3 hours against the existing 50K SIFT/ESCI
benchmarks and the synthetic harness. Full raw results:
`simulations/population/overnight_sweep_results.jsonl`. This round is
measurement-only; no production code changed except adding `--strategy` and
`--profile-stage select` flags to `profile_real_benchmark.py`.

### Correction to Round 4: budget_fraction=1.0 does not hang -- it gets
### killed by memory pressure, and only after select_candidates finishes

Round 4 reported `budget_fraction=1.0` as "not completed in ~8.5 min" and
attributed it entirely to `select_candidates`. That framing was incomplete.
This round's evidence, from four independent `budget_fraction=1.0` runs
(`point`/`greedy_cost`, `ball`/`greedy_cost`, ESCI/`greedy_cost`, and a
cProfile-instrumented rerun of the first), all ending in `returncode=-9`
(SIGKILL, consistent with this 8 GB machine's memory-pressure kill) between
1065s and 1170s:

- `select_candidates` **does finish** at `budget_fraction=1.0` -- the
  cProfile rerun shows it completing in 456s (with profiler overhead; ~472s
  unprofiled, inferred from log timestamps in the other three runs), with
  13,855-19,263 entries selected depending on dataset/method.
- The process is then killed **later**, during `build_kv_cache` or
  `simulate_oracle` on that much larger (13K-19K entry) cache -- consistent
  with `simulate_oracle`'s cost scaling sharply with cache size (see below),
  compounding on top of `select_candidates`' already-elevated ~4.1 GB peak
  from Round 4.
- The cProfile breakdown of the 456s `select_candidates(B=50000)` call
  confirms Round 4's original diagnosis precisely: 491 iterations each
  rebuild a fresh `R_c_matrix` (`np.array` over ~tens of thousands of active
  candidates, 129s cumulative across those calls) and reduce it
  (`count_nonzero`/`sum`, 150.8s cumulative) -- the active set never shrinks
  because `_fits_mask`'s capacity check almost never rejects at this budget,
  exactly as predicted.

Net effect: this is still an unsolved complexity-class problem, but the
actual failure mode is memory pressure from the combination of
`select_candidates`'s peak plus a much larger resulting cache feeding into
`simulate_oracle`, not an infinite or near-infinite loop in selection alone.

### New finding: simulate_oracle, not select_candidates, dominates total time from budget_fraction~0.5 upward

The budget-fraction knee curve (SIFT, `point`, `greedy_cost`,
`sift_b50k_c2048k100_t512n20_s42`, `k_test=5`):

| B (fraction) | select_candidates | simulate_oracle | simulate_non_oracle | total | status |
|---|---|---|---|---|---|
| 15,000 (0.3) | 14.3s | 1.8s | 0.6s | 28.5s | ok |
| 20,000 (0.4) | 51.9s | 3.6s | 1.2s | 70.5s | ok |
| 25,000 (0.5) | 36.2s | 73.3s | 21.9s | 140.3s | ok |
| 30,000 (0.6) | 26.9s | 160.3s | 55.9s | 253.2s | ok |
| 35,000 (0.7) | 30.8s | 162.8s | 57.7s | 261.5s | ok |
| 40,000 (0.8) | 48.4s | 199.2s | 75.4s | 334.0s | ok |
| 45,000 (0.9) | 79.5s | 210.1s | 83.2s | 383.9s | ok |
| 50,000 (1.0) | ~472s | (not reached) | (not reached) | killed at 1146s | SIGKILL |

Two distinct scaling problems are visible, not one:

1. `select_candidates` stays under 80s through `budget_fraction=0.9`, then
   jumps ~6x for the last 10% of budget (79.5s -> ~472s) -- a sharp
   discontinuity right at the edge, not a smooth curve. This matches the
   `_fits_mask` capacity-check theory: rejection only becomes rare in the
   last stretch before the budget is exhausted.
2. `simulate_oracle` (+ `simulate_non_oracle`) grows smoothly and much
   earlier -- a 20x jump between 0.4 and 0.5 (3.6s -> 73.3s), then continues
   climbing, and **is the larger cost at every budget from 0.5 upward**
   (76% of total time at 0.9). This was not identified in Round 3 or 4,
   which both focused exclusively on `select_candidates`. It was not
   root-caused this round (out of scope -- this was a measurement pass) but
   is now the bigger open question for the next round, arguably ahead of
   `select_candidates` itself given it dominates wall-clock across most of
   the useful budget range.

### Strategy comparison: greedy_naive and greedy_cost_static are a viable workaround at budget_fraction=1.0; random is not

| Strategy | B=25,000 (0.5) | B=50,000 (1.0) |
|---|---|---|
| `greedy_cost` | 140.3s (see table above) | SIGKILL at 1146s |
| `greedy_naive` | 119.8s | **140.3s, completes cleanly** |
| `greedy_cost_static` | 27.8s | **138.6s, completes cleanly** |
| `random` | timeout at 1200s (stuck in `simulate_oracle`, past a completed `build_kv_cache`) | timeout at 2700s |

This is an actionable, previously-unknown result: **`greedy_naive` and
`greedy_cost_static` both complete at `budget_fraction=1.0` in ~140s**,
roughly 8x faster than `greedy_cost` even manages to get through just
`select_candidates` before being killed. Both strategies only need `R_c` for
the (cheap, short-circuited) fits-check rather than for scoring every
candidate every iteration, per Round 4's vectorization work -- this is
exactly the mechanism that avoids `greedy_cost`'s wall. If a near-term
production workaround is needed before `greedy_cost`'s complexity-class fix
lands, switching strategy at large budgets is a real, measured option, not
just a theoretical one.

`random` is the opposite story: its `select_candidates` call is fast (2.6s,
confirmed via the Round 4 scalar-path code), but it selects far more,
less-coverage-efficient entries into the cache, and the resulting larger
cache makes `simulate_oracle` itself the bottleneck -- it timed out at both
budgets tested without finishing simulation. (stdout was empty for the
`budget_fraction=1.0` random run because the child's stdout was block- not
line-buffered and lost on kill; the 0.5 run's partial output was enough to
localize the stall to `simulate_oracle`.)

### ball method and ESCI: same failure pattern as point/SIFT

`ball` (SIFT, `greedy_cost`): 19.4s at 0.3, 128.6s at 0.5, SIGKILL at 1065.6s
for 1.0 -- same shape as `point`, confirming Round 3's "ball behaves like
point" finding still holds, now including the failure mode at
`budget_fraction=1.0`.

ESCI (`point`, `greedy_cost`): 44.5s at 0.3, SIGKILL at 1169.7s for 1.0 --
same pattern, confirming the finding transfers across datasets (expected,
since ESCI and SIFT share the same downstream `select_candidates`/simulate
code).

### Synthetic n_base scaling: the wall is driven by candidate count, not n_base

Fixed `n_cache=256` (-> 5,120 total candidates, deliberately bounded to keep
memory safe at large `n_base`), swept `n_base` from 4,096 to 200,000 (~50x):

| n_base | select(B=1.0) | build_kv_cache(B=1.0) | simulate(B=1.0) | n_selected |
|---|---|---|---|---|
| 4,096 | 0.13s | 0.21s | 0.10s | 1,395 |
| 16,384 | 0.33s | 1.36s | 0.26s | 2,014 |
| 50,000 | 0.88s | 3.46s | 0.72s | 1,842 |
| 100,000 | 2.07s | 8.26s | 0.94s | 2,154 |
| 200,000 | 5.80s | 13.13s | 2.79s | 1,619 |

No wall at any scale -- everything grows smoothly and stays well under a
second to low tens of seconds even at 200,000 base vectors, a 4x larger base
than the real SIFT/ESCI benchmarks. This is the key clarifying result: the
real benchmarks have `n_cache=2048` (-> ~41,000 candidates, 8x the synthetic
grid's fixed 5,120), and it's **candidate count (driven by n_cache, i.e. the
number of cache queries), not n_base**, that determines whether
`select_candidates` hits the `budget_fraction=1.0` wall. `n_base` mainly
affects per-candidate `R_c` memory (linear, already well understood since
Round 4), not the iteration-count blowup.

### Audit items: dataloaders.py hot spots are not actually hot

Both suspected hot spots from this round's pre-sweep audit turned out to be
negligible in practice:

- `load_sift_dataset` (the `load_fvecs`/`load_ivecs` per-vector Python
  unpack loop): 8.4s to load the full 1M-vector raw SIFT files (base + learn
  + query + groundtruth). This function is only called from
  `prepare_benchmark.py`, i.e. once per benchmark preparation, never in the
  per-run hot path -- not worth optimizing.
- `populate_cache_from_precomputed_population`: 0.10s for SIFT and 0.10s for
  ESCI at 2,048 cache entries. Negligible; the per-query Python loop and
  nested list comprehension flagged in this round's audit are not a
  meaningful cost at this scale.
- `run_hit_policy.py --debug` (5 queries) and the full `run_synthetic.py`
  (192 synthetic datasets) both completed without incident (0.5s and 163.7s
  respectively), confirming both share the already-optimized
  `CacheSimulator` hot path with no separate slow path of their own.

Recommendation: no action needed on either dataloader item; remove them from
future audit lists.

### Updated punch list for the next round

1. **`simulate_oracle`'s scaling (new, from this round) is now at least as
   urgent as `select_candidates`'s complexity-class problem** -- it dominates
   wall-clock from `budget_fraction=0.5` upward and was not profiled this
   round (only stage-level timing, not a cProfile breakdown). Next round
   should cProfile `simulate_oracle` at a mid-range budget (e.g. 0.5-0.7,
   where it's already the dominant cost but the process doesn't OOM) the
   same way this round profiled `select_candidates`.
2. **`select_candidates` at `budget_fraction=1.0` is now understood to
   complete (in ~470s) but be followed by a memory-pressure kill in a later
   stage** -- the fix target shifts from "make selection finish" (it does)
   to "reduce combined peak memory across select + build_kv_cache +
   simulate at large budgets," or reduce the cache size selection produces
   in this regime.
3. **Switching to `greedy_naive` or `greedy_cost_static` at large budgets is
   a validated, measured short-term workaround** (~140s vs. OOM-kill) if a
   production need for `budget_fraction=1.0` arises before the underlying
   fix lands -- worth flagging to anyone consuming this pipeline now, not
   just deferring to "the next round."
4. `random`'s poor scaling (timeout, not OOM) at large budgets is a
   consequence of selecting a much larger, less coverage-efficient cache,
   not a bug -- expected given it ignores cost/coverage entirely. Not worth
   further investigation unless `random` becomes a real strategy candidate.
5. Re-run the full budget-fraction sweep (including 1.0) against real SIFT
   and ESCI benchmarks once (1)/(2) above are resolved -- carried over from
   Round 4's punch list, now with a clearer target (memory, not an infinite
   loop).

## Round 6: fixing greedy_cost at high budgets (algorithm + memory)

Round 5 left `greedy_cost` -- the only selection strategy actually used in
practice -- SIGKILLed at `budget_fraction=1.0`, plus a second scaling problem
in `simulate_oracle`. This round implemented three independent,
correctness-preserving fixes plus one more discovered during verification,
all confirmed byte-identical against pre-fix output on real data.

1. **`_select_greedy_cost` (`utils/base/selection.py`)**: replaced the
   per-iteration full recompute of `marginal`/`newly` (a fresh
   `(|active|, n_base)` matrix build + reduce, ~491 times at
   `budget_fraction=1.0`) with incremental delta maintenance --
   `cached_mask`/`covered_train_mask` only grow, so each iteration only
   needs to subtract the bits the just-selected candidate newly
   contributed, not recompute from scratch. Numerically exact (integer
   popcounts, no reassociation).
2. **`CandidateEntry.to_cache_entry()` (`utils/base/population.py`)**:
   `top_k_indices` changed from a Python list of boxed ints
   (`np.where(...)[0].tolist()`) to a compact `np.int32` array
   (`np.flatnonzero(...).astype(np.int32)`) -- ~9x memory reduction,
   directly fixing the SIGKILL (was ~11-15GB across ~14-19K selected entries
   at real scale).
3. **`_cig_distances`/`lemma1_no_union_sigma` (`utils/base/lemmas.py`,
   `lemmas_sigma.py`)**: added an optional `precomputed_base_dist` parameter
   -- the full per-query distance-to-base-vector array is now computed once
   per query (in `lemma1_no_union_sigma`, mirroring the existing
   `get_d_Q_batch` precedent) instead of once per cache entry, replacing a
   redundant `O(rc_size x dim)` matmul per entry with a cheap `O(rc_size)`
   gather. Exact by construction (row-independent matrix-vector product).
4. **`_write_selection_log` (`simulations/population/population_runner.py`)
   -- found only during this round's own re-verification, not part of the
   original plan**: its `min_dist_to_prev` diagnostic field was computed via
   a Python-level `angular_distance()` call for every (step, prior-step)
   pair -- `O(n^2)` Python calls, invisible to any profiled stage since it
   sits between two `on_stage`-wrapped calls. At `budget_fraction=1.0`
   (13,855 selected entries) this alone cost **~900s of untimed wall-clock**
   -- more than every profiled stage combined. Fixed by batching each step's
   distance computation into one vectorized matmul against all
   previously-selected queries (same `O(n^2)` total distance evaluations,
   but via BLAS instead of ~96M individual Python calls).

### Real-data results (SIFT, `point`, `greedy_cost`, `k_test=5`)

| Budget (fraction) | select_candidates | simulate_oracle | Total | vs. Round 5 |
|---|---|---|---|---|
| 25,000 (0.5) | 36.2s -> **23.5s** | 73.3s -> **6.9s** | 140.3s -> **45.6s** | 3.1x faster |
| 45,000 (0.9) | 79.5s -> **74.0s** | 210.1s -> **17.2s** | 383.9s -> **108.9s** | 3.5x faster |
| 50,000 (1.0) | SIGKILL at 1146s (never finished) | never reached | SIGKILL -> **328.0s, completes** | now completes |

Hit rates at 0.5 and 0.9 are **byte-identical** to Round 5's pre-fix numbers
(15.53%/12.70% oracle/non-oracle at 0.5; 48.88%/42.38% at 0.9), confirming
"same output, just faster" held on real data, not just the unit-test
cross-checks (`test_vectorized_matches_naive_reference`,
`test_greedy_cost_incremental_matches_reference_at_high_budget`,
`test_cig_distances_precomputed_base_dist_matches_gather`,
`test_lemma1_no_union_sigma_index_only_matches_real_vectors`,
`test_write_selection_log_matches_naive_reference` -- all new or extended
this round). Full `pytest tests/ -q`: 234 passed.

`select_candidates`' peak memory is unchanged (~4.1GB, the one-time initial
full-matrix pass is still required and wasn't a target this round);
`build_kv_cache`'s peak dropped from an estimated ~11-15GB to ~1.5GB. Peak
process memory across the whole `budget_fraction=1.0` run stayed comfortably
within the 8GB test machine's budget.

### Remaining open items

> **Update (Round 7)**: the first item below (the initial `O(N x n_base)`
> full-matrix pass) has since been eliminated -- see Round 7. The rest were
> still true as of this writing but haven't been re-verified since; treat
> this list as historical unless re-checked against current code.

- `select_candidates`' initial one-time full-matrix pass (before the
  incremental loop takes over) is still `O(N x n_base)` and still peaks at
  ~4.1GB -- not addressed this round; only the *repeated* recomputation was
  eliminated.
- `random`/`greedy_naive`/`greedy_cost_static` were explicitly out of scope
  (not the user's concern) and untouched.
- No further budget-fraction wall was found up to 1.0 for `greedy_cost` on
  the real 50K SIFT benchmark -- the original blocking issue is resolved,
  not just mitigated.

## Round 7: bit-packed delta counting, and the initial full-matrix pass

Two further, unrelated pieces of follow-up work, done together in this
round:

1. **`_select_greedy_cost`'s initial `marginal` computation** (the first
   remaining open item from Round 6) turned out to already be eliminated by
   the time this round started: the initial `marginal` for every candidate
   is simply its precomputed `rc_size` (`cached_mask` is all-zero before the
   loop starts, so `count_nonzero(R_c & ~cached_mask) == rc_size` exactly,
   with no matrix construction needed). The `~4.1GB` one-time `(|active|,
   n_base)` stack Round 6 flagged as unaddressed no longer exists. One
   smaller full-matrix pass remains at initialization --
   `_newly_covered_mask`'s one-time `(|active|, n_train_queries)` stack for
   the initial `newly` counts -- but `n_train_queries` (up to 2048 in this
   codebase's benchmarks) is far smaller than `n_base` (50000), so this is a
   much cheaper remaining cost than the one Round 6 described.
2. **Bit-packed delta counting**: `_chunked_delta_counts` (the incremental
   per-iteration delta-update from Round 6) gained a second code path,
   `np.packbits` plus a 256-entry byte popcount table, selected per chunk
   when the number of newly-set bits being counted (`idx`) is large relative
   to the attribute array's length (`_BITPACK_IDX_FRACTION = 1/20`,
   measured empirically against real SIFT c2048/K=5 data). The existing
   fancy-index gather path (`batch[:, idx]` + `count_nonzero`) stays for
   small `idx` (most calls in a real run), where it wins outright; bit
   packing wins by up to ~12x at the largest real `idx` sizes observed
   (~33000, i.e. late-budget iterations where each pick contributes many
   newly-covered bits and the active set barely shrinks). Both paths verified
   bit-identical via `np.array_equal` in `tests/utils/test_selection.py` --
   a representation/performance choice, not a behavior change.
3. Alongside this, a real bug was fixed in `debug_trajectory.py`'s
   `compute_budget_pct_view`: a `searchsorted(..., side='left')` call
   collapsed all budget-percentage lookups to step 1's under-converged
   value whenever `mem_util` saturated immediately, disagreeing with
   `memory_vs_hitrate.png`'s true 100%-budget values. Fixed to
   `side='right') - 1` semantics (with an explicit `0.0` for unaffordable
   budgets); see `docs/POPULATION_DESIGN.md`'s test-coverage map for the
   regression tests.
