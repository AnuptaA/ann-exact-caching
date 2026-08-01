# Population Pipeline: Design, Theory, and Test Coverage

This is a deep-dive reference for the "population" side of the codebase --
given a query budget, which candidate cache entries should be materialized to
maximize certified hit coverage. It complements `docs/CODEBASE_OVERVIEW.md`
(repo-wide map) and `docs/KNOWN_ISSUES.md` (open items) with detail specific
to population: the theory behind each design choice, the implementation
quirks that theory produces, and what's actually tested.

Every claim below is either a direct quote/paraphrase of a comment or
docstring already in the code, or is marked with an inline callout like the
one below when it's *my* reading of why something is designed the way it is,
rather than something the code states outright. Please read those callouts
critically -- confirming or correcting them is the point of this document.

> **My understanding (please confirm):** this is the format every inferred
> claim uses in this doc.

## 1. Pipeline overview

Four stages, always in this order, implemented once in
`simulations/population/population_runner.py::run_population_benchmark` and
shared by every entry point (`run_population.py`'s CLI sweep, the profiling
scripts, and the e2e tests):

```
propose_candidates_*  ->  select_candidates  ->  KVCache.add_entry (x N)  ->  CacheSimulator.run_query (oracle + non_oracle, per sigma)
   (population.py)         (selection.py)         (kv_cache.py)                (simulate.py, via lemmas_sigma.py)
```

1. **Propose**: given the cache/train queries and their true top-K ground
   truth, produce a large pool of `CandidateEntry` objects -- each one a
   candidate cache center (a query) plus a radius `r_c` and the *complete*
   set `R_c` of base vectors within that radius.
2. **Select**: given a memory budget `B` (in base-vector count) and a
   strategy, greedily choose a subset of candidates that fits `B` while
   maximizing certified query coverage.
3. **Build**: materialize the selected candidates into a `KVCache` (one
   `CacheEntry` per selected candidate).
4. **Simulate**: run `CacheSimulator.run_query` for every query in two modes
   (oracle against the cache's own queries, non_oracle against held-out test
   queries), at every requested `sigma`, recording hit rate and recall.

## 2. Shared certification theory (CIG, HGG, sigma relaxation)

This theory isn't population-specific -- it's shared with the hit-policy
pipeline -- but population's proposal/selection stages are built entirely
around producing candidates that satisfy it, so it has to be understood first.

### CIG (Circular Inclusion Guarantee)

For a cached entry `(Q, R_Q, r_Q)` and a new query `q`, a base vector `v` is
certified correct without touching the database if:

```
D(q, v) <= r_Q - D(Q, q)
```

**Why this is sound (triangle inequality):** for any vector `v`,
`D(Q, v) <= D(Q, q) + D(q, v)`. If `D(q,v) <= r_Q - D(Q,q)`, then
`D(Q,v) <= D(Q,q) + (r_Q - D(Q,q)) = r_Q`. Since `r_Q` is exactly the
distance from `Q` to its K-th (farthest) cached neighbor, anything within
`r_Q` of `Q` is guaranteed to already be in `R_Q` (or closer) -- so `v` is
certified to be one of `q`'s true nearest neighbors without any distance
computation against the rest of the database.

Implementation (`utils/base/lemmas.py::lemma1_no_union`, algorithm 2, the
canonical "no-union" CIG): an "O1 gate" first checks `d_Q <= r_Q` (if
`d_Q > r_Q`, no vector can possibly satisfy the CIG inequality since
distances are non-negative -- skip the entry entirely without touching its
vectors), then a direct threshold mask:

```python
threshold_dist = r_Q - d_Q
passing = np.where(distances_to_q <= threshold_dist)[0]
if len(passing) >= N:
    ...  # certified hit
```

### Binary search removal

`utils/base/lemmas.py:75-95` still defines `binary_search_last_index`, but
its own docstring says plainly: "Kept for backward compatibility / tests.
The hot-path lemma functions use a direct threshold mask (`np.where`)
instead of sorting + binary search."

The old approach relied on distances being pre-sorted, so that
`sorted_distances[mid] + d_q < r_Q` is monotonic and a binary search can find
the cutoff index in `O(log n)`. The replacement,
`passing = np.where(distances_to_q <= threshold_dist)[0]` (used identically
at `lemmas.py:217, 292, 549`), instead vectorizes the membership test across
*all* candidate distances in a single NumPy comparison.

> **My understanding (please confirm):** this trade `O(log n)` binary search
> for `O(n)` vectorized comparison because the hot path checks *every*
> candidate vector in a cache entry against the threshold anyway (there's no
> large pre-sorted structure to search *into* -- you'd have to sort first,
> which is `O(n log n)`), and a NumPy vectorized pass over `n` elements is
> faster in wall-clock than a Python-level binary search loop making `O(log
> n)` individually-interpreted comparisons, since each Python-level
> comparison pays interpreter overhead that a single vectorized comparison
> amortizes away. Only the *count* of `passing` elements is needed to decide
> if the threshold `N` is met; a second `np.argsort` still runs afterward,
> but only over the (typically much smaller) `passing` subset, to pick the
> closest `N` for the actual returned result -- so sorting cost moved from
> "sort everything up front" to "sort only the certified subset, only when
> there's a hit."

### HGG (Half-Gap Guarantee)

If `D(Q, q) < half_gap` (half the gap between the K-th and (K+1)-th true
neighbor of `Q`), the entry's *entire* cached top-K is guaranteed to equal
`q`'s own top-K exactly -- zero per-vector distance computation, but
all-or-nothing on the full `K` (`utils/base/lemmas.py::lemma2_no_union`,
algorithm 4):

```python
if d_Q < half_gap and len(R_Q) >= N:
    ...  # entire cached top-N returned, certified correct
```

> **My understanding (please confirm):** the margin absorbs the maximum
> possible top-K reordering caused by moving the query point from `Q` to
> `q`. Any base vector *outside* Q's true top-K is at least `2*half_gap`
> farther from `Q` than the K-th vector (that's the definition of
> `half_gap`: half the gap to the (K+1)-th neighbor). If `q` is within
> `half_gap` of `Q`, the worst-case perturbation to any distance is
> `+/- half_gap`, which isn't enough to let an outside vector become closer
> to `q` than an inside one was to `Q` -- so the ordering of the top-K is
> provably preserved between `Q` and `q`.

### No-union vs. deprecated union variants

Both CIG and HGG above are **single-cache-entry ("no-union")** guarantees:
one entry alone must supply the full `N` results. Union variants exist in
`lemmas.py` (`lemma1_circular_inclusion` -- "algorithm 1: CIG with union",
`lemma2_half_gap` -- "algorithm 3: HGG with union", `combined_algorithm` --
"algorithm 5: combined with union") and accumulate results across *multiple*
cache entries until `len(R_q) >= N`. They're not deleted, but
`simulations/core/simulate.py` deprecates them one layer up: their imports
and dispatch `elif` branches are commented out, e.g.:

```python
# elif algorithm == "lemma1":
#     # deprecated: union-based CIG. lemma1_sigma(sigma=1.0) is
#     # mathematically identical to lemma1_circular_inclusion.
#     result, is_hit, metadata = lemma1_sigma(...)
```

Calling `run_query` with `algorithm in {"lemma1", "lemma2", "combined"}`
today falls through to `raise ValueError(f"Unknown algorithm: {algorithm}")`
-- they're unreachable via the simulator. Only `"brute"`, `"lemma1_no_union"`,
and `"lemma2_no_union"` are live dispatch targets. Population's own
`run_population.py` only ever uses `ALGORITHMS = ['lemma1_no_union']`.

### Sigma-relaxed no-union variants (always in the dispatch path)

`utils/base/lemmas_sigma.py` defines `lemma1_no_union_sigma` and
`lemma2_no_union_sigma`, which relax the certification threshold from the
full `N` to `ceil(sigma * N)`:

```python
threshold = math.ceil(sigma * N)
...
if len(passing) >= threshold:
    ...
```

`sigma=1.0` gives `threshold = ceil(1.0*N) == N`, so this is algebraically
and behaviorally identical to the exact (non-sigma) functions at the default.
**Critically, `CacheSimulator.run_query` always routes `"lemma1_no_union"`/
`"lemma2_no_union"` through these sigma-aware versions** -- it doesn't even
import the plain functions from `lemmas.py`:

```python
from utils.base.lemmas_sigma import lemma1_no_union_sigma, lemma2_no_union_sigma
```

So callers never choose between a sigma and non-sigma entry point; `sigma`
defaults to `1.0` in `run_query`'s signature and every caller either accepts
that default or explicitly sweeps it. This is what makes population's
`--sigma` sweep (Section 6) possible without any branching in the simulate
call itself -- it's the same code path at every sigma value.

### N=0 (population's containment mode, K_test=0): already correct, no code change needed

`threshold = math.ceil(sigma * N)` is `0` when `N=0`. The existing structure
of `lemma1_no_union_sigma` -- O1 gate first (`if d_Q > r_Q: continue`),
*then* `passing = np.where(distances_to_q <= threshold_dist)[0]`, *then*
`if len(passing) >= threshold` -- already reduces correctly to "hit iff
contained" at `threshold=0`: `len(passing) >= 0` is always true once an
entry has passed the O1 containment gate, and the O1 gate already runs
unconditionally for every entry before this check, so an uncontained entry
never reaches it. No special-casing was added for this (an earlier draft of
this round's work proposed one; it was correctly identified as redundant --
the existing code was already right, once `N=0` was actually allowed to
reach it).

That "actually allowed to reach it" is the real fix, in
`population_runner.py:460`:

```python
sim_K = K  # was: sim_K = max(K, 1)
```

The old clamp existed because `population_runner.py`'s mean-recall
accumulation divides by `K` (`recall_sum += len(...) / K`), which would
raise `ZeroDivisionError` at `K=0` -- not because the CIG algorithm itself
needed `K>=1`. The fix is a guard, not a clamp:

```python
if K > 0 and result.certified_indices is not None and test_gt_indices is not None:
    recall_sum += len(set(result.certified_indices) & true_gt) / K
    n_hits_with_recall += 1
```

`mean_recall` simply stays at its existing default (`0.0`, meaning "not
computed") for K=0 runs, rather than attempting a division that's undefined
for a zero-size target anyway ("recall against 0 required results" isn't a
meaningful quantity to average).

## 3. Candidate proposal theory (`utils/base/population.py`)

### `CandidateEntry` and the completeness requirement on `R_c`

```python
@dataclass
class CandidateEntry:
    query: np.ndarray
    r_Q: float
    alpha: Optional[float]             # None for nested-sphere (ball) proposals
    r_c: float                         # used as cache radius at hit time
    R_c: object                        # bool array (len n_base) when store_rc=True, else None
    rc_size: int                       # R_c.sum(); always set regardless of store_rc
    covered_train_indices: object      # bool array (len n_train_queries)
    covered_test_indices: object       # bool array (len n_test_queries)
    top_k_vectors: List[np.ndarray]
    top_k_indices: List[int]
    half_gap: float
    source_index: Optional[int] = None  # index into observed_queries this candidate was proposed from
    n: Optional[int] = None             # neighbor rank (1..n_max) this candidate's sphere was built from
```

`to_cache_entry()`'s own comment states the key soundness constraint
directly: "CIG soundness requires the stored set to be the COMPLETE set of
base vectors within `r_c` of the query, i.e. `R_c` -- not the (possibly much
smaller) top_k set loaded at the eval K." A candidate's cached content must
be everything within its radius, not just its nominal top-K, or the CIG
distance-threshold check downstream would be checking against an incomplete
set and could wrongly reject (or, worse, wrongly accept) a hit.

`to_cache_entry()` also builds an **index-only** `CacheEntry` (`top_k_vectors
= []`, only `top_k_indices` set) -- the comment explains why: candidates'
`R_c` sets overlap heavily by construction (nested spheres, see below), so a
private per-entry vector copy would duplicate the same base vectors across
many entries. Distance computation instead goes through a shared
`base_hat`/`base_vectors` matrix passed at hit-check time. A second comment
notes `R_c`'s indices are stored as a compact `int32` array, not a Python
list of boxed ints, because at real scale (tens of thousands of entries with
~44% `R_c` fill over a 50k+ base) the boxed-int list cost ~9x more memory and
"was a direct contributor to OOM kills at high budgets."

### Generic alpha-scaling proposal

```python
r_c = alpha * r_Q
```

`propose_candidates(alpha_values=...)`: the simplest scheme, scaling a
query's own top-K radius by a constant factor. No triangle-inequality
construction is used -- coverage is *checked* post-hoc via a CIG mask rather
than *constructed* to guarantee it. Still supported (`test_e2e_runner_alpha_backward_compat`)
but not used by the production sweep (`run_population.py` only calls
`propose_candidates_ball`).

### The nested-sphere family: `propose_candidates_ball` (all K, including K=0)

`propose_candidates_ball` (a standalone `propose_candidates_point` existed
before Phase 3 Round 12; it was exactly `propose_candidates_ball(K=0)` and
was removed as redundant once that was fully confirmed) delegates to one
shared engine, `_propose_nested_spheres`, using **one uniform formula for
every K, including K=0**:

```
r_c(i, n) = D(query_i, neighbor_n) + margin_lookup[neighbor_n]
```

where `neighbor_n` is cache query `i`'s n-th nearest *other* cache-query
neighbor (`n` ranges `1..n_max`, so `n_max` nested spheres are proposed per
query, one per neighbor rank -- not just the single nearest). No epsilon
term -- see "Exactness and the removal of `NUMERICAL_EPS`" below.

**What `margin_lookup[neighbor_n]` physically is**: uniformly, for every K
including K=0, "distance from `neighbor_n` to its own K-th nearest **base
vector**" -- the radius of a ball centered on `neighbor_n` containing
`neighbor_n`'s own K nearest base vectors, computed via
`_precompute_kth_base_dist`. At K=0 this is trivially `0` (a 0-nearest-base-
vector ball has no radius) -- not a special substitution, just the natural
value of the same formula:

```python
if K == 0:
    margin_lookup = np.zeros(len(observed_queries), dtype=np.float32)
else:
    obs_hat = _normalize(observed_queries.copy())
    base_hat = _normalize(base_vectors.copy())
    margin_lookup = _precompute_kth_base_dist(obs_hat, base_hat, K, metric)
```

**Why the formula is sound for K>=1 (triangle-inequality containment
argument):** the sphere of radius `r_c` centered at cache query
`Q = query_i` is sized to *fully contain* `neighbor_n`'s own K-ball. For any
point `x` with `D(neighbor_n, x) <= margin_lookup[neighbor_n]`:

```
D(Q, x) <= D(Q, neighbor_n) + D(neighbor_n, x) <= D(Q, neighbor_n) + margin_lookup[neighbor_n] = r_c
```

So anything within `neighbor_n`'s own K-ball is guaranteed to also be within
`Q`'s radius `r_c`.

**What this means at K=0, worked through carefully** (this took several
wrong turns to get right this round -- see the note at the end of this
section): with `margin_lookup=0`, `r_c(i,n) = D(query_i, neighbor_n)`
exactly -- the sphere is simply "large enough to reach `neighbor_n`," no
more. The CIG certification check itself
(`D(query_i, t) + train_r_t[t] <= r_c` for any query `t`) is unconditionally
sound for *any* `r_c`, regardless of how it was constructed -- nothing is
ever wrongly certified. What changes at K=0 is that `train_r_t` (and
`test_r_t`) are *also* zero in containment mode (see "Containment mode:
`train_r_t`/`test_r_t` are zero, not the loaded K=1 ground truth" below), so
the coverage condition collapses to `D(query_i, t) <= r_c` -- pure distance
containment, with no per-query margin contribution from either side. This
matches simulate-time behavior exactly: once `sim_K`'s clamp to `>= 1` was
removed (Section 6), the CIG algorithm's existing O1 gate plus
`len(passing) >= threshold` logic *already* reduces to "hit iff contained"
at `threshold=0`, with no code change needed in the lemma itself (Section 2
covers this).

**Wrong turns this round, worth recording**: I initially proposed keeping
`margin_lookup = train_r_t` at K=0 (reusing the real, nonzero ground-truth
K-th-neighbor-query radius, matching what the old standalone
`propose_candidates_point` did) -- reasoning that simulate-time always
requires a real neighbor-based certification. That was wrong: it was only
true because of the `sim_K` clamp forcing `K=1` even in containment mode. I
also briefly proposed an explicit `if threshold <= 0: <early-return>` branch
in the CIG algorithm itself -- also unnecessary, since the existing
`len(passing) >= threshold` check already handles `threshold=0` correctly
once it's reachable at all.

### `precompute_ball_invariants`: caching K-independent work across a K-sweep

`_propose_nested_spheres` accepts an optional `invariants` parameter, filled
by calling `precompute_ball_invariants(observed_queries, train_queries,
test_queries, base_vectors, metric)` once and reusing it across every K in a
sweep (e.g. `debug_trajectory.py`'s K in `[0, 1, 5, 10]` per benchmark).
Every observed query's distance to every base vector / train query / test
query, and the cache queries' pairwise nearest-neighbor ordering, are
entirely K-independent (only `margin_lookup` and the K-specific
`train_r_t`/`test_r_t` depend on K), so recomputing them once per K value was
pure waste -- profiling found `_base_dists_for_query` alone was 4.1s of a
12.8s `propose_candidates_ball` call (SIFT c2048/K=5), recomputed on every
one of 4 swept K values for no reason.

The largest held array, `all_dists` (`n_cache x n_base`), is ~410MB in
float32 at the largest real scale in this codebase (c2048, `n_base=50000`)
-- held only for the duration of the caller's K-sweep, and only by callers
that opt in; every existing call site that doesn't pass `invariants` is
unaffected.

`precompute_ball_invariants` also introduces `_raw_pairwise_dists`, a shared
helper for the full `(n, n)` pairwise distance matrix among a set of
pre-normalized vectors (diagonal left as the real near-zero self-distance,
not masked). Both `_pairwise_cache_dists` (which sorts and masks the
diagonal to `inf`) and `precompute_ball_invariants` (which reuses the raw
matrix directly as `train_dist_only` when `train_queries is
observed_queries`, true for every real caller) now derive from this single
canonical matmul, rather than two independently-rounded ones that could
disagree by a few ULPs at a certification boundary -- the same class of risk
discussed in "Exact named-neighbor coverage" below.

### Containment mode: `train_r_t`/`test_r_t` are zero, not the loaded K=1 ground truth

`population_runner.py`'s `_load_benchmark_and_resolve_K`, `k_test_override
== 0` branch:

```python
cache_load_K = 1
obs_r_Q = np.zeros(n_cache, dtype=np.float32)
test_r_t = np.zeros(n_test, dtype=np.float32)
```

`obs_r_Q` is passed as *both* `observed_r_Q` and `train_r_t` at the
`propose_fn` call site (`population_runner.py:399-407`, literally the same
array both times) -- so zeroing it here zeroes the coverage-formula's
`train_r_t` too, symmetric with `test_r_t` (which was already zeroed before
this round). `cache_load_K` stays `1`: cache entries still need real top-1
ground truth (vectors/half-gaps) for entry construction -- that's unrelated
to `obs_r_Q`, which (confirmed by tracing every use of
`CandidateEntry.r_Q`/`observed_r_Q`) has no functional role beyond an
informational field in `_write_selection_log`'s debug output.
`to_cache_entry()` builds the real `CacheEntry` from `r_c`, never from
`observed_r_Q`.

### Exact named-neighbor coverage, and the removal of `NUMERICAL_EPS`

`_propose_nested_spheres` overrides the coverage check for the specific
`neighbor_n` a sphere was sized around, using `margin_lookup[nn_index]`
directly (the same value `r_c` was built from) rather than the separately
recomputed general formula:

```python
train_covered[nn_index] = nn_dist + float(margin_lookup[nn_index]) <= r_c
```

Since `r_c = nn_dist + margin_lookup[nn_index]` exactly, this holds by
construction -- an equality, not a floating-point coincidence -- for
*every* K uniformly, including K=0. This is what let `NUMERICAL_EPS` be
removed entirely (it existed in an earlier version of this formula,
`r_c = ... + NUMERICAL_EPS`): its only real justification was defending
against the *general* (non-`neighbor_n`) coverage formula recomputing
`D(query_i, neighbor_n)` via a different code path
(`_pairwise_cache_dists`'s full pairwise gemm vs. `_base_dists_for_query`'s
per-query gemv) than the one `r_c` was built from -- a real, measured
ULP-scale drift (~1e-7), but one that only mattered for the specific
neighbor a sphere was named after, which the `margin_lookup`-based override
above now makes exact regardless. Confirmed by a dedicated test
(`test_named_neighbor_coverage_exact_no_epsilon`) using bit-for-bit
identical internal values (population.py's own `_pairwise_cache_dists`/
`_precompute_kth_base_dist`, not an independently re-derived distance), at
both K=0 and K>=1.

Residual, much narrower floating-point risk still exists in two unrelated
places that don't have the same clean "reuse the exact value" fix: the
general coverage formula for queries *other* than the named neighbor (no
single value they can exactly reuse -- their `D(query_i, t)` is a genuinely
independent computation with no construction-time counterpart), and the
`R_c`/base-vector-membership boundary (`all_dists <= r_c`, checked against
the full base vector set, not a single named target). Both are effectively
measure-zero for continuous real data (an exact floating-point tie between
two independently-computed values essentially never happens by chance) --
not something `NUMERICAL_EPS` was ever comprehensively guarding against
either, so removing it doesn't introduce a new risk, just stops paying for
one that was never fully closed.

### `_precompute_kth_base_dist`'s `k >= 1` guard

```python
assert k >= 1, f"_precompute_kth_base_dist requires k >= 1, got k={k}"
```

This exists because `np.partition(d_batch, k-1, axis=1)[:, k-1]` at `k=0`
becomes `partition(..., -1, ...)`, which extracts the **maximum** distance,
not zero or any meaningful quantity -- a silent-wrong-answer failure mode if
this function were ever called with K=0 instead of routing through the
`margin_lookup = np.zeros(...)` branch above. Guarded loudly rather than
silently. Tested directly by `test_precompute_kth_base_dist_rejects_k0`.

### No dominance pruning at proposal time

All `n_max` nested spheres per query are proposed and kept (unless a
sphere's `covered_train_indices` is empty), even though a smaller sphere for
the same query is strictly dominated once a larger sphere for that query
exists in the pool.

> **My understanding (please confirm):** pruning is left implicit to the
> selection stage's cost-efficiency scoring rather than done at proposal
> time -- a strictly-dominated smaller sphere can never win `select_candidates`'
> `newly-covered / marginal-cost` ratio over the larger sphere it's a strict
> subset of, so it simply never gets selected, without needing an explicit
> dominance check in the proposal code. This keeps proposal a pure
> "generate everything geometrically valid" stage and pushes all trade-off
> decisions into one place (selection).

### `n_max` constraint

`_propose_nested_spheres` raises `ValueError` if `n_max >= n_cache` -- it
needs at least `n_max` other cache queries to rank against per query.

## 4. Selection theory (`utils/base/selection.py`)

`select_candidates(candidates, budget_B, strategy, n_train_queries,
n_test_queries, seed=42) -> (selected, trajectory)`.

**Budget accounting**: budget `B` is over *base-vector cache footprint*
(`rc_size` / `R_c`), not candidate count. `_fits_budget` short-circuits via
`rc_size` (cheapest case: zero overlap with what's already cached) and only
computes the true marginal `R_c & ~cached_mask` when the cheap check can't
decide.

**Coverage**: a train (or test) query index `t` is "covered" once any
selected candidate's `covered_train_indices[t]` (computed at proposal time
via the CIG condition) is True. Selection maintains a running OR of all
selected candidates' coverage masks.

**Trajectory**: one dict appended per accepted candidate --
`{num_entries, vectors_cached, cig_train_coverage, cig_test_coverage,
new_train_coverage, new_test_coverage}` -- letting downstream analysis plot
coverage/hit-rate against memory spent incrementally, not just at the final
budget.

### The four strategies

1. **`random`**: fixed-seed permutation, single pass, accept iff fits
   budget. No re-scoring.
2. **`greedy_naive`**: repeatedly picks the candidate with the most newly-
   covered train queries, ignoring cost/size entirely.
3. **`greedy_cost_static`**: scores `newly / static_cost`, where
   `static_cost = rc_size` -- the candidate's fixed, precomputed total ball
   size, **not** its marginal cost against the current cache. A candidate
   whose `R_c` heavily overlaps already-cached vectors is scored as if it
   still costs its full ball size.
4. **`greedy_cost`** (the strategy actually used in production,
   `STRATEGIES = ['greedy_cost']` in `run_population.py`): scores
   `newly / marginal`, where `marginal` is the *true* incremental cost
   (`R_c & ~cached_mask`) recomputed as selection proceeds -- classic
   weighted-set-cover-style benefit-per-unit-marginal-cost greedy.

`greedy_cost_static` and `greedy_cost` provably select identically when
candidates have zero `R_c` overlap (tested directly:
`test_greedy_cost_static_matches_dynamic_when_no_overlap`), since in that
case static cost and marginal cost coincide.

**Zero-marginal-cost candidates**: `greedy_cost` defers candidates with zero
marginal cost and positive benefit (their `newly/marginal` score would be
undefined/infinite) into a separate list, flushed at the end sorted by
benefit against the final coverage mask, added for free since their `R_c` is
already a subset of what's cached.

### `greedy_cost`'s incremental delta-update design

This is the most involved piece of `selection.py`, and its history matters
for understanding why it looks the way it does. There have been four
generations of this code:

1. **Original**: `R_c`, `covered_train_indices`, `covered_test_indices` were
   Python `set`s of ints; the greedy loops did `set` union/difference every
   iteration -- expensive due to per-element Python object overhead (~28
   bytes/int).
2. **`selection_fast.py`** (now deleted from the working tree, recovered from
   git history for this doc, commit `7685bd5`): replaced sets with a flat
   numpy bool array for "what's cached" and numpy bool matrices for
   train/test coverage, plus a CSR-like flat `R_c` structure with
   `np.add.reduceat` for batched marginal-cost computation. Its own
   benchmark script (`bench_selection.py`, also deleted) validated exact
   trajectory-dict equality (to `1e-9`) against the original at every step,
   for every strategy, and measured speedups of 6.8x-18.3x on synthetic data
   (documented in the now-deleted `docs/selection_optimization.md`).
   `selection_fast.py` still recomputed `marginal`/`newly` fully from
   scratch every iteration via its batched helper -- faster per-recompute,
   but still `O(|active| * n_base)` per iteration.
3. **Current `selection.py`**: fully absorbed the fast-path techniques
   directly (`cached_mask`/`covered_train_mask`/`covered_test_mask` as numpy
   bool arrays, `CandidateEntry.R_c` itself now a bool array with a cached
   `rc_size`), **and** added a further optimization beyond even
   `selection_fast.py`: an incremental delta-update scheme. After picking
   `best`, rather than recomputing `marginal`/`newly` for all remaining
   `active` candidates from scratch, it computes exactly the bit positions
   `best` newly contributed:

```python
new_base_idx = np.flatnonzero(best.R_c & ~cached_mask)
new_train_idx = np.flatnonzero(best.covered_train_indices & ~covered_train_mask)
```

   (computed *before* mutating the masks), then subtracts, for each
   remaining active candidate, how many of those specific newly-set bits it
   also has:

```python
marginal = marginal - _chunked_delta_counts(active, 'R_c', new_base_idx)
newly = newly - _chunked_delta_counts(active, 'covered_train_indices', new_train_idx)
```

   This moves the dominant per-iteration cost from `O(|active| * n_base)` to
   `O(|active| * k)` where `k` = number of newly-set bits that iteration, and
   bounds total work over the whole run (summed `k` <= `n_base` for
   marginal, <= `n_train_queries` for newly). `_chunked_delta_counts`
   processes `active` in fixed batches (`_DELTA_CHUNK_SIZE = 6000`) to bound
   peak transient memory.
4. **Bit-packed delta counting** (current): `_chunked_delta_counts` gained a
   second, bit-packed code path, chosen per chunk based on how large `idx`
   (the number of newly-set bits to count) is relative to the attribute
   array's length (`_BITPACK_IDX_FRACTION = 1/20`). The original fancy-index
   gather (`batch[:, idx]` then `count_nonzero`) scales with `len(idx)`;
   packing (`np.packbits` + a 256-entry byte popcount lookup table,
   `_POPCOUNT_TABLE`) costs roughly a fixed amount per chunk regardless of
   `idx` size, dominated by `np.packbits(chunk, axis=1)`. The two cross over
   empirically around `idx ~= attr_len/20` (measured against real SIFT
   c2048/K=5 data): below that, gather wins, often heavily, since most calls
   in a real run have `idx` in the single digits to low hundreds; above it,
   packing wins by up to ~12x at the largest real `idx` sizes observed
   (~33000, i.e. late-budget iterations where the active set barely shrinks
   and each pick contributes many newly-covered bits). Both paths compute
   the exact same integer popcount -- verified via `np.array_equal` in
   `tests/utils/test_selection.py` -- so this is purely a representation/
   performance choice, not a behavior change.

**Correctness of the delta-update -- required reading, not optional
optimization detail**: this is designed and tested to be *exactly*
equivalent to full recomputation, not an approximation. The source comment
itself states it's "numerically exact... integer popcounts of set
differences, not an approximation." Three tests in
`tests/utils/test_selection.py` specifically guard this:

- `test_vectorized_matches_naive_reference`
- `test_greedy_cost_incremental_matches_reference_at_high_budget`
- `test_greedy_cost_chunked_delta_matches_reference_across_chunk_boundary`
  (specifically targets the chunking boundary, where a naive implementation
  could plausibly double-count or miss a delta at the edge between
  `_DELTA_CHUNK_SIZE` batches)

A large comment block in `selection.py` (lines ~126-166) also documents an
**abandoned** approach: an earlier attempt maintained a persistent stacked
matrix shrunk via repeated boolean-mask indexing, creating several large
transient copies per iteration and "ballooning peak memory to ~6GB with no
speed benefit over the original Python loop at real scale." The design
principle that survived: `greedy_naive`/`greedy_cost_static` build a fresh
stacked matrix each iteration and discard it (their active-set filters
shrink quickly, so this stays cheap), while `greedy_cost` "no longer follows
this pattern" once budget fraction approaches 1.0 (the active set barely
shrinks, so hundreds of full recomputes would dominate) -- motivating the
incremental-delta approach as a genuinely different strategy, not just a
faster version of the same one.

## 5. `KVCache` internals (`utils/base/kv_cache.py`)

### `CacheEntry` fields

- `query` -- the cached query vector Q.
- `top_k_vectors` -- the K closest base vectors to `query` (can be `[]` for
  index-only entries, see below).
- `r_Q` -- distance to the K-th (farthest cached) vector; the CIG radius.
- `half_gap` -- half the gap between the K-th and (K+1)-th vector; the HGG
  threshold.
- `top_k_indices` -- stable base-vector indices for `top_k_vectors`.
- `top_k_matrix` -- pre-stacked, pre-normalized matrix enabling a single-
  matmul CIG distance computation; built by `CandidateEntry.to_cache_entry()`
  in the population pipeline, `None` when created via `add_entry` directly.

### Two ways entries get built

- `populate_from_queries(queries, main_memory, K)`: high-level, brute-force
  path -- for each query runs `main_memory.top_k_search`, then
  `add_entry(query, top_k_vecs, r_Q=top_k_dists[-1], half_gap=gap/2.0,
  top_k_indices)`. Used by hit-policy's on-the-fly mode and synthetic.
- `add_entry(...)` directly: the low-level primitive population uses via
  `CandidateEntry.to_cache_entry()` -- and notably, `simulate.py`'s own
  `populate_cache()` also calls `add_entry` directly with an **empty**
  `top_k_vectors` list, creating index-only entries (see Section 3's
  discussion of why -- avoiding duplicated per-entry vector copies across
  heavily-overlapping `R_c` sets).

### The lazy-build pattern: `get_d_Q_batch` / `get_entry_radii`

```python
def get_entry_radii(self) -> np.ndarray:
    """(n_entries,) array of each cache entry's r_Q, for vectorizing the
    O1 gate (d_Q <= r_Q) across all entries at once instead of a
    per-entry Python comparison. Same lazy-build as get_d_Q_batch."""
    if self._entry_radii is None:
        self.build_query_matrix()
    return self._entry_radii
```

Both `get_d_Q_batch(query)` and `get_entry_radii()` check if their backing
array is `None` and, if so, call the *same* `build_query_matrix()`, which
populates `_query_matrix` and `_entry_radii` together in one pass -- one
lazy build serves both accessors, avoiding redundant recomputation whenever
the cache is mutated (both are invalidated together on `add_entry`). The
purpose is exactly what the docstring says: letting `CacheSimulator.run_query`
vectorize the CIG "O1 gate" (`d_Q <= r_Q`) across *all* cache entries in one
NumPy comparison, rather than a Python loop doing one comparison per entry --
this is the concrete call site (`simulate.py`'s dispatch for
`"lemma1_no_union"`) that justifies `get_entry_radii()`'s existence.

## 6. Pipeline orchestration (`population_runner.py`)

### Stage-by-stage, with what runs how often

Inside `run_population_benchmark()`:

1. Load benchmark + resolve K (once).
2. Load cache ground-truth arrays (once).
3. **Propose candidates** -- runs **once total** per benchmark, before the
   budget loop. Candidates don't depend on budget.
4. For each `budget_B` in `budget_values`, for each `strategy`:
   - **Select** under `budget_B` (`select_candidates`).
   - **Build** a fresh `KVCache` from the selection.
   - For each `sigma` in `sigma_values`: **simulate** oracle, then simulate
     non_oracle.

### oracle vs. non_oracle

The only difference is which query set is passed as the evaluation set to
`_run_simulation`:

- **oracle**: `test_queries=obs_queries` -- the *same* queries the cache was
  populated from, with `test_gt_indices=obs_ti` (the cache's own top-k
  ground truth).
- **non_oracle**: `test_queries=test_q` -- the held-out test query set,
  evaluated against separately-loaded ground truth.

> **My understanding (please confirm):** oracle measures an upper bound /
> sanity check ("if you ask the cache about the exact queries it was
> populated with, how often does it certify a hit"), while non_oracle
> measures generalization to unseen queries, which is the actual metric of
> interest for real cache effectiveness. Consistent with this,
> `analyze_population_proposal.py` only ever reads `non_oracle` summaries
> for plotting -- oracle results are produced and written to disk but not
> consumed by analysis, which reads as "oracle exists for debugging/sanity,
> not as a reported result."

### Sigma sweep cost model

Confirmed directly, `population_runner.py:462-485`:

```python
for sigma in sigma_values:
    sigma_dir = b_dir if single_default_sigma else b_dir / f'sigma_{sigma:.2f}'

    oracle_results = _timed_stage(..., _run_simulation,
        kv_cache, base, obs_queries, sim_K, algorithms,
        effective_name, strategy, budget_B, 'oracle', metric, sigma=sigma,
        test_gt_indices=obs_ti)
    ...
    non_oracle_results = _timed_stage(..., _run_simulation,
        kv_cache, base, test_q, sim_K, algorithms,
        effective_name, strategy, budget_B, 'non_oracle', metric, sigma=sigma,
        test_gt_indices=test_gt_indices)
    ...
```

This loop is nested *inside* the `(budget, strategy)` loop, so `select_candidates`
and `_build_kv_cache` run once per (budget, strategy) pair, before this sigma
loop, and the same `kv_cache` object is reused unchanged across every sigma
value -- **only the simulate stage repeats per sigma**, not propose/select/
build. This was measured directly this session: a sigma sweep of 4 values
(`0.7, 0.8, 0.9, 1.0`) took ~2.9x the wall time of a single sigma=1.0 run for
the same config, consistent with "only 2 of 6 pipeline stages (oracle +
non_oracle simulate) repeat 4x, everything else runs once."

### Output nesting

```python
sigma_values = sigma_values if sigma_values else [1.0]
single_default_sigma = sigma_values == [1.0]
...
sigma_dir = b_dir if single_default_sigma else b_dir / f'sigma_{sigma:.2f}'
```

Flat layout (`.../B{budget}/oracle/{strategy}/...`) when only the single
default `sigma=1.0` is used; nests one level deeper
(`.../B{budget}/sigma_{value:.2f}/oracle/{strategy}/...`) for any other sweep,
including a single non-default value like `[0.9]`. Trajectory/selection-log
files (sigma-independent, since they're produced before the sigma loop) stay
directly under `b_dir` regardless.

### Budget semantics

`run_population.py::_budget_values` computes budgets as fractions of `n_base`
(the base-vector corpus size for that benchmark, from `base_vectors.npy`'s
row count), not the cache/train query count:

```python
def _budget_values(benchmark_dir: Path) -> list:
    n_base = np.load(base, mmap_mode='r').shape[0]
    return [max(1, int(f * n_base)) for f in BUDGET_FRACTIONS]
```

Each resulting `budget_B` is an absolute base-vector-count cap consumed by
`select_candidates` as described in Section 4.

## 7. CLI driver (`run_population.py`)

### Current CLI surface

`--dataset` (required), `--n-max` (default 20), `--k-test` (default
`[0, 1, 5, 10]`), `--benchmark` (default: all research benchmarks),
`--strategies` (default `['greedy_cost']`), `--sigma` (default `[1.0]`),
`--workers` (default 1), `--seed` (default 42). **`--method` does not
exist** -- point-like proposal was removed as a standalone function; this
script "only ever produces `ball/` output now"; there is no `_run_point`
function left in the file.

### `K = k_test`

```python
# K_cache = K_test = K, always -- including K=0 (containment mode) --
# no separate k_cache/k_test distinction, no clamp needed.
K = k_test
```

An earlier version of this line was `K = k_test if k_test > 0 else 1` -- a
clamp that forced `K=1` in containment mode. Fixed in an earlier round;
directly regression-tested by `test_e2e_ball_runner_k_test_zero_containment`
in `tests/population/test_e2e_population_proposal.py`. This `K` (the ball
margin's own parameter) is distinct from `population_runner.py`'s
*separate* `sim_K` (the simulate-time certification target, Section 2's
"N=0" subsection) -- both are now simply `k_test`/`K` unconditionally,
neither clamped, but they're different variables serving different roles
(one sizes candidates at proposal time, the other is passed as `N` to the
CIG algorithm at simulate time).

(A separate variable, `override_K = k_test if k_test > 0 else None`, feeds a
*different* code path in `population_runner.py`'s ground-truth-loading logic
-- not to be confused with the `K` used for `propose_candidates_ball`.)

### Sweep dimensions

```python
BUDGET_FRACTIONS = [
    0.05, 0.10, 0.15, 0.30, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98, 1.0,
]
DEFAULT_K_TEST = [0, 1, 5, 10]
DEFAULT_N_MAX = 20
STRATEGIES = ['greedy_cost']
```

Cache-size sweep is encoded in the benchmark names themselves (`c256`,
`c512`, `c1024`, `c2048` cache/train queries per `DATASET_CONFIG`'s
`research_benchmarks` lists, 4 sizes x 2 datasets = 8 benchmarks total).

`BUDGET_FRACTIONS` was densified from an original 7 values to the 15 above,
adding finer resolution in the `0.45-0.98` range specifically -- where the
hit-rate curves for both datasets are furthest from either extreme and most
sensitive to budget, per the disappointing `memory_vs_hitrate.png`/
`budget_pct_vs_metrics.png` results this investigated.

### Work dispatch

```python
work = [(bm, args.n_max, k_test) for bm in benchmarks for k_test in k_test_vals]
```

One work item per (benchmark, k_test) pair (n_max isn't swept within one CLI
invocation). `--workers 1` runs serially in-process; `--workers > 1` uses
`concurrent.futures.ProcessPoolExecutor`, submits all items eagerly, and
drains via `as_completed` (`.result()` re-raises any worker exception in the
main process -- no partial-failure suppression). All workers share one
`run_ts` so a parallel run lands under a single timestamped output tree.

### Output naming

```python
sub_dir = output_dir / run_ts / 'ball' / f'nmax{n_max}_k{k_test}'
```

i.e. `{output_dir}/{run_ts}/ball/nmax{n_max}_k{k_test}/{benchmark_name}/B{budget}/...`.
Directly regression-tested by `test_analyze_regex_matches_real_naming`,
which also asserts two previously-used incorrect patterns
(`n(\d+)_k(\d+)` and `n(\d+)_kc(\d+)_k(\d+)`) do *not* match -- see Section 8.

## 8. Analysis (`analyze_population_proposal.py`)

### What it produces

`n_max` is fixed at 20 in every real run (`--n-max` takes a single value,
never actually swept), so Phase 3 Round 12 replaced the previous `n x
K_test` grid with **`sigma x K_test`** (rows=sigma, columns=K_test), and
replaced the old per-sigma-directory separate plot files
(`sigma_{v:.2f}/ball/...`) with **one combined file per metric per
dataset** -- no more `sigma_*/` output nesting. `analyze_dataset()` walks
`run_dir / 'ball'`, parses each `nmax{n}_k{k_test}` directory, loads every
swept sigma value for each, and produces:

- `ball/memory_vs_hitrate.png` -- hit rate vs. memory utilization
  (`vectors_cached / n_base`), one grid cell per (sigma, K_test), **both
  train and test hit rate overlaid per cell** (via `_plot_grid_train_test`,
  shared train/test legend), one line per cache size, with a `y=x`
  reference line. `n_max` noted in the title.
- `ball/memory_vs_candidates_cached.png` -- same grid, y=candidate count
  (`num_entries`, not vector count). Since candidate counts aren't naturally
  bounded like a rate, `_plot_grid` computes a shared y-axis from the global
  max across every cell (rather than each subplot auto-scaling to its own
  local max) so cells stay visually comparable.
- `ball/recall_vs_sigma.png` -- one combined grid: rows=K_test,
  columns=budget fraction, x=sigma, y=`mean_recall` (from
  `SimulationResult.mean_recall`, already serialized into `summary.json`),
  one line per cache size per cell. Only produced when a real sigma sweep
  exists (`len(sigma_vals) >= 2`) -- nothing meaningful to plot sigma
  against with a single value.

`memory_vs_train_coverage.png` (a separate CIG-train-coverage grid) and
`aggregate/max_budget_hit_rate_vs_num_train_queries.png` (a max-budget-only
bar chart) were both produced by earlier versions of this script and have
since been removed -- train coverage is no longer plotted standalone (train
hit rate, plotted alongside test above, was the more useful signal), and the
aggregate bar chart was dropped along with the rest of the `aggregate/`
subdirectory as part of simplifying this script's output.
`compile_proposal.py`'s `_method_images` still opportunistically globs an
`aggregate/` subdirectory if one exists (harmless "skip if missing" -- see
its convention for every optional debug image), but nothing currently
populates it.

If multiple `n` values were ever found (never expected in practice, since
`n_max` isn't swept), `analyze_dataset()` prints a warning and uses the
first one found for the title -- a defensive fallback, not a supported
sweep dimension.

### Directory-parsing regex

```python
m = re.fullmatch(r'nmax(\d+)_k(\d+)', param_dir.name)
```

Matches real `run_population.py` output (`nmax{n}_k{k_test}`, no `kc`
component since `K_cache` has always equaled `K_test` in the real
pipeline). An earlier version of this regex never matched real pipeline
output at all (expected a different naming shape) -- fixed in an earlier
round, directly regression-tested by `test_analyze_regex_matches_real_naming`.

### Sigma detection and the single-call restructure

`_sigma_values_in()` peeks into one representative `B{budget}/` directory
for `sigma_\d+\.\d+`-named subdirectories, returning `[]` for the flat
(single default sigma) layout. `main()` calls `analyze_dataset()` **once
per dataset** regardless of whether a sigma sweep exists (previously it
looped and called `analyze_dataset()` once per sigma value, producing
separate output trees) -- `analyze_dataset()` itself now loops over
`(param_dir, sigma)` pairs internally, building one combined `all_data`
dict keyed by `(sigma, k_test)` and passing it through a single set of grid
plots.

### Debug JSON

Written to `{debug_dir}/{label}_ball_debug.json`, structured as
`dataset -> params -> benchmarks -> budgets -> {hit_rate, mean_recall,
memory_util, budget_utilization, vectors_cached, num_entries,
train_coverage_final, ...}`, plus `per_benchmark_sanity` flags (e.g.
"hit_rate near zero across all budgets", "no candidates cached at max
budget") and roll-up `param_combo_summary`/`dataset_summary` blocks
identifying best/worst-performing combos. Param keys are now
`sigma{sigma:.2f}_k{kt}` (were `n{n}_k{kt}`), and the top-level JSON
includes an `n_max` field for provenance now that it's not part of the key
structure.

> **My understanding (please confirm):** the purpose is automated auditing of
> a full sweep without eyeballing every plot -- a machine-readable place to
> catch degenerate runs (zero coverage, empty cache, low budget utilization)
> systematically rather than visually.

### Downstream PDF compilation: `compile_proposal_sigma.py` removed

`compile_proposal_sigma.py` existed specifically to walk the old
`sigma_{v:.2f}/` output directories and render one PDF section per sigma
value. Once analysis stopped producing that nesting (this round), it would
have found zero sigma directories and silently produced an empty section --
removed entirely rather than left as a dead, misleading path.
`compile_proposal.py` (the plain compiler, already glob-based over whatever
PNGs exist in `processed_dir / 'ball'`) needed only one addition --
`recall_vs_sigma*.png` added to its glob patterns -- to pick up everything
the reworked analysis script now produces in a single pass. The Makefile's
`population-proposal-analyze` target was updated to call
`compile_proposal.py` instead of the now-deleted `compile_proposal_sigma.py`.
Its output now lands under `simulations/population_proposal_sigma/`
(the directory name the deleted script used to write to) rather than
directly under `simulations/`, and the per-dataset "Ball-like Proposal"
header page was dropped from the PDF -- ball is the only method, so it was
a redundant page between the dataset divider and the images
(`simulations/pdf/report.py::render_sections` now skips the header page
entirely when a `header()` node's title is falsy).

## 9. Test coverage map

| File | What it validates |
|---|---|
| `tests/utils/test_proposals.py` | Sphere-sweep coverage invariants (all `n` in `[1,n_max]` proposed, `R_c` monotone in `n`); ball geometric correctness at both K=0 and K>=1 (n-th neighbor coverage boundary, `R_c` membership, CIG satisfaction, self-exclusion); `test_named_neighbor_coverage_exact_no_epsilon` -- proves the named-neighbor coverage check is exact by construction (no `NUMERICAL_EPS`) at both K=0 and K>=1, via bit-for-bit-identical internal values; `test_propose_candidates_ball_r_c_monotone_in_K` -- r_c monotone non-decreasing in K; `test_precompute_kth_base_dist_rejects_k0` -- the K>=1 guard; numerical helpers cross-checked against brute force. |
| `tests/utils/test_lemmas.py` | `test_lemma1_no_union_sigma_n_zero_hits_when_contained`/`_misses_when_not_contained` -- pin down that `N=0` correctly reduces to "hit iff contained" via the *existing*, unmodified CIG algorithm structure, now that `population_runner.py`'s `sim_K` clamp no longer prevents `N=0` from being reached. |
| `tests/utils/test_selection.py` | All 4 strategies' behavior and invariants; `test_greedy_cost_static_matches_dynamic_when_no_overlap`; vectorized/incremental/chunked `greedy_cost` correctness against a naive reference, including the chunk-boundary edge case. |
| `tests/utils/test_selection_with_new_candidates.py` | Selection against realistic ball(K=0)/ball(K>=1)-shaped `CandidateEntry` data from a clustered synthetic benchmark. |
| `tests/population/test_population_runner.py` | `_run_simulation` sigma/hit-rate correctness, recall bounds (`recall >= sigma` for every hit) for K>=1, index-space consistency between certified and ground-truth indices, `mean_recall` excluding misses; `test_run_simulation_k_zero_no_zero_division` -- confirms the recall-division guard at K=0 (no `ZeroDivisionError`, `mean_recall` stays at its default); `_write_selection_log`'s vectorized computation against a naive per-pair reference. |
| `tests/population/test_e2e_population_proposal.py` | Full pipeline e2e via `run_population_benchmark` and via `_run_ball` directly: K=0/K=5 smoke tests, containment-only path, `n_max` sweep, no-K_cache-constraint check, **`test_e2e_ball_runner_k_test_zero_containment`** (the direct regression test for the fixed clamp bug, strengthened this round to also assert `mean_recall == 0.0`), budget-utilization range, sigma output-structure nesting, alpha-proposal backward compatibility. |
| `tests/population/test_e2e_population.py` | Lower-level e2e: manual propose -> select (all 4 strategies) -> build -> simulate chain with selection invariants; output-file schema (`summary.json`, `per_query.csv`, trajectory JSON). |
| `tests/population/test_analyze_proposal.py` | `test_analyze_regex_matches_real_naming` -- direct regex regression test; expected-plots-created check (including the combined `recall_vs_sigma.png` grid) with explicit negative checks for deprecated artifact filenames, old per-K_test recall filenames, and old `sigma_*/` nesting; `test_analyze_no_recall_plot_without_sigma_sweep` -- confirms no recall plot without a real sigma sweep; memory-utilization range; debug JSON schema (including the new `mean_recall` field and `sigma{v:.2f}_k{kt}` param keys). |
| `tests/population/microbenchmarks/*` | `microbenchmark_containment` only (`microbenchmark_candidates`/`microbenchmark_k` and their dedicated test files were deleted -- they exercised the generic alpha-scaling `propose_candidates` path, unrelated to and unused by the production ball pipeline). Fixed a real strict-`<` vs. non-strict-`<=` inconsistency; still intentionally has its own separate `K_cache`/`K_test` distinction, `K_CACHE_VALUES`/`K_TEST_VALUES` valid only where `k_test < k_cache` -- a narrower diagnostic tool, not a leftover inconsistency with the main pipeline's now-unified `K`. |
| `tests/population/test_debug_trajectory.py` | `debug_trajectory.py`'s standalone diagnostics: `test_monotonic_hit_rate_pass_matches_reference` (the most important test -- `monotonic_hit_rate_pass` against a naive reference implementation), its `dist_matrix`-supplied variant, its sigma!=1.0 rejection, `classify_phases`'s deferred-flush/strictly-increasing labeling, and `compute_budget_pct_view` (including the fixed `searchsorted` boundary semantics -- immediate-saturation and beyond-plateau cases both use the last step, not an under-converged early one). |

## 10. Known gaps

Carried forward from `docs/KNOWN_ISSUES.md`, re-confirmed by this round's
research:

- **Proposal function duplication (partially addressed)**: `propose_candidates`
  (generic alpha) remains a fully separate code path from the nested-sphere
  family (`propose_candidates_ball`, covering every K including 0). They
  share only low-level helpers like `_cig_mask`/`_normalize`, not the
  coverage/construction logic.
- **`QueryResult.error` is effectively dead**: `run_query` sets
  `error=metadata.get('error')`, but no lemma function in `lemmas.py` or
  `lemmas_sigma.py` ever sets `metadata["error"]`, and there is no
  `try/except` anywhere in `run_query` itself. In practice, real failures
  (an unknown algorithm string, a downstream exception from a lemma call)
  propagate directly out of `run_query` and would abort the simulation
  loop rather than being captured per-query -- confirmed still true by this
  round's research. Worth deciding explicitly whether that's the desired
  behavior (defensive per-query capture vs. fail-fast) rather than leaving
  it as an unused dataclass field that looks like it does something it
  doesn't.
