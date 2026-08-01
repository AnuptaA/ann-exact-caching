# Known Issues / Notable Gotchas

Resolved-but-instructive findings worth remembering, rather than a live punch
list -- small deferred items were pruned from here since they weren't worth
the ongoing attention; check git history if you need the pruned items back.

## Floating-point drift between two distance-computation paths for the same value

`utils/base/population.py`'s candidate-proposal code computes the distance
from a cache query to another cache query (its "named neighbor," the one a
sphere's radius `r_c` was sized around) via two genuinely different code
paths depending on where in the pipeline the value is needed:

- `_pairwise_cache_dists` -- a full `(n_cache, n_cache)` pairwise gemm,
  sorted per row.
- `_base_dists_for_query` -- a per-query gemv against a single query vector.

These two paths can disagree at the ULP scale (~1e-7), confirmed by direct
measurement. That's normally negligible, except at exactly the boundary the
code cares about: `r_c` is *defined* as `D(query_i, neighbor_n) +
margin_lookup[neighbor_n]`, so `neighbor_n` should trivially satisfy its own
coverage check (`D(query_i, neighbor_n) <= r_c`) by construction. If that
distance is *recomputed* via the other code path instead of reusing the
value `r_c` was built from, the ULP drift could occasionally push the
recomputed distance a hair above `r_c`, making the named neighbor fail its
own coverage check -- a real, if rare, correctness risk, not a hypothetical
one.

**Resolution**: `_propose_nested_spheres` now reuses the exact
construction-time value (`margin_lookup[nn_index]`, the same value `r_c` was
built from) for the named neighbor's coverage check instead of recomputing
it via the general formula:

```python
train_covered[nn_index] = nn_dist + float(margin_lookup[nn_index]) <= r_c
```

Since `r_c = nn_dist + margin_lookup[nn_index]` exactly, this holds by
construction -- an equality, not a floating-point coincidence -- for every
K uniformly, including K=0. This also let a previously-present
`NUMERICAL_EPS` fudge factor be removed entirely: its only real
justification was defending against exactly this cross-path drift, and the
"reuse the exact value" fix makes it unnecessary rather than just
papering over it. Confirmed by a dedicated test,
`test_named_neighbor_coverage_exact_no_epsilon`, using bit-for-bit-identical
internal values (not an independently re-derived distance) at both K=0 and
K>=1.

**Residual, narrower risk (not fully closed, and not expected to matter)**:
two other places still compute distances independently with no single
construction-time value to reuse -- the general coverage formula for queries
*other* than the named neighbor, and the `R_c`/base-vector-membership
boundary (`all_dists <= r_c`, checked against the full base set). Both are
effectively measure-zero for continuous real data (an exact floating-point
tie between two independently-computed values essentially never happens by
chance), so this isn't a new risk introduced by removing `NUMERICAL_EPS` --
just an acknowledgment that it was never a comprehensive guard against this
class of bug in the first place. See `docs/POPULATION_DESIGN.md`'s "Exact
named-neighbor coverage, and the removal of `NUMERICAL_EPS`" section for the
full writeup, including the `_precompute_kth_base_dist` `k >= 1` guard this
same investigation motivated.

**Why this is worth remembering**: if a similar pattern shows up elsewhere
(a value computed once at construction time, then independently
recomputed later via a different code path for a boundary check), the
same class of ULP-scale disagreement is possible. The fix pattern -- reuse
the exact construction-time value instead of recomputing -- generalizes.
