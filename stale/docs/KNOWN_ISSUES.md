# Known Issues / Notable Gotchas

Resolved-but-instructive findings worth remembering, not a live punch list.
Small deferred items were pruned; check git history if a pruned item needs
revisiting.

## Floating-point drift between two distance-computation paths for the same value

`utils/base/population.py`'s candidate-proposal code computes the distance
from a cache query to another cache query (its "named neighbor," the one a
sphere's radius `r_c` was sized around) via two different code paths
depending on where in the pipeline the value is needed: `_pairwise_cache_dists`
(a full pairwise matrix, sorted per row) and `_base_dists_for_query` (a
per-query gemv). These two paths can disagree at the ULP scale (about 1e-7).
Normally negligible, except at the exact boundary the code cares about:
`r_c` is defined as `D(query_i, neighbor_n) + margin_lookup[neighbor_n]`, so
`neighbor_n` should trivially satisfy its own coverage check
(`D(query_i, neighbor_n) <= r_c`) by construction. If that distance is
recomputed via the other path instead of reusing the value `r_c` was built
from, ULP drift could occasionally push it a hair above `r_c`, making the
named neighbor fail its own coverage check.

**Resolution**: `_propose_nested_spheres` reuses the exact construction-time
value (`margin_lookup[nn_index]`) for the named neighbor's coverage check
instead of recomputing it:

```python
train_covered[nn_index] = nn_dist + float(margin_lookup[nn_index]) <= r_c
```

Since `r_c = nn_dist + margin_lookup[nn_index]` exactly, this holds by
construction, an equality, not a floating-point coincidence, for every K
uniformly, including K=0. This also let a previously-present
`NUMERICAL_EPS` fudge factor be removed entirely, since its only real
justification was defending against exactly this cross-path drift.
Confirmed by `test_named_neighbor_coverage_exact_no_epsilon`, using
bit-for-bit-identical internal values at both K=0 and K>=1.

**Residual, narrower risk**: two other places still compute distances
independently with no single construction-time value to reuse, the general
coverage formula for queries other than the named neighbor, and the
`R_c`/base-vector-membership boundary against the full base set. Both are
effectively measure-zero for continuous real data, so this isn't a new risk
introduced by removing `NUMERICAL_EPS`, just an acknowledgment it was never
a comprehensive guard against this class of bug. See
`docs/POPULATION_DESIGN.md` section 3 for the full writeup.

**Why this is worth remembering**: if a similar pattern shows up elsewhere
(a value computed once at construction time, later independently recomputed
via a different code path for a boundary check), the same class of
ULP-scale disagreement is possible. The fix generalizes: reuse the exact
construction-time value instead of recomputing.
