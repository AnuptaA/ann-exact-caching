# Post-mortem: full-corpus-dependent recall estimators (deprecated)

Tag: `deprecated-full-corpus-recall-estimators`

## What was built

- **Median-halving tree** (`halving/`): a recursive bucket tree for exact density
  counting. Abandoned for query-time margin collapse at high dimension.
- **Multi-table Horvitz-Thompson LSH estimator** (`common/lsh_horvitz_thompson.py`):
  validated unbiased, credited to Wu, Charikar & Natchu (2018), with the
  enumeration-vs-sampling adaptation justified via the HT-weighted quantile-estimation
  literature.
- **`recall_hat` radius-inversion fix**: `per_cache_query_recall_hat`'s radius was a
  biased order statistic; `per_test_query_recall_hat` fixed it by estimating the
  query's own K-NN radius directly. Verified against exact and noisy ground truth.
- **IF oracle-radius confound**: IF's query-side radius was historically the
  brute-force ground-truth K-NN radius, not an estimate - found and fixed by moving
  it onto the same estimated radius as `per_test_query_recall_hat`.

Each of these was validated at the time on its own terms.

## The late-discovered problem

Every one of the above - including the "fixed" IF - requires building or consulting a
structure with full-corpus access at query time. The actual system constraint only
allows full-corpus access at construction time; query time only has the incoming
query and cached entries' own stored data. This invalidates all of the above as
deployable, independent of their individual accuracy.

## Two in-constraint alternatives tried, both failed

- **Bucket-as-cap** (model a bucket cell as an equivalent-volume spherical cap):
  catastrophically wrong shape - bucket points sat 33-83 degrees from the modeled
  center.
- **Orthant-probability estimation** (multivariate-normal bucket-cap overlap):
  computationally tractable, but a real, systematic, growing overestimation bias
  (1.4x-3x+) in exactly the small-radius/high-separation regime that matters, and
  unverifiable past d=128 via Monte Carlo.

## Successor

Inverse-coverage counting - see the feasibility check under `schemes/inverse_coverage_feasibility_run_experiment.py`
and `analysis/inverse_coverage_feasibility_plots.py`.
