# LSH-Bucket Radius Estimation: Bias Analysis

Estimates the distance to the K-th nearest neighbor (the "radius") of a test
query by averaging ground-truth radii of train queries that fall into nearby
LSH buckets (within a Hamming threshold). Uses plain random-hyperplane LSH.

**All results below are on synthetic data.** They are mechanism demonstrations,
not measurements on SIFT/ESCI.

## Files

- `lsh_radius_estimator.py` - random-hyperplane LSH estimator + exact radius.
- `scenarios.py` - the five data/query scenarios.
- `analytical.py` - bias decomposition and summary statistics.
- `run_analysis.py` - driver: runs all scenarios, writes CSV/JSON/plots.

Run:

```
python simulations/radius_estimation/run_analysis.py \
    --k 10 --dim 8 --n-bits 16 --hamming 0
```

## The question

Is the LSH-bucket average an unbiased estimator of a test query's true radius?
If biased, does the bias depend on the magnitude of the true radius? And how do
uniformity vs non-uniformity of data / train queries / test queries matter?

## Framework

The true radius field `r_true(x)` (distance to the k-th data vector) depends
only on the DATA. Train/test query distributions only decide where the field is
observed and evaluated. In the infinite-train limit the estimator converges to
a train-density-weighted average of `r_true` over the bucket region R:

```
err(q) = E_{p_train | R}[r_true] - r_true(q)
       = (r_bar_R - r_true(q))         # smoothing / regression-to-the-mean
       + (E_w[r_true] - r_bar_R)       # count-weighting
```

- **Smoothing term**: non-zero whenever `r_true` varies within R (non-uniform
  data). Produces regression to the mean: over-estimate small radii,
  under-estimate large radii.
- **Count-weighting term**: non-zero only when train-query density correlates
  with `r_true` inside R (queries tracking the data). Zero for uniform queries
  or a flat field.

## The five scenarios

| Scenario | Field varies? | Queries track data? | Isolates |
|---|---|---|---|
| uniform_data_uniform_queries | no | no | control (both off) |
| uniform_data_queries_from_data | no | yes | weighting on a flat field |
| clustered_data_uniform_queries | yes | no | smoothing alone |
| clustered_data_queries_from_data | yes | yes | both, aligned (realistic) |
| clustered_data_tight_queries | yes (globally) | yes | low LOCAL variation |

## Headline findings (synthetic)

Config: k=10, dim=8, n_bits=16, Hamming=0, n_data=5000, n_train=4000,
n_test=2000.

| scenario | mean_err | stdComp | bias_lo | bias_hi |
|---|---|---|---|---|
| uniform_data_uniform_queries | -0.0005 | 0.33 | +0.055 | -0.072 |
| uniform_data_queries_from_data | +0.0000 | 0.34 | +0.056 | -0.067 |
| clustered_data_uniform_queries | +0.0010 | 0.46 | +0.199 | -0.197 |
| clustered_data_queries_from_data | +0.0002 | 0.24 | +0.029 | -0.040 |
| clustered_data_tight_queries | +0.0025 | 0.29 | +0.020 | -0.021 |

- `stdComp = std(estimate) / std(true)` - fraction of true-radius variation the
  estimator recovers.
- `bias_lo / bias_hi` - mean error in the lowest / highest true-radius bin.

### 1. The aggregate (mean) error is ~0 in every scenario

Averaged over the test distribution, the estimator looks nearly unbiased
(|mean_err| < 0.003 everywhere). Taken alone this is **misleading**: the mean
cancels a strong radius-dependent bias.

### 2. The error is strongly radius-dependent (the real result)

Every scenario over-estimates small radii (`bias_lo > 0`) and under-estimates
large radii (`bias_hi < 0`). This is regression to the mean: the Hamming-ball
average compresses the radius toward the regional mean. `stdComp` is 0.24-0.46
everywhere - the estimator discards 54-76% of the true-radius variation even in
the best configuration found.

### 3. Non-uniform data makes it worse; count-weighting partly rescues it

`clustered_data_uniform_queries` (smoothing on, weighting off) has the largest
per-bin bias (bias_lo +0.199, bias_hi -0.197): a varying field with queries that
do NOT track it is the worst case, because buckets average wildly different
radii with no corrective weighting.

Adding query-tracking (`clustered_data_queries_from_data`) shrinks the per-bin
bias (+0.029 / -0.040) because train queries concentrate where the test queries
are, so the average is taken over a more relevant sub-population. The
count-weighting term opposes the smoothing term (see caveat below).

### 4. Local co-location is what actually helps

`clustered_data_tight_queries` has small per-bin bias (+0.020 / -0.021) despite
a globally non-uniform field: when train and test queries are co-located, the
field is nearly constant over the points actually averaged. **It is local field
variation across the sampled points, not global non-uniformity, that drives the
bias.**

### 5. Plain random-hyperplane LSH is a weak basis in moderate dimension

Even in dim=2, the best `stdComp` observed was ~0.29; Hamming>=1 collapses the
estimate to near-constant (`stdComp` ~ 0.03-0.11). Random-hyperplane buckets are
large cones that do not localize the radius field; you cannot shrink them
without buckets going empty. This limits the method independent of the bias
mechanisms above.

## Answering the original questions

- **Unbiased?** Not pointwise. The aggregate mean error is ~0, but that hides a
  systematic radius-dependent bias.
- **Does bias depend on true radius?** Yes, strongly and consistently:
  over-estimate small radii, under-estimate large radii (regression to the
  mean), in every scenario.
- **Role of uniformity?** Non-uniform DATA drives the bias (varying field).
  Query distribution modulates it: uniform queries over a varying field are
  worst; queries co-located with test queries are best. Query non-uniformity
  over a FLAT field adds essentially no bias.

## Caveats and limitations

- Synthetic data only; Euclidean distance on raw vectors.
- The smoothing/weighting decomposition uses a uniform-in-cube probe to estimate
  the unweighted regional mean. For clustered data a bucket spans a cluster plus
  large empty space, so the uniform-probe mean is dominated by empty-space
  (large-radius) points while the train mean reflects in-cluster (small-radius)
  points. The two terms are therefore large and nearly cancel; treat them as
  "the two opposing forces" rather than a precise attribution. The `stdComp`
  and binned-bias metrics do not depend on the probe and are the primary
  evidence.
- The global-mean fallback never triggered here (fallback=0%); its bias
  contribution is untested in these runs.
- Results depend on k, n_bits, Hamming, dim; the driver exposes all as flags.
