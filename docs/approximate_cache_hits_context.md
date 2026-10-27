# Approximate Cache Hits via Radius Estimation: Problem Context

Working notes that capture the current problem framing, what has been learned
so far, and what is still open. Read this before continuing the radius
estimation / approximate-hit work.

## 1. Background: exact vector caching (the original project)

- The cache stores past queries Q with their exact top-K results R_Q (a "ball"
  around Q of radius r_Q = distance to the K-th result).
- Two hit policies certify that a new query q can be answered from cache with
  results identical to a full search:
  - CIG (Circular Inclusion Guarantee): cached vector E is certified for q if
    D(q, E) + D(q, Q) < r_Q. Implemented per cached vector in
    `utils/base/lemmas.py`; a hit needs at least k certified vectors.
  - HGG (Half-Gap Guarantee): D(q, Q) < half the gap between Q's K-th and
    (K+1)-th neighbor.
- Cache population: candidates are observed query balls expanded by a factor
  alpha (`utils/base/population.py`); selection picks candidates under a budget
  B measured in distinct cached base vectors (`utils/base/selection.py`,
  `utils/base/selection_fast.py`).
- Note: the CIG/HGG hit checks do NOT use the test query's own radius. The true
  test radius appears only offline, in `propose_candidates`, to compute which
  queries each candidate ball would cover.

## 2. Why exact hits are rare

For k > 1, high exact-hit rates require cache balls that are huge relative to
the budget. Working interpretation: the condition D(q, Q) + r_t < r_c forces
the cache ball to enclose the query's whole k-NN ball, and in high dimensions
that region contains a large share of the data. Caveats raised in discussion:
- The economic bar is whether a cache is worth having at all (hit rate vs
  cache size as a fraction of the data), not improvement over random.
- Dimensionality is a major factor but not the only one; query/data skew and
  workload locality also matter.
- Non-ball cache shapes were discussed; they lose the clean triangle-inequality
  proofs.

## 3. New direction: approximate hits with a user-supplied target

- Allow a cache hit even when CIG/HGG does not hold, as long as quality stays
  acceptable.
- The user supplies a target T for a quality metric defined relative to the
  full search (working name: "retention"; see terminology below).
- Hit policy: keep the long-term running average of estimated retention at or
  above T.
- To decide whether a given query can be a hit, estimate the retention we
  would get by answering it from cache.

### Why we need the test query's radius

Let r_t be the distance from q to its k-th nearest data vector, T_q its true
top-k, and C the cached vectors considered for q. Assuming no distance ties:

- T_q = {x in data : D(q, x) <= r_t}.
- If the cache answer A is the k nearest vectors to q within C (and |C| >= k),
  then A intersect T_q = C intersect T_q. Every cached vector within r_t ranks
  ahead of every cached vector outside r_t, and there are at most k of them.
- Therefore recall(A) = |{c in C : D(q, c) <= r_t}| / k exactly.

So with r_t known, retention follows exactly from distances to cached vectors
(no full search). We do not know r_t, so we estimate it (r_hat) and compute
estimated retention = min(1, |{c in C : D(q, c) <= r_hat}| / k).

## 4. Key risks for this approach

1. Error amplification. Count of points in a ball grows roughly like
   r^(intrinsic dimension). Illustration: at intrinsic dimension 15, a 10%
   radius overestimate inflates the count about 1.1^15 = 4.2x. Radius RMSE is
   therefore the wrong evaluation target; evaluate retention-estimate error
   directly.
2. Dangerous bias direction. Overestimating r_t overestimates retention, which
   is the unsafe direction (claims hits that are worse than the target). The
   estimators tested so far overestimate small radii, which occur in dense
   regions, where an overestimate pulls in the most extra vectors.
3. Selection effect (winner's curse). Admitting hits preferentially when
   estimated retention is high selects queries whose retention was
   overestimated. The running average of estimated retention over admitted
   hits will exceed the true average even for an estimator that is unbiased
   overall. A margin or calibration step is likely needed; this can be
   measured offline against ground truth.
4. Estimator footprint. The estimator's memory must be small relative to the
   cache budget B (budgets in this repo range from 250 to 50,000 vectors), and
   its lookup must be cheaper than the search it avoids. An estimator that
   stores a large vector set defeats the purpose.

## 5. Definitional decisions

Decided:
- Terminology: "retention" = fraction of the full search's result set that the
  cached answer keeps.
- Running average is over all queries; misses count as retention 1.
- Retention is relative to the backend's actual result set. For simplicity the
  backend is assumed to do exact nearest neighbor search.
- C is the union of all cached vectors; it can be evaluated one cache entry
  (ball) at a time.
- Track both radius error and retention-estimate error; retention-estimate
  error is the more directly relevant one.
- k (query result count) vs K (vectors stored per cache entry): keep distinct.

Consequence to resolve: with misses = 1 and only an average target T, an empty
cache can label a (1 - T) fraction of queries as hits (returning nothing) and
still meet T. A per-hit retention floor is needed; with floor = T the average
target holds automatically and the policy reduces to "hit iff retention >= T".

## 5b. End-to-end experiment (planned)

- Goal: best-case hit rate vs cache size. x = cache size as a fraction of the
  database (distinct vectors), y = fraction of test queries that hit, one line
  per target retention. Success = a reasonably high target whose line is
  above y = x.
- Best case uses the true test radius (oracle), so retention = |true top-k
  ids present in cache| / k. Estimators are added later as extra lines.
- Data: SIFT (ESCI later). Database = 50k SIFT base subsample; train = SIFT
  learn queries; test = official SIFT query set; test k = 10.
- Offline population: candidates = top-K ball of each train query for several
  K values. Greedy selection by utility = (additional train hits) /
  (additional distinct cached vectors), computed by rerunning the hit policy
  with and without the candidate.
- Known issue: threshold-based hits make marginal hits often 0 for useful
  candidates (greedy stalls). Plan: tie-break by marginal capped-retention gain
  per new vector.
- Exclude a candidate's source query from its own utility (or split train into
  candidate-generating and scoring subsets).
- Expectation: SIFT queries follow the data distribution with no workload
  skew, so lines near or below y = x are plausible; ESCI has real query skew.

Decided after review: the hit policy uses a hits-only running average
(misses excluded), which removes free hits at zero cache size and still
implies the all-queries average (misses = 1) is >= T. Candidate-generating
and scoring train queries are disjoint halves; greedy tie-breaks by retained
top-k vectors per new cached vector.

## 5c. End-to-end oracle results on SIFT (first run)

Code: `simulations/approximate_hits/` (`greedy_population.py`,
`run_e2e_oracle_sift.py`); tests in `tests/approximate_hits/`.
Setup: 50k SIFT1M base subsample, 5k candidate-generating + 5k scoring learn
queries, 10k official test queries, k = 10, candidate K in
{10, 20, 50, 100, 200, 500}, true radius (oracle), best-case admission.

Test hit rate at 10% / 30% cache size:
- T = 1.0: 5.4% / 18.8% (well below y = x)
- T = 0.95: 9.7% / 33.9% (about y = x)
- T = 0.9: 12.7% / 39.8% (~1.3x above y = x)
- T = 0.8: 17.9% / 54.5%
- T = 0.7: 24.5% / 69.0%
- T = 0.5: 43.2% / 100%

Other observations: online admission is ~5-15% (relative) below best case;
train (scoring) hit rate exceeds test (e.g. 17.4% vs 12.7% at T = 0.9, 10%
cache), so greedy overfits to the scoring queries.

Effective retention (all test queries, misses = 1) = 1 - hit rate +
retention-weighted hit rate. It is far above T when the hit rate is low
(e.g. T = 0.9, 10% cache: 98.7%) and approaches T only as the hit rate
approaches 100% (T = 0.5, 30% cache: 53.8%). A dev query set is planned
before tuning any knobs (K values, number of scoring queries, tie-break),
and more scoring queries is the lever for reducing the overfitting.

## 5d. Follow-up runs (more scoring queries, ESCI, synthetic skew)

Results in `simulations/approximate_hits/results/<run>/`; print tables with
`python simulations/approximate_hits/summarize.py results/*/results.json`.

- More scoring queries (5k candidate-generating queries throughout): the
  train/test gap closes (T = 0.9, 10% cache: train/test 17.4/12.7 at 5k
  scoring, 15.1/14.4 at 20k, 14.8/14.5 at 40k) and test hit rates rise
  slightly. 20k is enough; 40k adds little.
- ESCI (real queries, 20k scoring): much better than SIFT. T = 1.0: 7.0% hits
  at 1% cache, 15.3% at 10%, 31.9% at 30%. T = 0.9: 12.4% / 30.9% / 63.9%.
  Only 0.14% of test queries have an identical-embedding train query, so the
  gain is semantic locality, not repeats. Untested hypothesis: hub products
  appearing in many queries' top-k contribute.
- ESCI embeddings were rebuilt locally (not Anupta's files):
  `datasets/esci/build_embedding_subsample.py`, 50k random products, all
  130,652 queries, paraphrase-multilingual-MiniLM-L12-v2, run with the repo
  `.venv` (torch + sentence-transformers, isolated from system python).
- Synthetic simZipf (Aker, Sec. 5.1; base = SIFT query set, 50 variants per
  group via interpolation with lambda ~ U(0, 0.5)): Zipf 0.30 behaves like
  real SIFT queries; 0.60 helps moderately; 0.99 is strongly favorable
  (T = 1.0: 23% hits at 1% cache). At 0.99, 48% of test requests are exact
  repeats of a scoring query. A small-Gaussian-perturbation variant (noise
  10% of norm, Zipf 0.99) is even more favorable (T = 1.0: 37% at 1%).

## 5e. Provable retention lower bounds (certificates) on oracle-selected caches

Code: `simulations/approximate_hits/eval_certificates.py` (writes
`certificates.json` per run), `plot_certificates.py`. Certificates: exact CIG
(union), A1 partial CIG, A3 neighbor-distance profile (next M neighbor
distances + ids per entry; intruders bounded via the ball of radius D_k + d
around Q). Admission uses the lower bound in place of true retention. Zero
bound violations across all runs (bounds never exceeded true retention).

Result: deterministic certificates recover very little of the oracle hit rate
(online, T = 0.95, 10% cache): ESCI 0.11% vs oracle 23.1%; simZipf 0.6: 2.4%
vs 14.1%; simZipf 0.99 interp: 24.9% vs 52.1% (mostly exact-repeat queries);
small-Gaussian 0.99: 7.0% vs 68.9%. A3 with M = 100 ~ M = 1000, so profile
length is not the bottleneck. Diagnostic (ESCI, retention >= 0.9 queries):
d(q, nearest entry) is ~0.8 x D_k, and the best entry's R / r_Q has median
1.65 (needs <= 1); the triangle-inequality region around Q is far larger than
the region that matters around q. Caveat: caches were selected for oracle
retention, not for the certificate. Implication: provable bounds are too
loose on real embeddings; statistical approaches (calibrated prediction, audit
sampling) are the more promising route.

## 5f. Calibrated admission (B5) on the same oracle-selected caches

Code: `simulations/approximate_hits/eval_calibrated.py` (writes
`calibrated.json`), `plot_calibrated.py`. Ridge predictor on 12 cache-only
features (+ squares) fit on 20k fresh FIT queries; admit iff score >= lambda,
lambda chosen on 20k fresh CALIBRATION queries via an empirical-Bernstein LCB
>= T, Bonferroni over a 101-point grid of FIT-score quantiles, delta = 0.05.
Fit/calibration queries are disjoint from the scoring queries used for cache
selection (ESCI: unused queries from the pool; simZipf: fresh draws, seed+2).

Results (test hit rate at 1% / 10% / 30% cache):
- T = 0.9: ESCI 8.0 / 18.4 / 45.0 (oracle online 11.6 / 28.7 / 60.2);
  simZipf 0.6: 3.4 / 13.1 / 34.2; simZipf 0.99: 26.9 / 56.6 / 81.0;
  gauss 0.99: 45.4 / 73.7 / 92.0. Certificates were ~0.05-27%.
- T = 0.95: ESCI 0 / 13.4 / 29.2 (oracle 9.6 / 23.1 / 48.3); bound too wide
  at small caches where few calibration queries are admissible.
- T = 1.0: zero everywhere; a statistical bound can never certify a mean of
  exactly 1. Certificates are the only option at T = 1.
- Guarantee held on test in all 204 evaluated states (realized hit retention
  >= T); the naive threshold (no confidence bound) fell below T in 13 states.
- Predictor ranking quality is moderate (test Spearman median 0.47-0.72), so
  better predictors / tighter confidence bounds are the main levers.
- Assumes calibration and test queries come from the same distribution; under
  drift, audit sampling (C7) is needed.

## 5g. Comparing threshold-selection procedures (repeated trials)

Code: `simulations/approximate_hits/eval_calibration_methods.py` (writes
`calibration_methods.json`), `summarize_calibration_methods.py`; tests in
`tests/approximate_hits/test_calibration_methods.py`. Per run: 8 cache
states (T in {0.9, 0.95} x ~1/5/10/30% cache), 200 trials each; each trial
draws 20k calibration queries from a large fresh pool (ESCI: ~75k unused
queries; simZipf: 150k fresh requests) and measures the admitted queries'
mean retention on the rest of the pool (stand-in for the truth).

Precedent: BARGAIN (Zeighami et al.) Alg. 1 = fixed-order testing (strict to
lenient, full level, stop at first failure; Lemma 3.5 allows eta tolerated
failures at level delta/(eta+1)). SUPG (Kang et al.) uses the normal
approximation; its precision-target algorithm (Alg. 3) uses a union bound
(Bonferroni) over candidates.

Results:
- Failure rate (target <= 0.05): naive 7-21% on average (up to 43% in one
  state); all guaranteed methods <= 1.6% on average. Normal + fixed order is
  the least conservative (worst state 9%, plausibly evaluation noise plus
  max over 40 states).
- Hit rates: Normal + fixed order is best among guaranteed methods and
  nearly matches naive (ESCI T = 0.95: 6.7 / 14.4 / 30.5% at 1 / 10 / 30%
  cache vs EB + Bonferroni 1.1 / 12.3 / 28.2%).
- EB + fixed order sometimes collapses (stops at the first strict threshold
  because EB's margin is wide when few queries are admitted).
- Normal + split is valid but loses a lot (halved data).
- Remaining gap to the oracle is mostly predictor quality, not the bound.
- Recommended: normal approximation + fixed-order testing.
- Adopted as the default in `threshold_selection.py` (shared by
  `eval_calibrated.py` and `eval_calibration_methods.py`); `calibrated.json`
  files regenerated with it. ESCI T = 0.95 guaranteed hit rate at
  1 / 10 / 30% cache is now 5.9 / 14.6 / 31.4% (was 0 / 13.4 / 29.2%).

Evaluation sizes (all k = 10): database 50k vectors (ESCI: 384-d product
subsample of 1.8M; simZipf: 128-d SIFT1M base subsample). ESCI queries:
5k candidate-generating + 20k scoring (cache selection), 20k fit, 20k
calibration (single draw) or a 75,652-query pool (repeated trials), 10k test.
simZipf requests: 5k candidate + 20k scoring, 20k fit, 20k calibration or a
150k-request pool, 100k test (45k-89k distinct vectors); one seed.

## 5h. Predictor improvements, k sweep, database-size sweep (in progress)

Code: `eval_predictors.py` (fit/dev/cal/test splits; run with `.venv`
python, needs scikit-learn), `summarize_predictors.py`, `summarize_trends.py`,
`run_eval_queue.py` (runs an eval script on runs as their greedy results
appear). Runs: `results/batch_k_db/` (k = 1/50/100 at 50k; k = 10 at 200k),
`results/batch_db1m/` (k = 10 at 1M; ESCI 1M embedding built locally).
Note: k = 50 / 100 runs use 2k / 1k candidate-generating and 10k scoring
queries (vs 5k / 20k at k = 1 / 10), which confounds the k comparison.

Predictors (50k, k = 10; mean over 8 states, dev-selected): gradient-boosted
regression (per state or pooled across states) is best; ESCI mean guaranteed
test hit rate 19.5% (ridge, base features) -> 21.1% (GBT); largest gains at
30% cache and T = 0.95 (ESCI 31.4 -> 35.3%, simZipf 0.6 24.9 -> 33.5%).
Classifier and 20th-percentile quantile models are worse. Extended features
help ridge too. On simZipf 0.99 the guaranteed rate already ~ oracle online.

k sweep (T = 0.9, ESCI, oracle online -> guaranteed, ridge, at 1/10/30%):
k=1 28.9/61.8/85.1 -> 9.9/31.0/61.2; k=10 11.6/28.7/60.2 -> 9.0/20.4/46.9;
k=50 9.1/21.1/45.4 -> 7.9/16.6/36.0; k=100 7.2/19.1/40.8 -> 5.7/13.7/33.0.
Oracle hit rate falls with k; the predictor gap is largest at k = 1 (binary
retention; several features degenerate when k = 1).

Database sweep (k = 10, oracle online at 10% cache): ESCI 28.7% (50k) ->
35.2% (200k); simZipf 0.6 17.4 -> 23.4 -> 33.8% (1M); simZipf 0.3 11.4 ->
11.7 -> 18.7%. A fixed cache fraction serves more queries as the database
grows. Unexplained: guaranteed rate for simZipf 0.3 drops at 200k (8.1 ->
5.3% at 10% cache).

Final (all runs complete): database sweep, oracle online T = 0.9 at 10%
cache: ESCI 28.7 (50k) / 35.2 (200k) / 37.5% (1M); simZipf 0.6 17.4 / 23.4 /
33.8%; guaranteed (ridge) ESCI 20.4 / 25.6 / 25.2%. Predictors: a GBT model
(per state or pooled) is dev-best in every k >= 10 run; the GBT classifier is
dev-best at k = 1 (binary retention). Mean guaranteed test hit rate, ridge ->
best: ESCI 50k 19.5 -> 21.1, 200k 22.8 -> 24.0, 1M 22.2 -> 23.4, k=1 24.4 ->
25.9 (oracle 54.9). Worst realized test retention minus T across all runs and
models: -0.003 (within test noise).
Mechanism check: hit rate rises monotonically with N at fixed k and falls
with k at fixed N, but k/N alone does not explain it (k = 1 at 50k, k/N =
2e-5, far exceeds k = 10 at 1M, k/N = 1e-5): k also hurts because all k
specific vectors must be cached, and nearby queries share their first few
neighbors far more than their 10th.

## 5i. Aker workloads (10M databases, k = 10, cache up to 5%)

Code: `run_e2e_oracle_trace.py`; runs in `results/batch_aker/`. Time-ordered
splits of each ~100k-request trace: cand 8%, score 32%, fit / dev / cal / test
15% each. spacev-real: SPACEV1B historical query log for all training splits,
official 29,316-query set as test. Exact neighbors (blocked top-k, verified
against brute force) saved per run in `topk.npz`.

Hit % at 0.1 / 1 / 5% cache, T = 0.9, oracle best case -> guaranteed (dev-best
predictor, GBT in all runs):
- SPACEV Aker Zipf 0.3: 4.7 / 14.2 / 25.7 -> 3.2 / 10.9 / 20.9
- SPACEV Aker Zipf 0.6: 12.9 / 25.4 / 36.8 -> 10.8 / 22.4 / 33.0
- SPACEV Aker Zipf 0.99: 49.6 / 59.9 / 67.1 -> 48.8 / 58.7 / 64.9
- TriviaQA/SPHERE Zipf 0.3: 7.2 / 39.7 / 87.3 -> 2.9 / 27.7 / 79.2
- TriviaQA/SPHERE Zipf 0.6: 17.0 / 52.4 / 91.1 -> 14.9 / 44.7 / 85.7
- TriviaQA/SPHERE Zipf 0.99: 61.7 / 84.8 / 100 -> 60.9 / 82.3 / 100
Guarantee held on all Aker traces (worst realized test retention - T:
-0.002, within test noise).

spacev-real: oracle hits ~0 on the official test queries, and the guarantee
FAILED (realized retention up to 0.68 below T). Cause: distribution shift.
Official queries are farther from the log (median nearest-log-query distance
134 vs 99 within the log); with a cache built from the log, 7.4% of
calibration (log) queries reach retention >= 0.9 vs 0.0% of test queries.
Calibration assumes future queries resemble calibration queries; under shift
it gives no protection. Motivates audit sampling (C7) / drift detection.

Figure: `results/summary/fig4_aker.png` (red x = realized test retention
below T).

Comparison with the Aker paper: Aker never evaluates a historical-log ->
official-query split. Its lowest-locality "uniform" workload replays the
unmodified SPACEV / TriviaQA queries, where repeated queries make equality
hits dominate (Fig. 9). Aker also populates online and reports recall rather
than a guaranteed retention target, so its hit ratios are not directly
comparable to ours.

## 5j. Aker's hit rule on our offline caches

Code: `eval_aker_thresholds.py` (writes `aker_thresholds[_ceiling].json`),
`summarize_aker_thresholds.py`. Same greedy caches (T = 0.9), only the hit rule
changes: Aker's per-entry tau (init minD / 4, x1.1 on equality hits, x0.25 on
approximate hits, cap minD; aliases for repeated approximate-hit queries; one
representative per cached query, nearest representative only). No insertion
or eviction. Cold = test requests only; warm = fit + dev + cal replayed first.
Ceiling = static rule d < minD (loosest Aker allows).

Hit % (mean union retention over hits) at 1% cache:
- SPACEV 0.3: ours 10.9 (0.92); Aker cold 1.3 (0.97); ceiling 22.2 (0.54)
- SPACEV 0.6: ours 22.4 (0.91); Aker 2.8 (0.99); ceiling 33.2 (0.65)
- SPACEV 0.99: ours 58.7 (0.91); Aker 16.0 (1.00); ceiling 61.9 (0.82)
- TriviaQA 0.3: ours 27.7 (0.91); Aker 4.1 (0.96); ceiling 93.6 (0.55)
- TriviaQA 0.6: ours 44.7 (0.90); Aker 8.2 (0.98); ceiling 94.2 (0.62)
- TriviaQA 0.99: ours 82.3 (0.90); Aker 36.9 (1.00); ceiling 95.9 (0.81)
- spacev-real: Aker 0 hits; ceiling 0 hits

Aker's rule is far more conservative than needed (retention 0.94-1.00) and
gets 4-10x fewer hits; most of its hits are equality hits. Warm-up lowers its
hit rate further: approximate hits shrink tau 4x and only equality hits grow
it back, which are rare with few representatives (Aker's own cache inserts on
every miss, so it has many more). The static ceiling gets more hits but
retention falls to 0.2-0.85. The rule has no target knob and no guarantee.

Cache dependence of our admission: features are cache-relative (any cache),
the per-state GBT is fit on one cache, gbt_pooled on the 8 nested checkpoints
of one greedy run (log cache size feature), entry-history features are tied
to specific entries, and lambda is calibrated per cache state; the guarantee
holds only for the cache it was calibrated on.

## 5k. Does a calibrated threshold transfer across caches?

Code: `eval_threshold_transfer.py` (writes `threshold_transfer.json`),
`summarize_threshold_transfer.py` (add `--matrices` for per-run matrices).
`eval_predictors.setup_run` was extracted so both scripts share features,
GBT and lambda selection. For one run, build a cache family: nested size
checkpoints, caches selected for other targets, and the base cache with a
fraction of entries replaced by random unselected candidates of the same K.
For every ordered pair (source A -> destination B), admission target T = 0.9:
- frozen: A's GBT and A's lambda, applied to B's test queries
- recal: A's GBT, lambda recalibrated on B's cal queries
- native: B's own GBT and lambda (the diagonal)
Features on B are always computed against B, as an online system could from
stored backend results. Ran all 7 Aker traces + ESCI 1M.

Off-diagonal pairs with realized test retention below T - 0.01 (worst gap):
                      frozen            recal
- larger cache:       0% (best case)    0%
- smaller cache:      33-67% (-0.33)    0%
- other-target cache: 6-39% (-0.12)     0%
- entries replaced:   10-36% (-0.13)    0%

Conclusions:
- lambda does NOT transfer. A frozen lambda is only safe moving to a LARGER
  cache (a strict superset retains at least as much), consistent with it being
  calibrated to one cache. Moving to a smaller, perturbed, or differently
  targeted cache violates the guarantee up to 2/3 of the time, worst -0.33.
- The GBT (score function) DOES transfer. After recalibrating only lambda on
  the destination's cal queries, the guarantee holds on every pair (worst
  +0.000) at a median 0.95-1.00 of native hit rate. So the expensive part
  (fitting the predictor) can be reused; only the scalar lambda must be
  refreshed as the cache changes.
- Implication for online population / eviction: keep the predictor, recalibrate
  lambda whenever the cache changes materially (cheap: recompute retention of
  recent queries against the current cache from stored exact results). Still
  needs an unbiased calibration sample (misses and insertion-driving queries
  are biased) -> audit sampling (C7). ESCI 1M frozen is the worst larger-cache
  case (sources as small as 1% transferred up to 5-10%): the score scale from a
  tiny cache is miscalibrated for a much bigger one, so even the superset
  direction violates unless lambda is recalibrated.

## 6. Radius-estimation findings so far

End-to-end comparison (`eval_radius_baselines.py`, writes
`radius_baselines.json`; figures via `make_summary_figures.py` into
`results/summary/`): on the same caches and queries (k = 10, 50k database),
LSH buckets (config chosen on dev) cut radius RMSE only 22% (ESCI: 8.5% vs
10.9% of mean radius) / 23% (SIFT: 13.2% vs 17.1%) vs the global mean, and
give no better retention estimates (ESCI, T = 0.9, 10% cache: MAE 0.196 LSH,
0.184 global, 0.152 ridge, 0.137 GBT, 0.005 true radius). Trusting LSH
retention estimates realizes 0.77 (ESCI) / 0.60 (SIFT) retention at T = 0.9;
calibrated admission with LSH- or global-mean-based scores certifies no
threshold (0% guaranteed hits in every state), vs GBT 22.2% (ESCI) / 9.8%
(SIFT) and true radius 25.8% / 12.2% (oracle best case 30.9% / 14.4%).

Code: `simulations/radius_estimation/`.

Estimator studied: plain random-hyperplane LSH. Store the average train-query
radius per bucket; estimate a test query's radius as the count-weighted average
over buckets within a Hamming threshold, expanding the threshold when empty.

Synthetic five-scenario study (`run_analysis.py`, `scenarios.py`):
- The average error over the test distribution is near 0, but this hides a
  strong radius-dependent bias: small radii are overestimated, large radii
  underestimated (regression to the mean).
- Clustered data drives the bias. Its impact depends on whether train queries
  sit where test queries sit; tightly co-located train/test queries give low
  bias.

Real SIFT (SIFT1M 50k base subsample, queries from the learn set, k = 10):
- Train and test following the data (`run_granularity_sweep.py`): true radius
  mean ~256, std ~43. Best RMSE ~34 vs 43.3 for always predicting the global
  mean, so skill (1 - RMSE/RMSE_global) peaks around 0.21 (16 bits, 4
  tables). Per-decile bias is about +55 (lowest radii) / -50 (highest).
- Granularity has a sweet spot. More bits beyond ~16 or more tables beyond ~4
  hurt (finer buckets go empty; more tables oversmooth).
- More train queries help modestly (skill ~0.05 to ~0.16 from 1k to 24k).
  Pairing more train data with finer buckets does not lift the ceiling,
  because finer random-hyperplane buckets starve faster than extra data
  refills them (`run_joint_bits_train.py`).
- Asymmetry mechanism (`run_sift_train_control.py`): with data-following train
  queries, high-radius buckets are starved (train queries per bucket drop
  ~4x from low to high radius) and borrow from denser, smaller-radius
  neighbors. With uniform train queries the starvation disappears (but the
  estimator is useless on clustered data, mean error +530).
- Density-feature binning (`run_trainsize_and_partition.py`): bin by distance
  to the k-th nearest train query, then average radius per bin. RMSE 11.4,
  skill 0.74, vs plain LSH 37.1 / 0.15 at matched bin count. Caveats:
  - It stores all train query vectors (16k, 32% of the database) and needs a
    k-NN search over them per query, so it fails the footprint requirement.
  - It works only because train queries follow the data distribution; under
    skewed workloads it measures query popularity, not data density.
  - Off-by-one bug: train features exclude self, test features do not, so
    test features run one neighbor further out.
- Comparing against Anupta's stages: Stage 1 (uniform DB, SIFT queries) is
  reproduced as symmetric regression to the mean. Stage 3 (SIFT DB, SIFT
  queries) shows the starvation mechanism. The fully one-sided curve (low
  radii accurate, high radii underestimated) was not fully reproduced.

## 7. Candidate next steps

- Evaluate retention-estimate error and its bias directly, not radius error,
  including the selection effect under a running-average policy.
- Compact density-aligned estimator: C cluster centers (e.g. a k-means coarse
  quantizer) with one stored radius each. Sweep C and report skill vs
  footprint as a fraction of the budget B.
- Fix the density-feature off-by-one and test it under queries that do not
  follow the data (e.g. Anupta's locality-constructed stages).

## 8. Environment notes

- SIFT data downloaded to `datasets/sift/raw_siftsmall/` (siftsmall, 10k
  base) and `datasets/sift/raw_sift1m/` (SIFT1M, ~168 MB archive). Source:
  ftp://ftp.irisa.fr/local/texmex/corpus/.
- Installing matplotlib upgraded numpy from 1.21.1 to 2.0.2 on this machine;
  `requirements.txt` does not pin numpy.
- All SIFT radius results above use k = 10 and Euclidean distance on raw
  vectors.
