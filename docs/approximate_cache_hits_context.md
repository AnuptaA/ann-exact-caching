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

## 5l. Radius-based scores, prediction targets, and where the gap comes from

Scripts (all `.venv`): `eval_radius_scores.py`, `eval_radius_models.py`,
`eval_focused.py`, `eval_gap.py`. ESCI 50k unless noted, k = 10, T = 0.9;
"hit" = calibrated test hit rate (normal_fixed), all met T.

- Bucket / past-query radius scores (count cached vectors within plausible
  radii): bucket expected retention + calibration reaches ~50-75% of the GBT
  on ESCI, 0 on TriviaQA (no discrimination). Nearest-past-query radius is
  overconfident (score 0.985 -> actual 0.80). Triangle-inequality bound from
  a past query: valid, useful only on near-duplicate traces.
- Good hits are mostly "cache holds the full true top-k" (6.6-29.7% of ESCI
  test queries at 1-30% cache); then true radius = k-th nearest cached
  distance exactly. Ranking at the top matters, not average accuracy: a
  query-only radius model was the most accurate and got 0 calibrated hits.
- Targets: retention, radius, ratio (true radius / k-th nearest cached
  distance, <= 1). With the same cache features all are within a few points;
  retention >= ratio >= radius. Reweighting / LambdaRank: no gain for
  retention; helps ratio / radius only at k = 1 and tiny caches.
- Gap decomposition (best cutoff chosen on test): ~75-80% of the gap to the
  oracle is ordering (missing information), the calibration margin is 1-3
  points. True database density (20th / 50th neighbor distance; a cheat)
  closes ~60-80% of the gap (ESCI 10%: 21.7 -> 28.2, oracle 30.9).
- Stored distance profiles of nearby entries: no gain. The nearest cached
  center is ~0.8-1.0x the query's 50th-neighbor distance away; estimate
  error (8-15% ESCI, ~2% TriviaQA) exceeds the signal's spread across queries.
- Uniform random sample of the database as probe features: <= +2 points;
  sees intruders mostly when they are plentiful (easy cases); about
  break-even once its memory is charged.
- Targeted shell sample (fraction p of the uncached vectors at ranks K+1..500
  around each cached entry, known from its backend answer): ESCI 10% cache
  21.7 -> 25.3 / 26.0 / 27.3 at p = 5 / 10 / 25% (+3.4 / 6.8 / 16.7% memory);
  30%: 49.1 -> 57.1 at p = 25% (cheat 58.3). TriviaQA 1%: 44.7 -> 50.0
  (+2.0% memory). Memory-fair (base measured at 2 / 15 / 20% cache): at full
  vector cost the shell sample loses or ties everywhere.

TODO (do not forget): store shell / sample vectors COMPRESSED (int8, PQ,
dimension reduction). They only feed coarse distance counts, never answers.
Break-even needs only ~1.2-1.3x compression at 10-30% cache (~3x at 1%); at 8x
the shell sample would net ~+3 points at 10% cache (arithmetic, untested).
Also: shells stop at rank 500 only because candidate generation fetched
top-500; larger shells (and rank-dependent sampling rates) are untested.

## 5m. Feature reset: per-ball stats + shell sample only

Dropped the original 20-feature model as a baseline (an unprincipled mix).
We now compare the oracle against models built only from:
- per-ball stats for the n best-fitting balls (one ball per cached past
  query, its largest K; ranked by fit = relative offset + relative size, the
  worst-case "neighborhood fits inside the ball" test): relative offset =
  offset / ball radius, relative size = neighborhood radius / ball radius;
- the shell sample (fraction p of the uncached vectors at ranks K+1..R around
  each ball, b-bit quantized).
Scripts: `eval_core.py` (knobs n, p, R, b), `eval_simple.py` (model types),
`eval_shell_ablation.py`; diagram `plot_feature_diagram.py`.

ESCI 50k, k = 10, T = 0.9 (calibrated hit rate at 1 / 5 / 10 / 30% cache;
oracle 12.4 / 22.3 / 30.9 / 63.9):
- n: 1 -> 3 balls helps (0.2-3 points); beyond 3 adds <= ~1 point.
- Compression: 8- and 4-bit match uncompressed; 2-bit loses 0-2 points.
  p = 25%, 4-bit costs +0.5 / 1.5 / 2.1 / 2.1% of the database.
- Model type: logistic and linear regression match the GBT within noise; a
  no-fitting rule (fewest sampled intruders, then best fit) is 1-3 points
  behind. No evidence of complex interactions.
- Uncalibrated: admitting iff predicted retention >= T is safe on test
  (realized 0.967-0.981) but gets ~60-70% of the calibrated hit rate, since
  calibration targets the average over hits, not each query.

DECISION: model inputs = 7 features: relative offset and relative size for
each of the 3 best-fitting balls, plus the intruder count (sampled uncached
points inside the neighborhood). Logistic regression on 3 balls + intruder
count: 10.9 / 18.6 / 26.7 / 55.7, vs 11.2 / 19.7 / 27.6 / 56.5 with all 8
shell features. Accepted ~1 point for a much simpler feature set.
FUTURE (only if absolutely necessary): add back shell features, e.g. counts
within 1.1x / 1.25x the neighborhood radius or the nearest sampled distances.

Sanity checks (ESCI 50k, 7-feature logistic): test 10k queries (s.e. of hit
rate 0.3-0.5 points); exact duplicates of cache-center queries 0.02%,
within 0.5x the query's radius 2.4%; mean retention over ALL test queries
0.15 / 0.40 / 0.70 at 1 / 10 / 30% (admit-everything fails T = 0.9);
training on shuffled labels or random scores -> ~0 hits; best ball's
relative offset alone gets ~half the hits; on test queries excluding
near-duplicates the model keeps the same share of the oracle (~87-88%).
Note: best_cutoff_rate cannot split tied scores, so on true retention
(values 0, 0.1, ...) it underestimates the oracle; the recorded greedy
oracle equals the optimal tie-splitting oracle.
Caveats that make the setup easier than deployment: test, calibration and
cache-population queries come from the same distribution (spacev-real shows
shift breaks the guarantee); the cache is chosen offline with oracle
knowledge; the oracle is "best admission for this cache", not a better cache.

Generalization (`eval_generalize.py`, ESCI 50k, T = 0.9):
- Across caches (sizes 1-30%, a T = 0.7 cache, 30% of entries replaced): a
  model fit on any one cache, with lambda recalibrated on the destination,
  is within ~1 point of the native model (worst: 1% -> 30%, 50.2 vs 55.7);
  realized retention >= 0.901 everywhere. Frozen lambda fails (0.83-0.88 on
  some pairs). One pooled model + one shared lambda also fails (0.888 on the
  T = 0.7 cache). => model transfers, lambda must be per cache state.
- Across k (one cache, labels at k = 1, 10, 50, 100): the k = 10 model with
  lambda recalibrated per k matches the native per-k models within ~1 point
  at every k. A pooled model (log k input) is 0.5-1.2 points worse. One shared
  lambda across k fails (0.85-0.87 at k = 50 / 100, over-conservative at k = 1).
  => one model for all k, lambda per k. Share of oracle: k = 1 ~67-84%,
  k = 10-100 ~85-91%.
Also tried (`eval_features_v2.py`): a geometric "region probe" (share of
random data-shaped probe points on the neighborhood boundary outside every
ball). Useless in high dimensions: ~100% uncovered for good and bad queries
alike; the shell sample's intruder count measures the uncovered region with
real points instead. Old 20-feature ablation: `eval_ablation.py`.
TriviaQA (`batch_aker/triviaqa-aker-0.6`): 7 features 15.0 / 46.7 / 86.7 at
0.1 / 1 / 5% cache vs all 8 shell features 16.1 / 49.1 / 87.9 (oracle 17.0 /
52.4 / 91.1); +0.07-0.25% memory; logistic = GBT; 2-bit hurts more than on
ESCI (-6 points at 1%), 4-bit is free.

## 5n. Cost of recalibration (analysis, not yet measured)

Calibration needs a set of n recent queries (we used 20k) with features
computed against the CURRENT cache and their true retention against the
current cache. lambda itself is a sort over n scores (milliseconds). The
model does not need refitting when the cache or k changes (5m).
- New cache state (insert / evict): no backend calls. Keep each calibration
  query's backend answer (top-k_max ids). Recompute retention = fraction of
  the stored ids now cached (set lookups) and features against the new cache
  (one cache-side feature computation per query, like serving it). Cost ~ n
  cache lookups.
- Different k: free if answers were stored at k_max (retention at any k <=
  k_max from the same ids); calibrate lambda per k.
- Ongoing requirement: calibration queries must be an unbiased sample of the
  workload, so a random fraction of hits (not just misses) must also go to
  the backend (audit sampling). Steady cost = audit rate x backend queries.
- Workload shift: old calibration queries no longer represent the workload;
  need ~n fresh queries with backend answers (normal top-k searches, not
  scans of the whole dataset). With a sliding audit window this happens
  continuously; the hard part is detecting shift quickly (spacev-real). The
  model may also need refitting (cross-workload transfer is untested).
- Data shift (inserts / deletes): stored answers go stale. Either re-run the
  n calibration queries on the backend, or update incrementally (each
  inserted vector: compare to the n stored queries' k-th distances; each
  deleted vector: drop it and fetch a replacement, cheaper if top-(k + delta)
  is stored). The cache itself also goes stale (balls stop being complete;
  shell samples too), which is a bigger problem than recalibration and needs
  a refresh mechanism (cf. Aker).
Untested: how hit rate degrades with smaller n (larger safety margin).

## 5o. Online simulator design (milestone 1 not yet built)

Terminology: the SERVE DECISION per query = compute the 7 features against
the current cache, logistic score, serve from cache iff score >= lambda
(answer = k nearest cached vectors, union of all entries), else miss ->
backend. ("Admission" is avoided: in caching it means inserting items.)

Simulation dimensions: fixed vs mixed k; insert / delete rate; data shift;
workload shift; skew / repetition; memory budget; (later) ANN backend.
- Data shift: cluster the database (e.g. k-means, 100 clusters); half the
  clusters are the initial set, half arrive over time (deletes from the
  first, inserts from the second), so shift is regional. ESCI metadata
  (locale, category) is an alternative split. Apply changes in batches
  between query epochs and compute exact top-k once per database snapshot
  (tractable on ESCI 1M or a 1M SPACEV slice, not 10M).
- Workload shift: cluster the queries, run one set of clusters then the
  other; abrupt vs gradual, recurring vs not. spacev-real is a natural
  abrupt shift.

Methods:
- Ours: periodic rebuilds with the offline greedy (fixed interval as the
  baseline; signal-triggered as the main variant), cache and lambda frozen
  within an epoch, recalibration from the audit window.
- Cost-aware population: maximize expected hits over the next epoch minus
  fetch cost (new entries need top-K and shell fetches; already-cached
  entries are free), with a churn discount.
- LRU baseline: insert every miss as an entry with fixed K (e.g. 50); an
  entry is USED when it supplies at least one of the k vectors of a served
  answer; evict least recently used (shared vectors reference counted);
  same serve decision, lambda recalibrated every delta requests (guarantee
  only approximate between recalibrations, since the cache keeps changing).
- Aker's per-entry threshold rule, online.
- Static cache (no rebuild); oracle-lambda upper bound.

Audits (random fraction of all queries also sent to the backend; their
top-k_max ids stored): (1) fresh unbiased calibration data, (2) drift test:
anytime-valid confidence sequence on audited hits' mean retention, alarm when
its upper bound < T -> recalibrate, (3) training data if the model must be
refit (signal: ranking quality on audits drops or lambda collapses hits),
(4) measured delivered retention. Free early warning: two-sample test of
incoming score / feature distribution vs the calibration set. Audit rate from
calibration precision + detection delay (see 5n).

Rebuild signal (utility, separate from calibration validity): shadow rebuild
= run the greedy on the recent window in the background and compare oracle
hit rates of the proposed vs current cache on that window; rebuild when the
gain over the expected epoch exceeds rebuild cost. The greedy works on id
sets only (candidates' top-K ids, window queries' top-k ids), so no 2x
vector memory: ~2 KB of ids per candidate plus the window's query vectors
(count these in the budget). Swap incrementally (evict leaving entries, then
fetch new ones). Cheap proxies: hit-rate drop, clustered misses, many
entries touched by inserts / deletes.

Maintenance under data changes:
- Insert x: for each ball (index over centers) with d(x, c) < radius: add x
  and evict the ball's farthest member (stays exact top-K, smaller radius),
  or keep both (top-(K+1)). Balls stay complete either way.
- Delete x: drop it from any ball; the ball stays complete (K - 1 members
  within the same radius). Optional exact reserve (Aker's delta) to refill.
- Shell sample: maintain incrementally: an inserted x in some ball's shell
  (radius < d <= stored R-th neighbor distance) joins the sample with
  probability p; deleted x is dropped. Alternative: leave stale until rebuild.
- Audited answers: same checks against each stored answer's k-th distance,
  or expire them.

Cost model: backend work in units of searches, distinguishing top-k, top-K
and top-R fetches; count misses, audits and rebuild fetches. Cold start:
warm-up period builds the first cache, model and lambda. Claim under drift:
target holds in stationary stretches; measure violation depth and duration
after each shift. Mixed k: one model, lambda per k, audits at k_max, entry K
chosen for the k mix.
Metrics over time: hit rate, realized retention per window (frequency /
depth / duration below T), total backend work, number of rebuilds.

Milestone 1: ESCI 1M, fixed k = 10, no data changes, abrupt workload shift
(query clusters) plus spacev-real; compare static cache, fixed-interval
rebuild, signal-triggered recalibration + shadow-rebuild triggering, LRU.

TODO (not urgent, do not forget): stop assuming an exact backend. With an
approximate backend (e.g. average recall 0.95 per search), cached entries'
top-K and the shell sample are themselves slightly wrong (balls are no
longer exactly complete), and retention should be measured against what the
backend returns (or against exact ground truth, both reported). Test with a
backend of controlled recall (e.g. drop / substitute a fraction of each
returned list) and with a real ANN index.
Baselines for all drift experiments: Aker (online, per-entry thresholds) and
QVCache, in addition to static / periodic / LRU.

## 5p. Online milestone 1 results (ESCI 1M, k = 10, T = 0.9, budget 10%)

Code: `online_prepare.py` (exact top-500 ground truth, query clusters,
streams), `online_sim.py` (static / periodic / triggered / LRU; ablation
switches), `online_baselines.py` (Aker, QVCache), `online_summary.py`.
Streams: random order (no shift) and an abrupt shift (20 k-means query
clusters split into two halves of ~65k queries; half A then half B).
Warm-up 20k queries, then batches of 1000; audits 10% (60% calibration role).

Bugs found and fixed while building (each made results look better than
they were): calibration audits reused to build caches (guarantee broke);
rebuild candidates overlapping scoring queries (self-serving balls); shadow
comparison on the proposal's own build queries; labeled-only drift test too
slow after a shift (few served queries) -> added a label-free score test;
score test comparing scores of two different caches after a rebuild.

Results (shift stream; hit rate / served retention; after shift = phase B):
- static 28.4% / 0.883 (phase B 0.790; 14 of 23 5k windows below T)
- periodic rebuild every 10k: 32.3% / 0.908
- triggered (score alarm + shadow rebuild): 35.6% / 0.911 (phase B 22.2% /
  0.919; 1 of 23 windows below T, the one containing the shift)
- LRU (K = 50): 27.6% / 0.912, but top-500 fetch per miss, 2.5M vectors
- Aker 0.4% / 0.914; QVCache (D = 0.075) 1.3% / 0.822 (below T). ESCI has no
  repeated queries, which removes Aker's main hit source; QVCache regions
  are mostly unseen. Needs workloads with repetition and tuning.
No-shift stream: static 31.1%, triggered 30.6% (no alarms, no rebuilds),
periodic 27.1% (rebuilds from partly labeled data hurt), LRU 24.4%.

Why phase B hit rate is lower: best offline cache built from phase B's own
queries reaches oracle 41.4% / 7-feature model 34.7% (phase A: 62.6% /
57.1%), so B is harder. Online: 6.2% before the first rebuild, 19.8% after it
(built from only 5k post-shift queries), 27.9% after the second (30k
window): slow adaptation and partly labeled rebuild data.

Trigger ablation (2 seeds, both streams): the retention alarm never fired
(redundant); the score alarm is what keeps the guarantee after a shift (1-2 vs
~10 windows below T); shadow rebuilds are what recover hit rate (phase B ~23%
vs 2.5-7%); per-batch recalibration adds nothing over recalibrating at alarms,
checks and rebuilds. Periodic shadow checks cost ~39k extra top-500 fetches on
the stable stream without ever rebuilding.

Calibration size (independent re-splits, 200 draws): ~2000 calibration
queries reach 95-97% of the 10k hit rate; below-T frequency 0-10% (tiny
shortfalls) at small n, ~5% from 2000 on. Bernstein bound unusable.
Budget split (cache vs shell, chosen on dev, 7 runs): never hurts, gains up
to +10.7 points (ESCI 1M 30%, TriviaQA 0.3 5%), small on SPACEV.

Next: alarm-scheduled rebuild checks (drop retention alarm, per-batch
recalibration, fixed shadow schedule); Aker-trace workloads with repetition
plus spacev-real; tune all baseline parameters.

## 5q. Agreed design space for the next round (after ESCI / SPACEV tuning)

Principle: drop a strategy only if it does not make sense logically, not
because it did nothing in one experiment. Candidate strategies are measured
separately; the final technique keeps the minimal set needed.

Dropped (logically bad):
- Periodic forced rebuilds: a fixed schedule rebuilds in stationary phases
  and the rebuilt cache is worse than the warm-up cache (rebuild data is
  mostly misses + few up-weighted audited hits, noisier than 20k fully
  labeled warm-up queries). Tuning shows a pure tradeoff (ESCI before / after
  shift hit: every 5k 47.3 / 21.5, 20k 50.9 / 22.6, 40k 54.0 / 17.8, static
  54.9 / 10.2). Installing a rebuild only on proven gain = the shadow rebuild.
- Calibrating on pre-shift queries (stale sliding windows): periodic / LRU
  recalibrated often but on a window of the last ~50k queries (3000
  calibration audits at 6% of queries), so lambda stayed too loose for ~40k
  queries after the shift (8-9 of 23 windows below T, phase B retention
  0.89). Every method resets calibration data on an alarm.

Dimension 1, detecting a shift (measure each separately):
- score test: two-sample KS test between the model's scores of the last 2000
  queries and of the calibration queries, both against the current cache.
  Why it means a shift: the guarantee assumes future (score, retention)
  pairs come from the calibration distribution; with cache and model fixed a
  score is a function of the query, so an unchanged query distribution gives
  an unchanged score distribution; a significant difference (false-alarm
  rate controlled) means the query distribution changed. Not proof the
  guarantee fails (scores may change while remaining accurate -> needless
  but harmless conservatism), and blind to changes that keep scores but
  change retention. Label-free, fast.
- retention test: audited served queries average significantly below T
  (Hoeffding). Directly tests the promise; catches what the score test
  cannot (e.g. database inserts change true neighbors while scores stay).
  Slow after a shift (few hits) and needs audits. Never fired on ESCI because
  the score test fired first; keep.
- hit-rate test: two-sided test on the fraction of queries scoring >= lambda,
  recent vs calibration. A one-number special case of the score test (one
  point of the score distribution); cheaper, less sensitive.
- Levels shrink with each check (delta / (j (j + 1))) so false alarms stay
  rare over long runs. After an alarm, keep only calibration data that
  arrived after the alarm (alternative not worth it here: estimate when the
  shift started from the score sequence and keep data from then on; our
  detection took one batch).
- The shadow rebuild is not a shift detector (a better cache can exist
  without a shift); it is a response.

Dimension 2, responding to a shift: recalibrate only, or recalibrate and
repopulate the cache via a shadow rebuild (utility-tested repopulation:
build a candidate cache from half of the recent labeled queries, compare
oracle hits with the current cache on the other half, install, rebuilt from
all of them, iff >= 10% better). Repopulating always implies recalibrating
(lambda is valid only for the cache it was calibrated on).
Cache policy: greedy (above) or LRU (continuous insert-on-miss; gets the same
alarm-reset calibration). External baselines with their own serve rules and
learned thresholds: Aker, QVCache (all parameters tuned).

Labels (backend answers): misses have them for free; hits are audited with
probability a (a backend call each). Calibration needs a random sample of
the workload (or known inclusion probabilities with inverse-probability
weights): the guarantee treats calibration queries as draws from the future
query distribution, and any selection related to score or retention (e.g.
misses only) biases the bound. Current implementation: a 10% coin flip for
every query (free for misses). Next: use ALL misses plus audited hits with
weight 1 / a (weighted confidence bound; more data for deciding whether to
lower lambda). Building the cache uses the same labels (audited hits up-
weighted to restore the hit / miss mix); its pool is split 20% candidate
entries (each a query's exact top-K ball, K in 10..500, plus current entries
free) / 80% scoring queries. Replace the fixed 60/40 calibration / build
split of audits with one pool and the rule: a cache is never calibrated on
the data used to build it (between rebuilds all labels calibrate).
After warm-up the model is never refit, so labels serve only building and
calibration (warm-up uses the four-way split candidates / scoring / fit /
calibration).

Why adaptive's hit rate barely drops after the shift although lambda uses
only post-shift calibration queries: ~1000 calibration queries already give
85-95% of the hit rate of 10k (calibration-size study), reached ~15k queries
after the alarm; and the baselines' post-shift hits are inflated by a stale,
too-loose lambda (their retention 0.89). Clear result: after the shift hit
rates are similar (adaptive 20.7%, periodic every 20k 22.6%, LRU K = 20
20.0%) but only adaptive keeps retention above T (0.918 vs 0.894 / 0.890).

Oracle references for shifting workloads: (1) per method, the oracle serve
decision for the cache it actually had (perfect knowledge of each query's
retention; per-batch retention histograms are now logged); (2) per phase, the
best cache built offline from that phase's own fully labeled queries (ESCI:
62.6% before / 41.4% after the shift). Planned: an oracle-calibrated lambda
reference (lambda from abundant fresh labels) to separate honest hits from
hits inflated by miscalibration.

Running experiments: parallelize to the machine (96 cores, 371 GB; jobs x
threads ~ cores; SPACEV 10M runs need ~5 GB each); timings are not measured,
so wall-clock parallelism does not matter for results.
Known bug (fixed in 5r): LRU K = 10 on spacev-real crashed after its cache
became empty.
SPACEV tuning grid (old code, seed 0, 1% budget): best setting per method
with mean served retention >= T (hit rate / oracle hit rate for the cache
the method had / retention; "below" = 5000-query windows below T):

| workload | ours (adaptive) | periodic | LRU | Aker | QVCache |
|---|---|---|---|---|---|
| aker-0.6 trace | 13.7 / 16.2 / 0.917, 0 below | 15.3 / 18.9 / 0.913, 1 | 11.7 (K 20) | 5.7 | 3.3 |
| aker-0.6 shift | 17.9 / 20.4 / 0.911, 1 | 19.0 / 22.6 / 0.913, 2 | 18.5 / 21.0 / 0.914, 0 | 8.9 | 6.9 (4 below) |
| spacev-real trace | 4.6 / 6.0 / 0.941, 1 | 4.8 / 6.3 / 0.933, 0 | 2.3 (K 50) | 4.0 (10 below) | 3.4 |

Aker / QVCache do better on the repeated-request traces than on ESCI (<= 1%)
but stay well below ours. Caveat: the old LRU results are handicapped by an
eviction bug (5r), and the old periodic / LRU used stale calibration windows.

## 5r. Round-2 implementation (detection x response) and launch

Code (`online_sim.py`, rewritten Sim; tests in test_online_sim.py):
- Methods: static, greedy (our cache), lru, aker, qvcache. Periodic,
  triggered and adaptive removed (adaptive = greedy --detect score
  --response rebuild with the post-alarm checks 5k / 15k / 30k).
- `--detect` any subset of score / retention / hitrate (each at
  delta_alarm / (m j (j + 1)), m = number enabled); `--response recal |
  rebuild` (lru always recal). An alarm resets calibration and the window.
- Calibration labels: every miss (weight 1) plus audited hits (weight
  1 / audit rate), serve-time (score, retention) pairs against the cache in
  force then (for LRU this is before the miss is inserted, so a cache is
  never calibrated on its own entries), among the last --cal-span = 20000
  queries since the last reset. Recalibrated every batch (cheap: no feature
  recomputation). Bound: `threshold_selection.select_weighted` (ratio
  estimate sum w y / sum w, variance sum w^2 (y - m)^2 / (sum w)^2, normal,
  fixed order; equals normal_fixed with unit weights).
- Score / hit-rate tests: last 2000 scores vs the earlier scores since the
  last reset (>= 1000 needed); hit-rate test = two-proportion z test.
- Rebuild: a random --cal-share = 20% of ALL window queries is held out (its
  labeled part calibrates the new cache; all its scores are the new
  reference), the labeled rest builds (shadow half / half, gain 10%).
  Replaces the fixed 60/40 calibration / build split of audits.
- `--oracle-cal`: every query labeled for calibration at no cost (reference
  separating honest hits from hits inflated by miscalibration).
- LRU eviction bug: chunk size assumed each entry frees 2 K vectors, so
  after the first batch it evicted all old entries and most new ones (K = 10
  emptied the cache and crashed; other K ran far under budget, so old
  SPACEV LRU numbers are handicapped). Now chunks = excess / average memory
  per entry, at most half the entries at once. Empty-cache guard in
  features (nothing served with < k cached vectors).
- Smoke test, ESCI shift, 90k queries: score alarm one batch after the
  shift (66000 vs 65623), rebuild at 71000 (shadow 4305 vs 1150 oracle
  hits), later check kept; ~4 min per ESCI run.

Round 2 (`online_round2_jobs.py` -> /tmp/round2_jobs.txt, 784 runs,
`xargs -P 56`, 1 thread each, SPACEV runs peak 5.7 GB; outputs
`results/online/<prep>/r2_<stream>_s<seed>/`, logs in `logs/`; flag file
`results/online/ROUND2_DONE`). Workloads: ESCI 1M shift + random (10%
budget), spacev-aker-0.6 trace + shift, spacev-aker-0.3 / 0.99 trace,
spacev-real trace (1% budget). Seeds 0, 1 for: static, greedy_none, greedy x
{score, retention, hitrate, all} x {recal, rebuild}, two oracle-cal
references, lru K in {10, 20, 50, 100} x {no tests, all tests}. Seed 0 for
the baseline grids (Aker 18, QVCache 54); the best baseline settings are
then re-run on fresh seeds so the reported numbers are not selected on the
same run.

## 5s. Multishift streams (several shifts of different sizes)

Why: every earlier workload has one shift (ESCI / Zipf shift streams: half
the clusters, then the other half; spacev-real: log, then official queries;
Zipf traces and random streams: none). That shift is maximal, so every test
catches it within a batch and the detector ablation cannot separate the
tests; one shift also gives one detection / recovery / rebuild event per run.

Stream `multishift` (`online_prepare.py --add-multishift`, written into
streams.npz atomically; `multishift_stream`):
- phase 0: warm-up (20k) + 5k stable queries, start mix = group A clusters
  in proportion to their sizes (an alarm here is a false alarm);
- phase 1, mild shift: 20% of queries from a few B clusters (new), and small
  A clusters holding 12-16% of the start mix left out (target 20%; cluster
  granularity), the kept A clusters in start proportions;
- phase 2, full shift: B only;
- phase 3, return: the start mix again (total-variation distance to phase 0
  = 0.0001), tests a second reset and recovering an old cache.
Exact per-cluster draw counts, each request used at most once (ESCI: each
query once, random order; traces: original order within a phase). The
clusters are split into A / B unevenly (A supplies phases 0, 1, 3) to
maximize the equal length L of phases 1-3. Lengths: ESCI 25.0k / 30.6k x 3
(116.7k of 130.7k queries used); Zipf 0.3 / 0.6 / 0.99 25.0k / 21.5-22.0k
x 3. On SPACEV a phase is shorter than the last scheduled rebuild check
(30k after an alarm), so only the 5k and 15k checks fall inside it.
Not used: repeating ESCI queries to lengthen the stream (an exact repeat of
a cached query is served perfectly, inflating hits); a second, milder level
(10%) would need more queries than we have.

Simulator: `--stream multishift`; results store `phase_starts`.
`online_summary.py --phases DIR...` prints per phase hit / oracle /
retention, detection delay (queries from the phase start to the first
alarm), alarms and rebuilds.
Runs (`online_round2_jobs.py --multishift`, 448 runs: our 20 configs x
seeds 0, 1 and the Aker / QVCache grids on seed 0, 4 workloads; outputs
`results/online/<prep>/r2_multishift_s<seed>/`) start automatically when
round 2 finishes (`/tmp/run_ms.sh` waits for ROUND2_DONE; flag
MULTISHIFT_DONE).

## 5t. Generalization of the features across cache states; data-change spec

### Which features generalize (reasoning, not yet measured)

Goal: one model whose score means the same thing in any cache state
(different caches, K mixes, budgets, shell settings, database versions), so
lambda is stable and only needs light recalibration. A feature generalizes if
P(retention | features) does not depend on hidden properties of the state.

- Relative offset and relative size (3 best balls): scale-free ratios
  within an exact ball. The core is geometric and state-independent: if
  offset + size <= 1 the query's neighborhood lies inside an exact ball, so
  retention = 1 for any cache and database. Beyond the boundary retention
  depends on how many uncached vectors lie in the uncovered part, which
  these features do not see (density, from the shell features). Depends on
  the state through ball overlap: with more overlapping balls the union
  covers more than the best 3 show (model too pessimistic: safe, fewer hits).
  Under data changes they keep their meaning only if balls stay exact
  (insert maintenance below).
- Intruder count, log1p(#sampled shell vectors inside the neighborhood):
  NOT invariant.
  (1) It scales with the sampling rate p: the same count means 1/p times
      as many intruders. p differs between caches (set to fill the budget)
      and changes at every online rebuild, while the model is fit once at
      warm-up. The offline transfer study kept p = 0.25 fixed, so this was
      never tested.
  (2) Shell coverage: shells exist only at ranks K+1..R around centers; an
      entry with K = R has none, and a neighborhood reaching outside every
      shell sees no intruders even if there are many. A count of 0 mixes
      "no intruders" with "not covered", and the mix depends on the K
      distribution and R.
  (3) Quantization (b bits) shifts which points count as inside.
  (4) Data changes: inserts outside all shells (data drift into new regions)
      are invisible.
- Likely fixes: an estimated intruder count, count / p (or log1p of it); a
  coverage feature, e.g. the neighborhood's relative size w.r.t. the best
  ball's shell radius (offset + size <= 1 in shell units = fully covered);
  train on many varied states (budgets, p, K mixes, before and after data
  changes) rather than one warm-up cache.
- Symptom already seen: lambda had to be recalibrated per cache state and
  per k, though the model's ranking transferred; consistent with scores
  whose meaning shifts with hidden state.
- Planned offline test (before the online data-change runs): fit on one
  state, test on states differing in p, R, bits, K mix and database version,
  raw count vs count / p + coverage; measure ranking (oracle share at the
  recalibrated lambda) and calibration drift (lambda change, retention at a
  frozen lambda).

### Data-change workloads (spec)

Two workloads, fixed query distribution (random stream) so only the data
changes; later combined with workload shifts:
- churn (no data drift): hold out a uniform random 10% of the database;
  initial database = the other 90%. After warm-up, the held-out vectors are
  inserted gradually over the stream, and as many uniformly random vectors of
  the initial database are deleted at the same rate (size stays constant,
  distribution stays the same).
- drift: k-means on the database (~50 clusters); hold out whole clusters
  totaling ~10% of the database, chosen so a meaningful share of queries
  (reported) has true neighbors in them; same insert / delete schedule
  (random deletes from the initial database), so the only difference from
  churn is where the inserts land. Queries near the new clusters get new
  true neighbors that start uncached, the dangerous case.
- Datasets: ESCI 1M first (131k queries, warm-up 20k, ~0.9 inserts and
  deletes per query), then SPACEV 10M traces.
- Schedule variants: gradual (default), and a burst (all inserts at once)
  to compare with known-change resets.

Ground truth without a database version per time: one ranking per query over
the union of all vectors (top-R' with R' = 1000; R' must exceed R = 500 plus
the vectors absent at any time; flag queries where fewer than R are
present). Truth at time t = the first k present ids of that ranking.
Prepared by `online_prepare.py --data-change churn|drift` (writes
db_events.npz: initial presence, insert / delete times and ids; topU.npy).

Cache maintenance (the ball invariant: every present vector within an
entry's radius is cached):
- an entry stores its ball radius and shell radius (distances to its K-th /
  R-th present neighbor at creation), i.e. prefix positions in its center's
  union ranking;
- insert x: distances from x to all entry centers (cost counted: one
  distance per entry); inside a ball -> cache x (memory + 1, fetched + 1);
  inside a shell -> joins the sample iff its per-vector draw < p;
- delete x: remove from the cache or sample (a cache must never return a
  deleted vector; applies to every method incl. baselines);
- over budget after inserts: lower p to fit (set_p_to_fill), so the p
  invariance above matters; entries are not dropped (the shadow rebuild
  handles it).
- ablation: insert maintenance off (balls silently become inexact).

Calibration under data changes (stored pairs were measured on an older
database):
- sliding (current code): pairs age out after --cal-span queries;
- maintained labels: keep each calibration query's top-(k + m) answer (m = 10
  buffer); an insert closer than the k-th updates it (one distance per
  stored query); a delete removes it; a label with fewer than k left
  expires; scores and retention recomputed against the current cache before
  recalibrating (every few batches to bound cost; cost counted);
- known-change reset (burst schedule only): reset calibration at the burst.
Detection: retention test (sees retention lost to inserts); score and
hit-rate tests (mostly blind to data changes: scores change only through
cache / sample changes); response recal / rebuild as before.

Grid (ESCI, churn and drift, 2 seeds): static (deletes removed only),
greedy x {sliding, maintained} x detect {none, retention, all} x {recal,
rebuild}, ablations (insert maintenance off; count / p feature), LRU (with
maintenance), Aker / QVCache (deletes removed; their own rules otherwise),
oracle-cal reference. Metrics as before plus: retention per window and for
queries whose true neighbors include inserted vectors, maintenance cost
(center distances, label updates), vectors fetched for in-ball inserts.

Order of work: (1) offline feature-invariance test above; (2) prepare
churn / drift for ESCI (union top-1000 for 131k queries over 1M); (3)
simulator support (presence mask, event processing, entry radii, label
maintenance, tests); (4) runs, after the workload-shift rounds finish.

### Step 1 (eval_invariance.py) and step 2 (online_prepare_data.py)

User's points: the intruder count is the only non-ratio feature, so it
should depend at least on k (a neighborhood holds k cached vectors, so
more intruders are expected at larger k) and on p. p is not fixed online:
set_p_to_fill runs at warm-up and after every rebuild while the model stays
the warm-up fit (LRU keeps p fixed); offline p = 0.25 always. Ratio form:
est = count / p estimates the uncached vectors inside the neighborhood, so
k / (k + est) is a direct, crude retention estimate.

Step 1, `eval_invariance.py` (ESCI 1M, online_sim.Cache, greedy cache
built like the online warm-up from 4k candidate / 8k scoring queries; fit
15k / cal 15k / test 30k queries shared by all states): 18 cache states
varying one property each (f, forced p, budget, R, bits, K mix, another
cache); variants base7 / est7 (log1p(count / p)) / ratio7 (k / (k + count /
p)) / cov8 (ratio7 + shell coverage of the best ball); per (state, k):
native vs base-model vs pooled-model hit rate at a recalibrated lambda, and
hit / retention at the base state's frozen lambda. `--state NAME` writes
results/invariance/NAME.npz, `--analyze` writes invariance.json.
First result (base state only, one seed; T = 0.9): ranking is the same for
all variants (recalibrated hit at k = 10 / 20 / 50: 31.6 / 26.8 / 19.4%,
oracle 37.4 / 30.8 / 22.1%), but the k = 10 lambda reused at other k:
base7 too strict (k = 50: 12.3% hits, retention 0.960); ratio7 transfers
(k = 50: 18.7%, retention 0.905; k = 20: 25.6%, 0.908). So the ratio form
makes the score's meaning k-invariant. Remaining 17 states running
(`/tmp/run_inv.sh`, 2 at a time; flag results/invariance/INV_DONE).

Step 2, `online_prepare_data.py --kind churn | drift` (ESCI 1M, 10% held
out, gradual insert + delete schedule from the end of warm-up to the end of
the stream, burst at 40000; union ranking topU1000.npy; drift: 50 database
clusters, held-out clusters with query share >= 0.5 x database share;
reports the share of queries whose true top-10 contains held-out vectors
and the share with < 500 present ranks at start / end). Running
(`/tmp/run_data_prep.sh`, flag esci1m/DATA_PREP_DONE).

Memory note: SPACEV LRU runs peak at up to 9 GB (not 5.7); with 56
parallel jobs available memory fell to ~12 GB, so round 2 was reduced to
48 slots (SIGUSR2 to GNU xargs) and the multishift round to 44.

## 5u. Round 2, multishift and invariance results

Round 2 (784 runs) and multishift (448) finished, no failures. Ours / LRU:
mean of seeds 0, 1; Aker / QVCache: best setting meeting T on seed 0 (not
yet re-run on fresh seeds). Hit rate % / mean served retention / 5000-query
windows below T. Full tables: `online_round2_summary.py [prep:stream ...]`
and `online_summary.py --phases`.

Single shift (ESCI 1M shift; SPACEV Zipf 0.6 shift):
| method | ESCI hit / ret / below | after shift hit / ret | SPACEV 0.6 shift |
|---|---|---|---|
| static | 29.0 / 0.880 / 14 of 23 | 10.7 / 0.788 | 9.8 / 0.897 / 10.5 of 16 |
| sliding recal, no test | 24.7 / 0.906 / 4 | 4.0 / 0.886 | 8.1 / 0.926 / 2 |
| score test, recal | 24.0 / 0.910 / 0.5 | 2.8 / 0.924 | 8.1 / 0.927 / 1 |
| score test, rebuild | 35.3 / 0.910 / 0.5 | 21.8 / 0.913 | 16.5 / 0.925 / 0 |
| same, oracle calibration | 38.6 / 0.906 / 3 | 26.8 / 0.908 | 17.8 / 0.910 / 0.5 |
| LRU K = 20, all tests | 29.3 / 0.912 / 0 | 17.0 / 0.917 | 17.3 / 0.918 / 0 |
| Aker best | 0.4 | | 8.9 |
| QVCache best | 1.0 (10 of 23 below) | | 6.9 (4 below) |
- Score, hit-rate and all-tests configurations are identical (the full
  shift fires every label-free test in the same batch). The retention test
  NEVER fires, even alone, although retention after the shift is 0.886:
  Hoeffding on <= 500 audited hits at level delta / (j (j + 1)) has a
  width of ~0.1, and audited hits are scarce after a shift (~4 per batch),
  so it can only catch large violations. Not useful for small deficits.
- Without a test, sliding recalibration alone violates after the shift
  (stale pairs stay in the 20k-query span).
- Rebuild is the main hit-rate lever after a shift (ESCI 2.8 -> 21.8%).
- Oracle calibration gains ~3 points: the calibration margin.
- Stationary streams (ESCI random, Zipf traces): no alarms in any of our
  configurations (no false alarms), all identical; static ~ greedy.
Stationary and trace workloads (hit %):
| workload | ours (greedy, any) | best LRU (K) | Aker | QVCache |
|---|---|---|---|---|
| ESCI random | 30.5 | 25.3 (10) | 0.2 | 0.7 |
| Zipf 0.3 trace | 4.1 | 4.3 (20) | none meet T | 0.7 |
| Zipf 0.6 trace | 13.0 | 11.2 (20) | 5.7 | 3.3 |
| Zipf 0.99 trace | 50.6 | 47.3 (20) | 35.8 (4 of 16 below) | 40.5 (5 below) |
| spacev-real trace | 4.4 | 3.1 (10) | 4.0 (10 of 21 below) | 3.4 |
After the eviction fix LRU is competitive on SPACEV (beats ours on the
Zipf 0.6 shift, 17.3-18.1 vs 16.5; ties on Zipf 0.3), and loses on ESCI.

Multishift (phases: start, mild, full, return; hit % / retention):
| ESCI | start | mild | full | return |
|---|---|---|---|---|
| static | 46.0 / 0.911 | 39.8 / 0.910 | 6.9 / 0.814 | 45.6 / 0.908 |
| no test | 45.1 / 0.915 | 40.3 / 0.907 | 3.4 / 0.860 | 41.3 / 0.921 |
| score, recal | 45.1 | 38.5 / 0.909 | 1.7 / 0.917 | 42.0 / 0.912 |
| score, rebuild | 45.1 | 37.8 / 0.914 | 25.3 / 0.915 | 29.1 / 0.911 |
| LRU K = 20, no test | 43.0 | 37.4 | 26.5 / 0.906 | 32.9 |
| LRU K = 20, all | 33.0 | 35.4 | 24.6 | 30.9 |
(SPACEV Zipf 0.6 / 0.99 return phase: score + rebuild 6.7 / 40.7 vs recal
only 14.0 / 54.0.)
- Detection delay: score test ~1000 queries (one or two batches) for the
  mild shift on all four workloads, ~450-1100 for full and return; the
  hit-rate test is slower on the mild shift (2000-3000) and missed it on
  Zipf 0.3; the retention test never fires.
- The mild shift does not break the guarantee even with no test (0.907),
  so its alarm only costs hits (reset).
- REBUILD HURTS ON THE RETURN: the start cache (built from 20k fully
  labeled warm-up queries) was replaced in the full phase; on the return
  the post-alarm rebuilds build from a short window of mostly misses, and
  the shadow check compares against the current (full-phase) cache, which
  is bad for the start mix, so a weak rebuild passes. Fixes to consider:
  keep earlier caches (entry lists) as candidates in the shadow check,
  later checks with a larger window, building quality.
- LRU with tests raises a false alarm right after warm-up (an evolving
  cache changes the score distribution), costing ~10 points in phase 0.

Invariance (step 1; 18 states, k = 10 / 50; frozen = base state's lambda):
- Ranking transfers everywhere: the base model with a recalibrated lambda
  is within ~1 point of a natively fit model on every state and variant.
- k: ratio7 makes the frozen lambda transfer across k (k = 50 at base:
  18.7% / 0.905 vs base7 12.3% / 0.960).
- p: NO variant is p-invariant. Frozen lambda at p = 0.1: retention 0.813
  (base7), 0.806 (est7), 0.824 (ratio7); p = 1.0: too strict. Reason:
  count / p has the right mean but its noise grows as p shrinks (at
  p = 0.1 most neighborhoods sample 0 intruders), so the evidence of a 0
  differs by p; a fixed transform cannot fix that. Needs p as an input
  (pooled over p) or an estimate that accounts for the uncertainty.
- R = 200 (thinner shells): ratio7 frozen 0.876 (base7 0.903); the coverage
  feature (cov8) changed nothing (the base state rarely has uncovered
  neighborhoods, so the model learns no use for it). K mixes: 0.876-0.881.
- 2 bits: frozen retention ~0.72 for every variant (quantization changes
  the inside test); 8 bits fine.
- So per-state recalibration stays necessary (cheap online; done after every
  rebuild, where p changes); an invariant score is a nice-to-have (frozen
  or pooled lambda).

Data-change prep (step 2): churn: 65% of queries have >= 1 held-out vector
in their final true top-10 (expected 1 - 0.9^10), 0.14% have >= 5; no query
below 500 present ranks. Drift (6 database clusters, 10.3% of the database,
query share 0.8-1.8% each vs database share 1.2-2.4%): 21% of queries have
>= 1 held-out vector in their top-10, 5.7% have >= 5. PROBLEM: at the start
3.1% of queries have < 500 present ranks in the union top-1000 (minimum 0:
their whole top-1000 is in held-out clusters), so their truth before the
inserts is unknown. Fix in step 3: for those queries also rank the initial
database (top-1000 over initially present vectors) and merge.

## 5v. Detector study (eval_detectors.py)

Method: the state at the moment a shift arrives (warm-up cache, model,
lambda frozen; no response yet) is fixed, so every query's score and
retention against it is computed once (--prepare), and detectors are
replayed on sampled streams (--run): P pre-shift queries from group A (the
warm-up mix), then H = 40k post-shift queries from (1 - eps) A + eps B,
eps in {0, .05, .1, .2, .3, .5, .7, 1}; 200 replications per eps; P = 20k
and 60k (60k: later in a run, so the level schedule is stricter). Since
cache and lambda are fixed, the post-shift served retention is exact:
| eps | 0 | .1 | .3 | .5 | .7 | 1 |
|---|---|---|---|---|---|---|
| ESCI served retention / hit rate | .909/.548 | .906/.503 | .900/.412 | .891/.321 | .873/.231 | .787/.095 |
| SPACEV Zipf 0.6 | .906/.197 | .905/.181 | .902/.150 | .898/.119 | .891/.088 | .859/.042 |
(harmful iff < 0.9: ESCI eps >= 0.5 (0.3 borderline), SPACEV eps >= 0.5).
Variants: score / hit-rate tests: recent window W in {500, 2000, 5000},
reference "cal" (4k warm-up calibration scores) or "grow" (all scores since
warm-up), check every B in {1000, 250}, level schedule jj (delta / (j (j +
1))), fixed (delta / J, J known), none. Retention test: data = last 100 /
500 audits, audits of the last 2k / 5k / 20k queries, or a scan over 1k /
2k / 5k / 10k / 20k-query windows (level / 5); bounds hoeffding, kl
(Chernoff form, same assumptions), eb (empirical Bernstein, same
assumptions), normal (CLT, the assumption calibration already makes);
audits fixed 0.1 / 0.3 or adaptive (next batch's rate = clip(A / hits in
the last batch, 0.1, 1), A = 10 / 30; audited retentions weighted 1 / rate).
Results: results/detectors/<prep>_P<P>_detectors.json;
`eval_detectors_summary.py FILE [--filter ...]`.

Findings (all from these runs):
1. A level correction is necessary: without one (per-check level delta)
   false alarms are 52-56% over 60 checks and 91-95% over 100 checks. With
   jj or fixed they stay <= 5% (one exception, score test with the "cal"
   reference and W = 500 on SPACEV: 6-7.5%; the SPACEV trace is
   time-ordered, so later A requests differ slightly from the warm-up's).
   jj and fixed perform alike.
2. Label-free tests: on harmful shifts the score and hit-rate tests both
   fire within one batch (1000 queries; 250-500 with B = 250, at the same
   false-alarm rate). On small shifts the score test is more sensitive:
   SPACEV eps = 0.3: score W = 500 94-98% within 1000 vs hit-rate W = 500
   23-28% (delay 4-12k); SPACEV eps = 0.1: score grow W = 5000 95% vs
   hit-rate 21%; on ESCI they are close for W >= 2000. The hit-rate test
   was never faster than the score test at any eps. The "grow" reference
   beats "cal" (larger reference: more sensitive at small eps, fewer false
   alarms). Neither test tracks harm: ESCI eps = 0.1 (retention 0.906,
   harmless) is detected 98% of the time.
3. Retention test: (a) no false alarms (<= 1%) in any variant; (b) bound:
   normal >> kl > hoeffding ~ eb at these sample sizes and strict levels
   (eb's 7 log(2 / level) / (3 (n - 1)) term dominates); (c) data: the old
   count window (last 500 audits) is slow (stale pre-shift audits); the
   scan over several time windows is best overall; (d) adaptive auditing
   is the biggest lever at almost no cost: SPACEV eps = 1, scan + normal:
   fixed 0.1 detects 2%, fixed 0.3 45%, adaptive A = 30 99% (median 13k
   queries) with 9 / 27 / 17 audits per 1000 queries; ESCI eps = 1: median
   delay 9000 -> 3000 at the same audit cost.
4. The retention test is limited by sample size, not tuning: queries to
   detect ~ (z sd / deficit)^2 / (audits per query) (sd of served
   retention ~0.16-0.21, z at the test level). Predicted vs observed median
   delay: ESCI eps = 1 (deficit 0.113) 2.2k vs 3k; ESCI eps = 0.7 (0.027)
   27k vs 22k (48% detected within 40k); SPACEV eps = 1 (0.041) 12k vs
   13k; deficits <= 0.01 need ~ 6000 audits (~200k queries at 30 audits per
   1000): undetectable here. Best variant: scan + normal + adaptive A = 30.
5. Later in a run (P = 60k vs 20k) the retention test is slower (ESCI eps
   = 1: 3000 -> 5000; SPACEV: 13k -> 20k): the jj level is stricter; the
   label-free tests are unaffected (strong signal).
Answer to "score > hit-rate > retention fundamental?": label-free tests are
faster than the retention test for workload shifts (they use every query;
the retention test needs (sd / deficit)^2 audits); the score test is at
least as fast as the hit-rate test everywhere and much more sensitive to
small shifts on SPACEV. But faster is not better: the label-free tests fire
on harmless shifts (needless resets), the retention test only on harmful
ones, and only it can see changes that leave scores unchanged (data drift).
Caveats: first shift only, frozen lambda, abrupt mixtures of A and B.

## 5w. Detection + response study (eval_response.py)

Question (user): if the response to an alarm is cheap and verifies itself,
can the tests run loosely (accept false alarms) for faster detection?
And should a shadow rebuild be part of the response?

Corrected reasoning (shadow recalibration): after an alarm the only
thresholds that can be certified are those certified by labels collected
after the alarm (lambda_fresh and anything stricter); using older labels to
loosen lambda assumes nothing changed, which is what the alarm disputes. So
the post-alarm lambda is always lambda_fresh; a "shadow" period can only (1)
keep serving at the pre-alarm lambda for a few batches while fresh labels
accumulate (grace; unverified serving), or (2) raise audits to refill
faster (boost). The cost of a needless alarm is the refill time (misses
are labeled free, so it is short). The shadow rebuild answers a different
question (is a better cache available?), independent of harm.

Method: replay as in 5v (graded mixtures eps, 100 replications, P = 20k,
H = 40k) with the serving loop's calibration (lambda recalibrated every
batch from misses + weighted audited hits of the last 20k queries since the
last reset; adaptive audits, 30 per batch) and detection (score test, KS
last 2000 vs earlier since reset; retention test, scan + normal bound).
Levels: strict = delta / (2 j (j + 1)) per test; loose = fixed per-check
level 1% / 5% / 20%. Responses: reset; grace 2 batches (an alarm during
grace ends it: evidence of harm); boost (100 audits per batch for 5
batches); shadow rebuild at +5k / +15k after an alarm with candidates
precomputed per (mix, window) from realistic windows (misses + weighted
audited hits, 20% held out to calibrate it), build queries disjoint from
replayed ones. Results results/response/<prep>_P20000_response.json;
`eval_response.py --summary FILE`.

Shadow decisions (candidate vs current oracle hits; install iff >= 1.1x):
installed only at eps = 1 (5-12x) and eps = 0.7 with the 15k window (ESCI
1.25x, SPACEV 1.63x); declined after a needless alarm (eps = 0: candidate
worse, 0.78-0.94x) and on mild shifts. So the shadow check already prevents
bad rebuilds after false alarms.

Results (post-shift hit % / served retention; ESCI | SPACEV Zipf 0.6):
| config | eps 0 (pre-shift hit) | eps 0.1 | eps 0.7 | eps 1.0 |
|---|---|---|---|---|
| no detection | 54.8 / .909 | 49.7 / .910 | 19.7 / .905 (29% windows < T) | 4.8 / .857 (50% < T) |
| strict, reset | 54.7 | 48.2 / .911 | 18.4 / .914 | 3.1 / .910 |
| strict, reset, rebuild | 54.7 | 48.2 / .911 | 20.6 / .913 | 17.6 / .912 |
| strict, grace 2 | 54.7 | 49.2 / .912 | 18.9 / .912 | 3.6 / .883 |
| loose 1%, reset | 54.1 | 47.6 | 18.2 | 3.0 / .910 |
| loose 5%, reset | 52.1 | 46.1 | 17.6 | 2.8 / .912 |
| loose 20%, reset | 46.6 | 41.8 | 15.8 | 2.3 / .914 |
| loose 5%, grace 2 | 54.2 | 48.7 | 18.6 | 3.4 / .888 |
| loose 20%, grace 2 | 53.2 | 47.6 | 18.0 | 3.2 / .893 |
SPACEV: no detection eps = 1: .906 mean but 32% of windows < T; strict +
reset + rebuild: eps = 1 hit 2.6 -> 7.0%, eps = 0.7 8.1 -> 10.5%,
retention >= .921 everywhere; loose 5% / 20% reset: pre-shift hit 21.1 ->
19.6 / 17.8; grace keeps retention >= T on SPACEV (deficits are smaller).
Findings:
1. Detection is needed (no detection: ESCI eps = 1 retention 0.857).
2. Strict detection + reset + shadow rebuild is safe at every eps and
   recovers hits after large shifts; the shadow check rebuilt nothing after
   needless alarms.
3. Tolerating false alarms brings nothing for workload shifts: the strict
   score test already catches harmful ones within a batch, so loose levels
   only add needless resets (each costs ~1 point of hit rate over a 60k-
   query run: ESCI 0.7 / 2.5 / 8.2 alarms -> 0.7 / 2.7 / 8.2 points).
4. Grace makes needless alarms almost free (loose 20%: 46.6 -> 53.2) but
   breaks the guarantee on severe shifts (ESCI eps = 1: 0.883-0.893), as it
   must: it serves at an unverified lambda. Not acceptable for a method
   whose point is the guarantee.
5. Boost: negligible effect (rows omitted above).
Where a loose retention test could still pay: changes the score test cannot
see (data drift: retention drops while scores stay), where it is the only
detector; to test with the data-change workloads.

## 5x. Data-change round 1 (ESCI 1M; setup)

Takeaway carried over from workload shifts: score test as the alarm, reset
(fresh-only calibration) on an alarm, shadow rebuild to decide whether a new
cache is worth it. For data changes the query stream is fixed, so scores
move only a little (inserts inside balls / shells); the audit-based
retention test is expected to be the main detector.

Setup (decided with the user): random query order; churn (random 10% of
the database held out, inserted gradually from the end of warm-up, as many
random deletes) and drift (6 whole clusters, 10.3%, same schedule); each
gradual and burst (all at query 40000); our method only; seeds 0, 1.
Ground-truth fix: 5209 drift queries (3.1% below 500 present ranks at the
start, some 0) got their union ranking merged with a ranking of the initial
database (width 2000); each then has >= 843 vectors present throughout.

Simulator (online_sim.py):
- Cache entries are balls with a radius fixed at creation (distance to the
  K-th present neighbor); members = present vectors inside, so inserts
  inside a ball join it and deletes leave (balls stay exact); shells
  likewise. --no-maintain freezes member / shell sets at creation (deletes
  still leave). Fixed-database runs reproduce round 2 exactly (served
  counts identical in every batch; lambda equal to 1e-6).
- Ground truth: first k present ids of each query's ranking
  (topk_present); greedy and shadow checks use the current database.
- Events applied before each batch; memory over budget -> lower p. Costs:
  center checks (insert x entries), label updates (insert x stored
  calibration queries, maintained mode).
- Calibration labels: serve (as measured at serve time; age out of the 20k
  window) or maintained (kept current; stored queries' scores and
  retentions recomputed every 5 batches and right after a rebuild).
- Retention test now as the detector study recommended: audited hits of
  the last 1k / 2k / 5k / 10k / 20k queries, normal bound; adaptive audits
  (aim at 30 per batch, default); --alarm-level for a fixed per-check
  false-alarm chance; --reset-on-change for a known large change.
- Logged per batch: queries whose current exact top-10 contains inserted
  vectors (affected), how many are served and their retention.
Configurations (online_data_jobs.py; 100 runs; results
esci1m/data_<kind>_<schedule>_s<seed>/): static; sliding labels x {no
test, score, retention, both, both loose 5%, retention loose 5%};
maintained labels x {no test, retention, both}; no upkeep + both; oracle
calibration; burst: known-change reset. Response: reset + shadow rebuild.
Memory: each run ~4.3 GB (ranking arrays); 84 parallel runs exhausted
memory (7 GB free), so 20 churn runs were stopped (re-run automatically)
Results (100 runs; `online_data_summary.py`; mean of seeds 0, 1; hit % /
mean served retention; "affected" = queries whose current exact top-10
contains inserted vectors):
| config | churn gradual | churn burst | drift gradual | drift burst (after burst) | drift: affected queries' retention |
|---|---|---|---|---|---|
| static (upkeep, no recalibration) | 33.1 / .905 | 33.1 / .906 | 35.5 / .887 | 38.7 / .861 | .846 / .818 |
| sliding labels, no test | 31.9 / .911 | 31.8 / .912 | 31.3 / .909 | 30.0 / .906 (3 of 18 windows < T, worst .868) | .878 / .872 |
| sliding + score test | same | same | same (never fires) | 28.8 / .911 (1 window, worst .894) | .878 / .878 |
| known-change reset (burst) | | 31.6 / .913 | | 29.2 / .911 (1.5 windows) | / .878 |
| maintained labels, any test | 32.0 / .911 | 31.8 / .912 | 31.0 / .911 | 29.4 / .910 | .880 / .877 |
| no upkeep + both tests | 21.0 / .903 (affected .837) | 27.5 / .908 | 25.3 / .908 | 22.5 / .904 | .584 / .570 |
| oracle calibration | 33.0 / .906 | 33.1 / .906 | 32.8 / .902 | 31.3 / .900 | .869 / .865 |
Findings:
1. Random churn does not threaten the promise, even without recalibration
   (static 0.905-0.906), as long as the cache is kept up to date.
2. Cache upkeep is essential: without it hit rate falls by a third (churn
   gradual 31.9 -> 21.0%) and queries whose answers include new vectors get
   retention 0.57-0.58 (drift) / 0.84 (churn).
3. Drift into new regions breaks the promise if the cutoff is never
   recalibrated (static 0.887 gradual, 0.861 after the burst).
4. Recalibrating every batch on recent labels (misses are free) already
   tracks gradual drift (0.909): no detector fired and none was needed.
5. After a burst, the sliding window lags (3 windows below T, worst 0.868);
   the score test fires (inserts change the cache's own contents: balls
   and shells), its reset matches knowing the change time (1-1.5 windows,
   worst 0.894).
6. The retention test (strict) fired only once in all data runs (naive
   cache, churn burst; led to a rebuild); loose 5% fired in half the drift
   burst runs, no earlier than the score test. Consistent with the
   sample-size law (deficits here 0.01-0.04).
7. Maintained labels: no measurable gain over sliding labels at this rate
   of change (<= 0.004 retention), ~5x compute.
8. The promise is on the average over hits: queries whose answers include
   new vectors (15-19% of drift queries) get 0.87-0.88 under every
   calibrated configuration while the average is 0.909-0.911.
9. "Windows below T" is noisy: oracle calibration sits closer to T (mean
   0.900-0.902), so more 5000-query windows dip below by chance.
Specific to this setup: 10% changed, ESCI, the 20k-query window, insert
rate ~0.9 per query, drift clusters chosen so queries hit them (some balls
there, which is why the score test sees the burst; a drift entirely outside
all balls and shells would leave scores unchanged).

## 5y. Takeaways so far (plain summary)

The promise: on average over cache hits, a served answer keeps at least
90% of the true top-10. The average is enough (user decision); no promise
per group of queries.

Workload shifts (the queries change):
- Watch the score distribution every batch. It is cheap, needs no labels,
  and catches harmful shifts within about one batch.
- On an alarm, recalibrate only on queries that arrive after it (older
  labels describe the old workload).
- A few thousand queries later, try a shadow rebuild: build a candidate
  cache from recent queries, switch only if it serves clearly more of them.
- The audit-based retention test is too slow to matter here: it needs
  about (noise / shortfall)^2 audited hits, i.e. thousands for small
  shortfalls.

Data changes (inserts and deletes):
- Keep the cache up to date: inserts inside a cached neighborhood join it,
  deletes leave. Essential; without it hits drop by a third and queries
  with new neighbors keep only 57-84% of them.
- Recalibrate every batch on recent labels (misses are labeled for free).
  This alone keeps the promise under gradual change; no alarm was needed.
- A sudden change can make the recent labels lag. The score test caught our
  burst (the inserts changed what the cache holds) and its reset fixed it.
  It cannot see changes far from anything cached; those mostly affect
  misses, which the backend answers exactly.
- Untested: a shadow rebuild every so much changed data (e.g. 5% of the
  database), to win back hits when no alarm fires. It can only help
  (switches only if better).
- The retention test fired once in 100 runs; no evidence it is needed.

Fundamental: need for upkeep; old labels can't certify a cutoff after a
change; label-free tests are fast but blind to harm; label-based checks
need many audits; only a new cache wins back hits.
Specific: ESCI only, 10% of the database changed, one change per query,
abrupt cluster-mixture workload shifts, untuned window sizes.

Next steps:
1. More datasets (SPACEV traces; data-change workloads need their own
   neighbor rankings over all vectors).
2. Baselines Aker and QVCache on the data-change workloads, using their
   own update rules (below).
3. Shadow rebuild triggered by the amount of changed data.

Baseline update rules (from the papers; correction: an earlier note said
neither paper specifies one):
- Aker (sections 4.5-4.6, refresh test 5.4): "del-consistency": never
  return a deleted vector; new vectors may be missing for a while.
  Each entry stores its top-k plus a reserve of D extra neighbors.
  Delete: remove the vector from every entry holding it, promote the
  closest reserve neighbor; an entry left with fewer than k is dropped
  (its query becomes a miss). Insert: appended to an insert log. Fast
  path: check it against the single entry whose query is closest; if
  closer than that entry's farthest stored neighbor, swap it in. Slow
  path: entries visited round-robin, each visit checks the next 16 log
  records against that entry; it runs when a risk score R >= 0.3, where
  R = (average over entries of d_k / d_{k+D}) x (average unchecked log
  records per entry) / (log size). Main experiments use D = 0; the
  refresh test uses D = 10 / 30, threshold 0.3, batch 16 (1M SPACEV:
  warm up, insert 5%, delete 5-25%, replay the queries).
- QVCache (arXiv 2602.02057): does not handle inserts or deletes; they go
  to the backend only ("future work"). Unpatched it returns deleted
  vectors and never sees new ones. Plan: run it as published, plus a
  minimal fix that removes deleted vectors from its mini-indexes (any
  deployment would need this; its index type supports deletion).

## 5z. Data-change round 2 (baselines, SPACEV; setup)

Baseline update rules implemented (online_baselines.py; tests in
test_online_baselines.py): Aker as in its paper (reserve D, immediate
deletes with promotion / drop below k, insert log with fast path and
round-robin slow path when risk >= 0.3, batch 16; reserves count against
the memory budget); QVCache as published (deletes kept and countable as
deleted_returned) or with deletes removed. Both use the current exact
top-k as the backend answer. Fixed-database runs reproduce round 2 exactly
for both (served counts and retention in every batch). QVCache's retention
now counts a cached true neighbor as returned if it is at least as close
as the answer's k-th vector (ESCI has equal-distance duplicate vectors;
deleted vectors still cached that are closer push true neighbors out).
Smoke runs (ESCI drift gradual, one setting each): Aker 0.1% hits; QVCache
~1% hits at retention 0.84-0.85 (affected queries 0.68-0.70).

Data prep generalized to traces (online_prepare_data.py reads meta.json):
spacev-aker-0.6 (94k distinct queries) and spacev-real (115k), 10M
vectors, 10% changed (~12 changes per query), churn and drift.

Round 2 (online_data_jobs.py --round 2; 1176 runs): ESCI: Aker grid (tau
init x alpha_T x cap x reserve 0 / 10; 36), QVCache grid (D x mini x
deletes keep / remove; PCA 16 x 8 fixed, best in every earlier grid; 36),
seed 0; LRU K 10 / 20 / 50 / 100 x tests none / both, seeds 0, 1. SPACEV
Zipf 0.6 and real log: our configurations (static, sliding none / score /
both / retention loose, no upkeep, oracle, known reset; maintained labels
skipped: no gain on ESCI at 5x cost) and LRU, seeds 0, 1; the same baseline
grids, seed 0. Plus (online_round2_jobs.py --best-baselines 1 2, 44 runs)
the best seed-0 Aker / QVCache setting per workload-shift workload re-run on
seeds 1 and 2 (best = highest hit rate with retention >= T; if none meets
T, the highest retention, reported as not meeting T). Orchestration
/tmp/run_round_dc2.sh: ESCI + re-runs (40 parallel), then SPACEV (30
parallel, ~8-9 GB each) after its prep; flags DC2_A_DONE, DC2_B_DONE in
results/online.

First results, ESCI half of round 2 (352 runs + 44 workload-shift baseline
re-runs on fresh seeds finished; no failures). Hit % / mean served
retention, mean of seeds 0, 1 (baselines: best setting on seed 0, highest
hit rate with retention >= 0.9, else highest retention):
| | drift gradual | churn gradual |
|---|---|---|
| ours (sliding labels, no test) | 31.3 / .909 | 31.9 / .911 |
| LRU (best K = 10) | 27.2 / .909 | 27.0 / .912 |
| Aker (best: tau init .25, alpha_T .9, cap 1, reserve 0) | 0.2 / .910 | 0.2 / .912 |
| QVCache, deletes kept (as published) | 0.6 / .884 (none meets T) | 0.6 / .890 (none meets T) |
| QVCache, deletes removed | 0.7 / .905 | 0.6 / .913 |
- Aker and QVCache reach under 1% hits on ESCI (as in the workload-shift
  runs: ESCI queries never repeat, and their reuse rules need near-
  identical queries), so ESCI cannot show how their update rules fare;
  the SPACEV traces with repeated requests are the real test.
- QVCache as published drops below T once vectors are deleted (0.884 under
  drift; 0.67 for queries whose answers include new vectors) because
  deleted vectors stay cached; removing them recovers 0.905. Small sample
  (about 700 hits), differences of ~0.02 are about 3 standard errors.
- Ours leads LRU by about 4-5 points at the same retention on ESCI.
`online_data_summary.py` now also prints the best LRU / Aker / QVCache
settings. SPACEV data prep (two exact top-1000 rankings over 10M vectors)
was still running after 74 minutes; the 824 SPACEV runs start after it.

## 5aa. Approximate backend, step 1 (controlled recall; built, first check)

Model (online_sim.py `approximate_rows`): for each query the index misses
each database vector independently with probability 1 - recall (a fixed
draw per query and rank position, seeded) and returns the best vectors it
still sees, in order, from the 1000-deep ranking. Recall at any k then
equals 1 - miss probability (a surviving true top-k vector is always inside
the returned top-k); measured recall is logged and matches (0.990 / 0.950 /
0.900 at k = 10 and at R = 500). Everything the system can observe uses the
backend's answers: cache balls and radii (from the answer list), the shell,
audit and miss labels, the greedy objective, calibration. `D["truth"]` (the
exact ranking) is used only to evaluate: every batch logs
`served_retention_sum` (against the backend's answer, the label we can
calibrate) and `served_retention_exact_sum` (against exact truth).
Flags: `--backend-recall`, `--backend-seed`; fixed-database runs load
`topU1000.npy`. Not yet supported with an approximate backend: Aker and
QVCache (the simulator refuses to run them).
Upkeep detail: membership comes from the backend's answer row, so an
inserted vector joins a ball only if the backend (index) would have
returned it for that center, i.e. upkeep is as imperfect as the index.

First check (ESCI 1M, stationary stream, 60k queries, 10% budget, score
test + rebuild, one seed; ~12,000 hits each):
| backend recall | hit % | retention vs backend answer | retention vs exact truth |
|---|---|---|---|
| 1.00 | 30.1 | .912 | .912 |
| 0.99 | 29.9 | .914 | .915 |
| 0.95 | 29.8 | .911 | .913 |
| 0.90 | 29.4 | .907 | .911 |
- The promise holds against the backend's answer at every recall (>= .907
  vs .90), and hits fall only 0.7 points at recall 0.9.
- Retention against exact truth is slightly HIGHER than against the
  backend's answer, the opposite of what I guessed beforehand. Likely
  reason (unverified): with independent misses, a vector one query's search
  missed is usually returned for nearby queries and so gets cached, while
  the substitutes from deeper ranks that the backend returns instead are
  farther and less often cached. Real index errors are correlated across
  nearby queries (the same hard vectors get missed), so this benign result
  need not hold for a real index: the reason step 2 (a real index tuned to a
  target recall) is needed.
- Small run: differences of 0.004 are about 2.5 standard errors.

## 5ab. Approximate backend, step 2 (real indexes; HNSW and IVF done)

Code: `eval_real_backend.py` (run with `.venv-ann`: faiss-cpu 1.8.0,
diskannpy 0.7.0, numpy 1.25, packaging 24.1; separate from `.venv` because
diskannpy pins numpy 1.25). Index over the ESCI 1M database; search effort
tuned on 5000 queries to the smallest setting whose mean recall@10 reaches
0.99 / 0.95 / 0.90; then every query answered twice, as the system asks:
top-10 answers (misses, audits; tuned effort) and top-500 deep lists
(cache entries and shells). Asking HNSW / DiskANN for 500 results forces a
search list >= 500, so their deep lists are much better than their
answers; IVF keeps its tuned number of probed lists, so its deep lists
degrade too. Simulator: `--backend-file results/backends/esci1m/<index>.npz
--backend-target 0.95`: D["rank"] = deep lists (entries, shells, greedy
candidates), D["answer"] = top-10 answers (labels, retention, scoring),
D["truth"] = exact (evaluation only). Real-index backends: fixed database
only.

Index recall (all 131k queries) and how concentrated misses are (share of
all misses on the 10% of vectors missed most often; in brackets the same
for independent misses at that rate):
| index, target | answers recall@10 | deep recall@10 / @500 | misses on top 10% vectors |
|---|---|---|---|
| HNSW (M 32), 0.99 (ef 160) | .992 | .998 / .977 | (too few misses) |
| HNSW, 0.95 (ef 40) | .957 | .998 / .977 | 57% (32%) |
| HNSW, 0.90 (ef 20) | .908 | .998 / .977 | 35% (18%) |
| IVF-Flat (4096 lists), 0.99 (nprobe 384) | .994 | .994 / .980 | (too few misses) |
| IVF, 0.95 (nprobe 64) | .960 | .960 / .885 | 87% (34%) |
| IVF, 0.90 (nprobe 24) | .913 | .913 / .777 | 60% (19%) |
Real errors are concentrated on particular vectors (IVF: a vector in a
cluster the search does not probe is missed by every nearby query), unlike
the independent misses of step 1.

Our method on these backends (ESCI 1M, stationary stream, 60k queries, 10%
budget, score test + rebuild, one seed, ~12,000 hits each; exact backend:
30.1% / .912):
| backend | hit % | retention vs backend answer | vs exact truth |
|---|---|---|---|
| HNSW .99 / .95 / .90 | 30.3 / 30.5 / 30.8 | .909 / .911 / .909 | .908 / .909 / .907 |
| IVF .99 / .95 / .90 | 30.2 / 30.8 / 30.7 | .912 / .909 / .908 | .912 / .906 / .901 |
- The promise holds against the backend's answer for every index and
  recall (>= .908), and the hit rate is unchanged.
- Against exact truth retention is slightly lower than against the
  backend, most for IVF at 0.90 (.901, gap .007): with concentrated
  errors the cache reproduces the backend's mistakes (its deep lists miss
  the same vectors). Opposite sign to step 1 (independent misses), as
  suspected there.
- Retention vs exact (.90-.91) is about the backend's own recall or above:
  the cache answers with the nearest cached vectors, which can include true
  neighbors the backend missed for that query (cached from other queries'
  deep lists).
- One seed; differences of .002-.003 are within about 1-2 standard errors.
With vs without the cache (recall against the exact top-10; re-run of the
0.90 cases logging the backend's own recall per query; 30.8% served):
| backend 0.90 | served queries: backend alone | served: cache | all queries: no cache | all: with cache |
|---|---|---|---|---|
| HNSW | .945 | .907 | .908 | .896 |
| IVF | .958 | .901 | .913 | .896 |
The cache serves the backend's easy queries, costs 4-6 points of recall on
them (within the target: hits keep >= 90% of the backend's answer; observed
.908-.909), and 1.2-1.7 points overall, while ~31% of queries skip the
backend. Correction to a misreading: .901 vs .908 is the same cached answers
against exact truth vs against the backend's answer, not cache vs no cache.
DiskANN (32 bytes per vector in memory, node cache off) still tuning.

## 5ac. Timing study, first round (ESCI 1M, DiskANN on EBS)

Code: `eval_timing.py` (run with `.venv-ann`), cache states from
`online_sim.py --save-state` after warm-up (stationary stream, exact
backend for building). One query at a time, single-threaded (BLAS 1
thread; DiskANN searches issued one at a time); 3000 timed queries per arm
after 200 untimed. Hit decision re-implemented per query exactly as the
simulator computes it (scores agree to < 1e-5, decisions 100%). DiskANN:
32-byte in-memory compressed vectors, search list 128 (recall@10 .96),
beam width 2. Arms: DiskANN alone (node cache off); our cache + DiskANN on
misses; DiskANN alone with its node cache given our cache's memory (nodes
chosen by DiskANN's standard warm-up on its sample queries).
Pitfalls hit: diskannpy's loader runs its node-cache warm-up (100k sample
searches) unless cache_mechanism = 0; and it hangs when opened with 1
thread (2 contexts used; searches still one at a time).

| cache budget | cache memory | hit rate | hit decision (all queries) | DiskANN alone | cache + DiskANN | DiskANN with our memory as node cache |
|---|---|---|---|---|---|---|
| 1% | 16.4 MB | 12.3% | 1.17 ms | 49.6 ms | 44.7 ms | 47.6 ms |
| 10% | 157.6 MB | 30.3% | 24.6 ms | 50.1 ms | 59.6 ms | 41.3 ms |
(mean per query; p99: 1% 56.2 vs 55.1 ms alone; 10% 79.9 vs 54.8 ms.)
- At 1% the cache cuts mean latency 10% (49.6 -> 44.7 ms) and beats
  giving the same memory to DiskANN (47.6 ms); p99 is ~1 ms worse (every
  miss pays the 1.2 ms decision first).
- At 10% the cache makes queries SLOWER (59.6 vs 50.1 ms) and giving the
  memory to DiskANN is better (41.3 ms). The hit decision costs 24.6 ms:
  10.5 ms for the 10 nearest cached vectors and 13.8 ms for the shell
  intruder count. Both are brute-force scans over everything cached
  (85k cached vectors, ~120k shell vectors), so their cost grows with the
  cache: one query reads ~130 MB of cached vectors plus the shell (held as
  float32 in the harness), about memory-bandwidth speed. The hit decision
  must not scan the whole cache.
- EBS inflates DiskANN (50 ms per query; local NVMe would be several times
  faster), which makes any cache look better. On fast storage even the 1%
  decision (1.2 ms) would be a large share of a miss.
- Hit rates are ESCI's (no repeated queries); traces with repeats hit more.
Fix candidates: an index over the cached vectors (approximate nearest
cached vectors, e.g. a small HNSW, as QVCache uses graph mini-indexes),
counting intruders only among shell vectors of the best-fitting balls,
computing on compressed vectors, batching queries for throughput.

Fixes, round 2 (`eval_timing.py --variants`; same queries, 2000 timed;
decisions compared with the simulator's for the same cache state and
intruder scope). Nearest cached vectors: scan, or a FAISS HNSW graph over
the cached vectors (M 32, efConstruction 200; search list 16 / 32 / 64).
Intruder count: scan at full precision; scan of the 4-bit codes themselves
(FAISS 4-bit scalar quantizer, one list, radius search); or only the
sampled shell vectors of the 3 best-fitting balls (`--shell-scope balls`,
which the model is then fit and calibrated with).
Hit decision, mean ms per query (stages: nearest cached vectors / intruder
count):
| variant | 1% | 10% |
|---|---|---|
| scan + scan (current) | 1.23 (0.53 / 0.64) | 25.5 (10.8 / 14.4) |
| scan + 4-bit scan | 1.93 (0.53 / 1.33) | 22.1 (9.2 / 12.7) |
| HNSW 32 + scan | 0.82 (0.11 / 0.65) | 12.7 (0.21 / 12.2) |
| scan + best balls | 0.61 (0.50 / 0.05) | 8.2 (7.9 / 0.11) |
| HNSW 32 + best balls | 0.21 (0.10 / 0.05) | 0.51 (0.20 / 0.10) |
- HNSW finds 89% / 95% / 98% of the exact 10 nearest cached vectors at
  search list 16 / 32 / 64; hit decisions agree with the exact scan in
  99.4-100% of queries. Its graph adds 2.2 MB (1%) / 22 MB (10%), about
  14% of the cache's memory.
- Scanning the 4-bit codes is not faster (slower at 1%, 12% faster at
  10%): decoding costs as much as the memory saved; agreement 98%.
- Accuracy of the best-balls count (simulator, ESCI stationary, 60k
  queries, one seed): 1% hit 9.9 -> 8.9%, 10% hit 30.1 -> 26.2%;
  retention .918 -> .922 / .912 -> .909. The restriction loses intruders
  from other balls' shells, so the model is less sure and serves less.
- Estimated end to end (decision time + DiskANN's measured 49.6 / 50.1 ms on
  misses, arithmetic): 1%: 0.21 + 0.911 x 49.6 = 45.4 ms (alone 49.6; equal
  memory to DiskANN 47.6); 10%: 0.51 + 0.738 x 50.1 = 37.5 ms (alone 50.1;
  equal memory 41.3, which does not yet charge the 22 MB HNSW graph).
Open: recover the hit rate lost at 10% (more balls in the count, or an
index over the shell sample); measure end to end directly.

Intruder count over more balls (`--shell-balls n`; the model is refit and
recalibrated for each; HNSW 32 for the nearest cached vectors; hit rate
and retention from the simulator, decision time from the timing harness;
overall = decision + (1 - hit rate) x DiskANN time, 49.6 / 50.1 ms; one
seed, ESCI stationary, 60k queries):
| budget | intruders over | hit % | retention | decision ms | overall ms | per-hit ms | per-miss ms |
|---|---|---|---|---|---|---|---|
| 1% | whole shell | 9.9 | .918 | 0.82 | | | |
| 1% | 3 / 5 / 10 / 20 balls | 8.9 / 9.1 / 9.1 / 9.2 | .922 / .917 / .918 / .920 | .21 / .20 / .24 / .30 | 45.4 / 45.3 / 45.3 / 45.3 | .2-.3 | 49.8-49.9 |
| 10% | whole shell | 30.1 | .912 | 12.7 | | | |
| 10% | 3 / 5 / 10 / 20 balls | 26.2 / 26.8 / 27.3 / 27.9 | .909 / .908 / .908 / .908 | .51 / .54 / .65 / .86 | 37.5 / 37.2 / 37.1 / 37.0 | .5-.9 | 50.6-51.0 |
- More balls recover only part of the lost hit rate (10%: +0.5-0.6 points
  per doubling of balls; 20 balls still 2.2 below the whole shell) and cost
  more time; overall latency is flat (37.5 -> 37.0 ms). The whole-shell
  count's extra 2.2 hits-points are worth about 1.1 ms but cost 12 ms.
  Default for now: 10 balls.
- Overall saving per query = hit rate x backend time - decision time
  (a hit saves the backend call; every query pays the decision), so the
  break-even hit rate is decision / backend time: 0.4-0.6% at 1%, 1.0-1.7%
  at 10% now, vs 49% at 10% before the fixes (24.6 / 50.1).
- Speedup on the queries that hit only: backend 49.6-50.1 ms vs 0.2-0.9 ms,
  about 60-250x; misses pay 0.2-0.9 ms extra (0.4-1.7%). Overall mean:
  9% faster at 1%, 26% faster at 10%.
- Aker's evaluation (section 5.3): overall QPS across workload skew (hit
  ratio rises with skew); overhead reported separately (lookup < 0.1 ms,
  a miss adds < 1 ms incl. entry insert); on DiskANN over SPACEV (3.7 ms per
  search) it says at low skew the hit rate "is not high enough to amortize"
  the lookup cost and QPS improves only at skew 0.99; on pgvector up to
  2-3x QPS; read amplification (pages read per query) and memory against
  the pgvector shared buffer.
Speedup limit: with hit rate h, end-to-end speedup is at most 1 / (1 - h)
even if hits were free (2x at 50%, 1.37x at 27%; ESCI 10% got 1.35x).
Aker's total hit ratios (its Figure 9, read off the plot, +- 0.02; in
brackets the exact-repeat part): SPACEV pool 1% / 5%: Zipf 0.30 .03 / .07,
0.60 .06 / .12, 0.99 .31 (.21) / .40 (.27), uniform replay .09 / .71 (all
repeats); TriviaQA 1% / 5%: 0.30 .25 / .34, 0.60 .29 / .40, 0.99 .58 (.32) /
.64 (.32), uniform replay >= .8 (all repeats). Above 50% only for TriviaQA
at skew 0.99 and for replays of unmodified queries; its "up to 2-3x QPS"
comes from those cases (DiskANN on SPACEV improves only at 0.99). Our
earlier simulated hit rates on the same traces (1%): Zipf 0.99 50.6% vs
Aker 35.8%, 0.6 13.0% vs 5.7%; so repeated-request workloads are where an
end-to-end gap should show. Report per run: overall speedup, hit rate (and
its 1 / (1 - h) limit), per-hit speedup and per-miss overhead.
End to end, measured directly (`eval_timing.py --decision hnsw32+balls10`;
state from a `--shell-scope balls --shell-balls 10` simulator run; each
query: hit decision, then DiskANN on a miss, timed together; 3000 queries
single-threaded; equal memory = DiskANN's node cache given our full memory
incl. the HNSW graph; ESCI 1M, EBS, DiskANN search list 128):
| | 1% cache | 10% cache |
|---|---|---|
| our memory (HNSW graph) | 18.6 MB (2.2) | 179.8 MB (22.1) |
| hit rate (decisions agree with simulator) | 11.8% (99.9%) | 27.9% |
| hit decision | 0.35 ms | 0.62 ms |
| DiskANN alone, mean / p99 | 48.8 / 53.8 ms | 49.2 / 54.4 ms |
| cache + DiskANN, mean / p99 | 43.5 / 54.5 ms | 36.1 / 54.9 ms |
| on hits / on misses | 0.32 / 49.3 ms | 0.58 / 49.8 ms |
| DiskANN with our memory, mean / p99 | 46.0 / 51.6 ms | 40.7 / 50.0 ms |
| overall speedup (limit 1 / (1 - h)) | 1.12x (1.13x) | 1.36x (1.39x) |
| per-hit speedup; miss overhead | 151x; +0.52 ms | 84x; +0.68 ms |
| vs DiskANN with our memory | 1.06x | 1.13x |
- Overall speedup is within 2% of the 1 / (1 - h) limit: the decision is now
  cheap enough that the hit rate alone decides the gain.
- The cache beats giving DiskANN the same memory, by 6% / 13% on the mean;
  but p99 is worse (every miss pays the decision; DiskANN's node cache
  speeds up every search).
- ESCI is a hard case (no repeated queries, hit rate <= 28%).
Open: throughput under concurrency (QPS, as Aker reports); workloads with
repeated requests (SPACEV Zipf), where the hit rate is far higher (needs a
DiskANN index over 10M vectors); fast local storage.
Old open list (kept for reference): measure end to end directly with the final decision path (HNSW + 10
balls) and equal memory (DiskANN's node cache also charged the HNSW graph);
throughput under concurrency (QPS, as Aker reports); workloads with
repeated requests (SPACEV Zipf), where the hit rate is far higher (needs a
DiskANN index over 10M vectors).

## 5ad. SPACEV data changes with baselines (round 2, gradual schedules)

Hit % / mean served retention (target .90); ours and LRU mean of seeds 0, 1;
Aker / QVCache best setting on seed 0 (highest hit rate with retention >= .9,
else highest retention); 1% budget; `online_data_summary.py --prep <prep>`:
| | Zipf 0.6 churn | Zipf 0.6 drift | real log churn | real log drift |
|---|---|---|---|---|
| static (no recalibration) | 13.6 / .911 | 15.1 / .867 | 4.1 / .947 | 4.9 / .917 (6 of 20 windows < T) |
| ours: recalibrate on recent labels | 12.9 / .922 | 12.7 / .916 | 4.1 / .946 | 4.3 / .948 |
| ours + score test | 12.9 / .922 | 11.3 / .922 (7.5 alarms) | 4.1 / .946 | 4.0 / .946 (7.5 alarms) |
| LRU (best K) | 11.8 / .918 | 11.6 / .919 | 3.4 / .935 | 3.5 / .935 |
| Aker | 5.7 / .907 | 5.7 / .901 (6 of 15 windows < T) | 2.6 / .896 (none meets T) | 2.6 / .892 (none meets T) |
| QVCache, deletes kept / removed | 2.7 / .925, 2.9 / .937 | 2.6 / .908, 2.9 / .906 | 2.7 / .927, 2.5 / .948 | 2.7 / .916, 2.5 / .923 |
| no cache upkeep | 9.7 / .907 | 6.3 / .898 | 1.7 / .921 | 1.9 / .915 |
- Ours leads at every workload: LRU by 0.7-1.1 points, Aker by 1.5-7 points,
  QVCache by 1.6-10 points. Aker meets the target only on Zipf 0.6 (and
  then with 6 of 15 windows below it under drift); on the real log no Aker
  setting meets it.
- Drift without recalibration breaks the promise on Zipf 0.6 (.867), as on
  ESCI; recalibrating on recent labels alone holds it.
- NEW: on SPACEV drift the score test fires repeatedly (7.5 alarms per run;
  0 on ESCI) and costs 0.3-1.4 points of hit rate without improving
  retention: ~12 database changes per query (vs ~0.9 on ESCI) change the
  cache's contents (upkeep) fast enough to move the score distribution, so
  the test sees cache changes, not workload changes. Possible fix: compare
  scores against the current cache (recompute the reference) or do not count
  score shifts caused by upkeep.
- Without upkeep, queries whose answers include new vectors keep only
  .71-.81 (drift).
Bursts and fresh-seed baseline re-runs: finished, not yet summarized.

## 5ae. Timing on SPACEV Zipf traces (repeated requests; DiskANN on EBS)

Setup: DiskANN 8-bit index over the SPACEV 10M base (100 dims, 100 bytes per
vector; graph degree 64, build list 100, 32-byte in-memory compressed codes;
built in 10 minutes, 5.7 GB on disk, recall@10 = .943 at search list 128;
`eval_real_backend.py --build-only --out-name spacev10m`, shared by all
SPACEV traces). Cache states from `online_sim.py --stream trace --shell-scope
balls --shell-balls 10` after the 20k-query warm-up; timed on the next 1000
requests of the trace (requests 20,000-21,000) (some repeat warm-up requests: Zipf 0.99 trace, 34% of
requests repeat an earlier one). Same harness as 5ac (decision: HNSW 32 over
cached vectors + intruders over 10 balls; DiskANN on misses; single
thread). Cached vectors are held as float32 here (400 bytes each) although
the data is 8-bit; "equal memory" gives DiskANN a node cache (360 bytes per
node) of our full memory incl. the HNSW graph.
| | Zipf 0.99, 1% | Zipf 0.99, 10% | Zipf 0.6, 1% | Zipf 0.6, 10% |
|---|---|---|---|---|
| our memory (vectors / shell / HNSW graph), MB | 64 (35 / 6 / 22) | 562 (336 / 7 / 214) | 64 (35 / 6 / 22) | 591 (347 / 17 / 221) |
| hit rate (decisions agree with simulator) | 50.9% (100%) | 61.8% (99.5%) | 13.5% (99.8%) | 23.0% (99.7%) |
| hit decision | 0.30 ms | 0.43 ms | 0.39 ms | 0.51 ms |
| DiskANN alone, mean / p99 | 51.4 / 65.8 ms | 51.7 / 67.6 | 52.7 / 71.5 | 51.0 / 65.9 |
| cache + DiskANN, mean / p99 | 25.8 / 62.7 | 20.3 / 60.5 | 46.7 / 71.9 | 40.1 / 65.7 |
| DiskANN with our memory, mean / p99 | 48.0 / 63.2 | 43.8 / 59.3 | 48.0 / 66.2 | 42.6 / 59.7 |
| overall speedup (limit 1 / (1 - h)) | 1.99x (2.04x) | 2.55x (2.62x) | 1.13x (1.16x) | 1.27x (1.30x) |
| per-hit speedup; extra cost per miss | 176x; +0.78 ms | 124x; +0.62 ms | 137x; +1.21 ms | 102x; +0.92 ms |
| vs DiskANN with our memory | 1.86x | 2.16x | 1.03x | 1.06x |
- The overall speedup is again within 2-4% of the 1 / (1 - h) limit; the gain
  is decided by the hit rate: 2.0-2.6x at skew 0.99, 1.1-1.3x at 0.6.
- The hit decision stays under 0.6 ms even with 850k cached vectors.
- Versus DiskANN given the same memory the gain is 1.9-2.2x at 0.99 and only
  1.03-1.06x at 0.6.
- The HNSW graph is 38-39% of our memory at 100 dimensions (22 of 64 MB at
  1%; 214 of 562 MB at 10%; M = 32), more than at 384 dimensions (12%).
  Smaller M, or storing cached vectors as 8-bit, would change the totals;
  not tried.
- p99 is about unchanged (misses pay the decision).
- Hit rates here are those of the 1000 post-warm-up requests; they match the
  simulator's longer runs (about 50% and 13% at 1%). No updates during the
  timed requests (static cache).
- Comparison with Aker's paper (different hardware and backend): its up to
  2-3x QPS at skew 0.99 on pgvector and, on DiskANN over SPACEV (3.7 ms per
  search there), improvement only at 0.99.

## 5af. Real Aker (the authors' code) on the SPACEV Zipf traces

What: github.com/sjoon-oh/aker (commit 6bb326a), built in Standard mode (its
adaptive per-entry threshold), driven through a thin C shim
(`aker_real/aker_shim.cc`) that follows the authors' pgvector integration:
look up; on an approximate hit link the query to the representative it
matched (so a repeat of it is answered from that representative); on a miss
the backend answers k + delta neighbors (delta = Aker's reserve neighbors)
and the entry is inserted. Requests are handled one at a time, so the cache
changes right away. Harness: `aker_real/eval_aker_real.py` (quality mode:
backend answers = exact top-k; `--timing`: real DiskANN on misses). Build
steps: `aker_real/build.sh` (deps in `~/third_party`: Boost 1.86, FAISS 1.8.0
C++, xxHash, YCSB-C). Budget = pool of budget x 10M vector slots. Same
traces, warm-up (first 20k requests not measured) and k = 10 as ours.

How Aker decides a hit, in plain terms: each cached query gets a radius,
starting at 1/4 of the squared distance to its nearest neighbor. A new query
is served from the closest cached query (found with a small HNSW, M = 4, ef
8) if it is inside that radius. Exact repeats widen the radius (x 1.1, up to
the nearest-neighbor distance); each approximate hit shrinks it (x
alpha_tighten). Nothing checks whether an approximate answer was right.

Two fixes before it worked (both outside Aker's caching logic):
- A memory bug in Aker: `StackFloatBuffer` (inc/utils/ak_stack_alloc.hh)
  calls alloca() inside its constructor, so the buffer is freed as soon as
  the constructor returns. The float copies of queries and representatives
  that feed the HNSW filter were dead stack memory; in our build the filter
  found nothing, so there were zero approximate hits. Reproduced in a
  standalone C++ test with no shim; the earlier smoke run's few hits were
  luck. Fix (`aker_real/aker_stackbuffer_fix.patch`, applied by build.sh):
  keep the buffer inside the object. After the fix Aker serves 287 of the
  472 queries its own rule says should hit in the first 3000 requests (the
  rest are missed by its small HNSW).
- Aker never sets a log level, so Boost printed several debug lines per
  request (250 MB per run) on the hot path; the shim keeps warnings only.
  Also, FAISS's OpenMP threads made 4 parallel runs crawl (load 300);
  all runs use OMP_NUM_THREADS=1.

Check against the paper: SPACEV Zipf 0.99 at 1% gives 31.1% hits (exact
16.5% + approximate 9.4% + linked 5.2%) with the authors' defaults (delta 5,
alpha_tighten 0.9); the paper reports about 31% (about 21% exact, 10%
approximate). It matches.

Quality (80k measured requests per run). Sweep: alpha_tighten 0.9 (the
authors' config file) or 0.25 (the paper); delta 0 / 5 / 10. Best hit rate per
cell (delta 0, alpha_tighten 0.9 everywhere):
| | Zipf 0.99, 1% | Zipf 0.99, 10% | Zipf 0.6, 1% | Zipf 0.6, 10% |
|---|---|---|---|---|
| real Aker: hit rate (exact + approx. + linked) | 34.1% (18.9 + 9.2 + 6.0) | 44.7% (25.5 + 7.1 + 12.1) | 7.1% (1.5 + 5.4 + 0.2) | 15.7% (5.3 + 9.2 + 1.2) |
| real Aker: share of right answer kept, all hits | .917 | .915 | .804 | .842 |
| ... on approximate / linked hits | .78 / .87 | .77 / .82 | .75 / .82 | .76 / .80 |
| real Aker memory (its own estimate) | 52 MB | 189 MB | 51 MB | 251 MB |
| ours: hit rate, kept (target .9) | 50.6%, .912 | 62.3%, .901 | 13.0%, .924 | 23.3%, .906 |
- Approximate hits keep only 75-78% of the right answer in every setting;
  neither alpha_tighten nor delta changes that. Aker clears .9 overall at
  skew 0.99 only because exact repeats (always right) are most of its hits.
  At skew 0.6 it misses .9 (.79-.87) in every setting.
- alpha_tighten 0.25 trades hits for quality (0.99, 1%: 31.7% at .941; 0.6,
  10%: 13.7% at .866), still below .9 at 0.6. Larger delta costs hits at 1%
  (reserve neighbors take slots) and nothing at 10%.
- At 10% Aker's pool never fills on either trace (65k distinct queries x 10-15
  neighbors < 1M slots): 10% is "cache everything" for Aker.
- Ours gets 1.4-1.5x Aker's hits at 0.99 and 1.5-1.8x at 0.6, while meeting
  the .9 target where Aker does not. (Ours at 1%: round-2 simulator runs, r2_trace, 2 seeds;
  at 10%: the 1000 timed requests of 5ae, no longer run exists.)
- Our reimplementation of Aker (5z; tau_init 0.25, alpha_t 0.9) gave 30.5%
  at .968 on 0.99 / 1%: about the same hit rate as the real code, but the
  real code's answers are worse (.913). Part of this is linked hits, which
  our reimplementation does not model.

Timing (`--timing`): same 1000 requests as 5ae (20,000-21,000), one at a
time on one thread: Aker lookup; on a miss DiskANN (8-bit SPACEV index, list
128, k + delta = 10 neighbors), read their vectors, insert. Equal memory:
DiskANN node cache = Aker's memory at that point (as in 5ae).
| | Zipf 0.99, 1% | Zipf 0.99, 10% | Zipf 0.6, 1% | Zipf 0.6, 10% |
|---|---|---|---|---|
| real Aker: hit rate on these 1000; kept | 35.6%; .919 | 39.5%; .919 | 5.3%; .828 | 9.6%; .836 |
| Aker lookup, mean | 0.019 ms | 0.018 ms | 0.024 ms | 0.024 ms |
| DiskANN alone, mean / p99 | 50.9 / 66.3 ms | 51.3 / 66.2 | 52.0 / 69.0 | 51.6 / 68.0 |
| Aker + DiskANN, mean / p99 | 33.3 / 65.3 | 31.6 / 65.1 | 49.4 / 66.7 | 46.4 / 66.2 |
| DiskANN with Aker's memory, mean | 48.1 | 48.7 | 48.6 | 47.9 |
| speedup (limit 1 / (1 - h)) | 1.53x (1.55x) | 1.62x (1.65x) | 1.05x (1.06x) | 1.11x (1.11x) |
| vs DiskANN with Aker's memory | 1.45x | 1.54x | 0.98x | 1.03x |
| ours (5ae): speedup; vs equal memory | 1.99x; 1.86x | 2.55x; 2.16x | 1.13x; 1.03x | 1.27x; 1.06x |
- Aker's hit check is 15-20x cheaper than ours (0.02 vs 0.3-0.5 ms) but
  both are under 1% of a 50 ms DiskANN search, so the speedup is set by the
  hit rate alone (both reach 97-100% of 1 / (1 - h)). Ours is faster
  end to end in all four cells.
- Aker's memory at the timing point is 52-75 MB (the 10% runs have only
  seen 20k requests); ours is 64 MB at 1% and 562-591 MB at 10%. At 1%
  memory is about equal, and ours wins on time (1.99x vs 1.53x at 0.99;
  1.13x vs 1.05x at 0.6) and on quality.
- The timed hit path included about 0.08 ms of retention bookkeeping per hit
  (fixed in the code afterwards; under 0.2% of the mean, against Aker).
- Not done: Aker's data-change path (write log, deletes) on our data-change
  workloads; the shim has `shim_write` / `shim_process` / `shim_delete`.

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

Priorities set by the user (after the current data-change runs finish):
1. Approximate backend: stop assuming exact answers. Both steps: a
   backend with controlled recall in the simulator, then a real ANN index.
   Report retention both against what the backend returns and against
   exact truth (user: both matter; decide later which the promise uses).
2. Real performance numbers: the cache in front of a DISK-BASED index
   (user decision). Time everything on the online hot path needed to decide
   hit vs miss (features: distances to cached vectors and the shell sample,
   the model, the cutoff check), plus the backend call on a miss.
   Disk-based indexes also keep data in memory (DiskANN: compressed PQ codes
   of all vectors, a cache of nodes near the entry point; SPANN / IVF:
   centroids), and the OS page cache can hold index pages; memory limits
   must be controlled and memory given to our cache compared at equal total
   memory with giving it to the index.
   Backends: DiskANN swept from small compressed codes (8-32 bytes per
   vector, node cache off) to its default, and pgvector IVFFlat (centers in
   memory, Postgres buffer at its minimum, process memory capped) as the
   minimal-memory end. Storage: EBS is fine for now (higher, more variable
   latency than local NVMe; note it).
   Cache budgets: start with 1% and 10% of the vectors (user decision; the
   full 0.1%-10% sweep later). Aker's own settings were 1% / 5% (hit
   ratios) and 1% / 2% / 5% (latency), counted as raw cached vectors.
Other open items (ranked list in the conversation of this session): the
rebuild-on-return flaw (5u), the score's dependence on the sampling rate p,
mixed k, gradual shifts, model refit, TriviaQA and smaller budgets, a
data-volume-triggered shadow rebuild, and the items below.

- Evaluate retention-estimate error and its bias directly, not radius error,
  including the selection effect under a running-average policy.
- Compact density-aligned estimator: C cluster centers (e.g. a k-means coarse
  quantizer) with one stored radius each. Sweep C and report skill vs
  footprint as a fraction of the budget B.
- Fix the density-feature off-by-one and test it under queries that do not
  follow the data (e.g. Anupta's locality-constructed stages).
- TODO (user, parked): a feature for the overlap between the top-n balls
  (with many overlapping balls their union covers more than the best 3
  show, so the model is too pessimistic there). Remind when asked for
  undone TODOs.

## 8. Environment notes

- SIFT data downloaded to `datasets/sift/raw_siftsmall/` (siftsmall, 10k
  base) and `datasets/sift/raw_sift1m/` (SIFT1M, ~168 MB archive). Source:
  ftp://ftp.irisa.fr/local/texmex/corpus/.
- Installing matplotlib upgraded numpy from 1.21.1 to 2.0.2 on this machine;
  `requirements.txt` does not pin numpy.
- All SIFT radius results above use k = 10 and Euclidean distance on raw
  vectors.

## 9. Experiment coverage matrix (as of 5af)

Methods: O = ours, L = LRU (our hit rule, LRU population), A = Aker
reimplementation, A* = real Aker code, Q = QVCache reimplementation (no
public code). Number types: S = simulator with an exact backend; Si =
simulator with an inexact backend (controlled recall, or real HNSW / IVF
answers); T = real code, timed against real DiskANN. Budget in brackets.
Online runs use k = 10, T = 0.9; ours / LRU 2 seeds; baselines tuned on seed 0.

Stable data and workload:
| workload | S | Si | T |
|---|---|---|---|
| ESCI 1M random | O L A Q (10%) | O (10%) | O (1%, 10%) |
| SPACEV Zipf 0.3 | O L A Q (1%) | - | - |
| SPACEV Zipf 0.6 | O L A Q (1%); A* (1%, 10%) | - | O A* (1%, 10%) |
| SPACEV Zipf 0.99 | O L A Q (1%); A* (1%, 10%) | - | O A* (1%, 10%) |
| SPACEV real log -> official | O L A Q (1%) | - | - |
| TriviaQA Zipf 0.3 / 0.6 / 0.99 | offline only (5i, 5j) | - | - |

Workload shift (single shift; multishift = start, mild, full, return):
| workload | S | Si | T |
|---|---|---|---|
| ESCI 1M shift, multishift | O L A Q (10%) | - | - |
| SPACEV Zipf 0.6 shift | O L A Q (1%) | - | - |
| SPACEV Zipf 0.3 / 0.6 / 0.99 multishift | O L A Q (1%) | - | - |

Data changes (churn / drift; 10% of the database changed; gradual / burst):
| workload | S | Si | T |
|---|---|---|---|
| ESCI 1M | O all configs, both schedules; L A Q gradual (10%) | - | - |
| SPACEV Zipf 0.6, real log | O L A Q, both schedules (1%; bursts and baseline fresh-seed re-runs not yet summarized) | - | - |
| SPACEV Zipf 0.3 / 0.99, TriviaQA | not prepared | - | - |

Not covered anywhere: workload + data change together; SPACEV online at 10%
or ESCI online at 1%; budgets other than 1% / 10% online; k != 10 or T != 0.9
online; baselines with an inexact backend (the simulator refuses A / Q);
change rates other than 10%; concurrency (QPS); local NVMe; QVCache timing.

### 9a. Campaign to fill the matrix (launched; results pending)

Code changes:
- Real Aker harness (`aker_real/eval_aker_real.py`): `--stream` (trace,
  shift, multishift, random), `--data-change churn|drift --schedule
  gradual|burst` (inserts reach Aker through its write log, deletes
  through markVectorDeleted, applied at the request where they are
  scheduled; the write-log slow path is called after every request as in
  the authors' integration; truth = first k present ids of the union
  ranking; returned deleted vectors counted), `--backend-recall`
  (the simulator's approximate_rows model) and `--backend-file` (real
  index answers). Fixed-data result reproduced exactly (31.12%).
- Simulator: Aker / QVCache reimplementations now run with an inexact
  backend (labels, cache contents and theta from the backend's answers;
  retention also logged against exact truth). Tests pass (29).
- Shared float32 copies of the bases (`datasets/aker/<stem>-f32.npy`,
  memory-mapped by `load_base_f32`; SPACEV 4 GB, SPHERE 31 GB): each
  SPACEV simulator process no longer holds a private 4 GB copy. `load_base`
  still returns int8 for SPACEV (DiskANN).
- Job generators: `online_round2_jobs.py --budget B --preps ...` (folders
  r2_<stream>_b<B>_s<seed>), `online_data_jobs.py --round 3 --preps ...
  --budget B` (data_<kind>_<sched>[_b<B>]_s<seed>), new
  `online_backend_jobs.py` (bk_<stream>_<backend>_s<seed>; simulated recall
  .95 / .9; real HNSW / IVF on ESCI, DiskANN on SPACEV),
  `aker_real/aker_real_jobs.py` (results/aker_real/<prep>/<setting>_b<B>/
  at<a>_d<d>.json; grid alpha_tighten .9 / .25 x delta 0 / 5 / 10, budgets
  1% / 10%). TriviaQA traces added to the round-2 and backend workloads.
- Driver: `run_matrix.sh` (copy of /tmp/mx/driver.sh): three pools (heavy:
  ours / LRU at a 10% SPACEV budget and on TriviaQA, ~17-20 GB each; light:
  the rest; real Aker), phases gated by the data preps; phase 4 re-runs the
  best seed-0 baseline settings on seeds 1, 2.

Prep running: SPACEV Zipf 0.99 / 0.3 data-change rankings (top-1000 over
10M); TriviaQA online prep (top-500 over SPHERE 10M, 768-d), multishift,
data-change rankings, serialized after the first two top-500 jobs (each
step ~65 GB; the 0.6 one was killed for memory when started alongside
everything else, so it now runs alone); real DiskANN answers for the four
SPACEV traces (recall@10 tuned to .95 / .9).
Pitfall: SPACEV simulator runs at a 10% budget peak ~20 GB (vs ~2 GB
private at 1%); 40 of them at once would not fit in 371 GB.

Not in this campaign (timing-sensitive, need a quiet machine): timing under
shifts and data changes, concurrency (QPS), NVMe.

First results, real Aker (phase 1 complete, 312 runs;
`aker_real/summarize_aker_real.py`; hit % / retention vs the backend's
answer; best grid setting, '!' = no setting reaches .9):
| workload | stable 1% / 10% | shift 1% / 10% | multishift 1% / 10% | churn gradual 1% / 10% | drift gradual 1% / 10% | sim. recall .9 backend 1% / 10% |
|---|---|---|---|---|---|---|
| Zipf 0.3 | 3.3 / .763 ! ; 9.2 / .813 ! | 6.1 / .778 ! ; 9.1 / .811 ! | 5.4 / .771 ! ; 8.7 / .808 ! | (prep) | (prep) | (prep) |
| Zipf 0.6 | 6.3 / .815 ! ; 13.7 / .866 ! | 9.7 / .829 ! ; 13.3 / .863 ! | 8.2 / .819 ! ; 12.9 / .857 ! | 5.9 / .813 ! ; 9.8 / .838 ! | 5.9 / .799 ! ; 9.9 / .803 ! | 6.3 / .751 ! ; 13.8 / .813 ! |
| Zipf 0.99 | 34.1 / .917 ; 44.7 / .915 | 38.6 / .916 ; 43.7 / .911 | 37.1 / .915 ; 42.8 / .919 | 31.3 / .907 ; 42.5 / .906 | (prep) | 31.6 / .911 ; 42.5 / .919 |
| real log | 5.1 / .735 ! ; 8.9 / .778 ! | | | 4.4 / .729 ! ; 8.1 / .786 ! | 4.5 / .729 ! ; 8.3 / .755 ! | 5.1 / .717 ! ; 9.3 / .765 ! |
- Real Aker reaches .9 only at Zipf 0.99 (exact repeats carry it); on every
  other workload and setting it stays at .72-.87, so it has no setting that
  meets our target there.
- Aker never returned a deleted vector (its delete path works); under data
  changes it keeps its hit rate, but at 10% the setting meeting .9 at 0.99
  churn burst costs half the hits (23.2 vs 44.0 for the default at .867).
- With a .9-recall backend, retention vs exact truth drops further (0.99,
  10%: .860 vs .919 against the backend); against its own backend it
  still reaches .91-.92 at 0.99.
- Multishift full phase (0.99): Aker falls to .875-.883 for that phase.

### 9b. TriviaQA embeddings (Aker's queries are in the wrong space) and campaign repairs

TriviaQA / SPHERE embeddings (checked 2026-11-10; raised by Anupta):
- Aker's released request vectors (triviaqa-sim-100k-<z>.npy) are
  BAAI/bge-base-en-v1.5 raw CLS outputs: re-encoding lines of triviaqa.txt
  reproduces them (L2 ~0.02). The paper says BGE-large, which is 1024-d;
  its own table lists 768-d. Aker's repo has no embedding code.
- The SPHERE base is not in that space. Its first 100K rows are exactly
  Hermes' SPHERE_100K (huggingface mohdumar/SPHERE_100K, which has the text;
  downloaded to datasets/aker/sphere100k/). Re-encoding that text with the
  passage side of Sphere's DPR_web bi-encoder (facebookresearch/Sphere,
  dpr_web_biencoder.cp; bert-base-uncased, CLS, 256 tokens; input =
  (title, text without the title) as a sentence pair) gives cosine .998 to the
  stored vectors (min .991; text only .992; bge-base ~.5). So the base is
  DPR_web passages and the queries should be DPR_web questions.
- With the released queries every query sees a flat neighbor profile (top-1
  and 1000th neighbors almost equally far): near-arbitrary ground truth.
- Fix (decided with Jialin): keep Aker's database vectors and request
  sequences, re-embed each request's question with the DPR_web question
  encoder (`datasets/reembed_triviaqa_queries.py`: identifies each released
  vector's line by its bge-base embedding, writes
  triviaqa-dpr-sim-100k-<z>.npy + match statistics). New workloads
  triviaqa-dpr-<z> (run_e2e_oracle_trace.WORKLOADS); triviaqa-aker-<z> is kept
  for the earlier offline runs (5i, 5j), which used the mismatched queries.
  Encoders in HF format: ~/third_party/sphere/dpr_web_{question,ctx}_encoder.
- Metric: DPR is an inner-product model; under Euclidean distance (Aker's
  choice, kept here) neighbors are still poorly separated (median d10 /
  d1000 = .967 for DPR_web questions) because passage norms enter the
  distance. With inner product the top passages are on topic even within
  100K random web passages (e.g. a Robert Redford question -> Hollywood
  screenwriter / actor biographies). Open option: the standard
  inner-product -> Euclidean transform (append sqrt(M^2 - |x|^2) to every
  database vector and 0 to every query; Euclidean neighbors then equal
  inner-product neighbors), which would keep all distance-based machinery.
  Not run yet (each 768-d exact-neighbor prep costs hours).
- The first TriviaQA preps (mismatched queries) were set aside as
  results/online/triviaqa-aker-<z>-bgequeries; no runs used them.

Campaign repairs (night of 11-09):
- Out-of-memory kills: with 20 light + 8 heavy slots, 37 runs died
  silently (kernel OOM killer, no traceback; logs stop mid-run), mostly LRU
  at a 10% budget (grows past 22 GB) and QVCache / Aker at 10% on the data
  workloads (12-18 GB each, not ~5 GB as assumed). Now every job goes
  through `mx_run_job.sh` (start only when enough memory is free: heavy 70 GB,
  10%-budget 45 GB, others 20 GB; retry jobs that die without output, at
  most 2 retries; never duplicate a running job) and `mx_watchdog.sh`
  (below 20 GB free, kill the most recently started job, which the runner
  retries later; status line every 10 min in /tmp/mx/status.log; restarts
  the driver if it dies). Pool caps lowered to heavy 5, 10%-budget 10,
  1%-budget 12.
- SPACEV real log + DiskANN: recall@10 reaches only .886 at search list
  1000 (real queries far from the 10M base), so the .95 / .9 targets give
  an empty answer file; re-tuned to .85 / .8 (`/tmp/mx/diskann_real.sh`).
- Driver v3: `run_matrix.sh`; TriviaQA: `run_matrix_triviaqa.sh` (prep
  serialized, one ~65 GB step at a time; runs queued as inputs land).

Campaign repairs, continued (11-10 morning):
- Hidden memory consumer: /tmp is a RAM-backed tmpfs here, and Aker exports
  ~300 MB of telemetry per run to /tmp/aker_trace_<time>_pid<PID>_gen<G>/;
  519 such dumps held 62 GB of RAM ("shared" in `free`), which caused the
  repeated memory crunches. The real-Aker harness now deletes its dump on
  close; the watchdog removes any left over (> 10 min old).
- A TriviaQA exact-neighbor prep (~65-80 GB) was itself OOM-killed while it
  ramped up next to the simulator pools. Now: each prep step waits for 120 GB
  free and sets /tmp/mx/HOLD for its first 15 min (mx_run_job.sh starts
  nothing new meanwhile); the watchdog sets oom_score_adj 800 on light jobs
  and 500 on heavy jobs so the kernel kills them before a prep step (0).
- Heavy pool reordered (`run_matrix_heavy4.sh`, 7 slots): every cell's seed 0
  first (ours, then LRU), then seed 1. Durations at a 10% SPACEV budget:
  ours ~28 min, LRU ~1 h 40 min per 100k-request run.
- Monitoring: `sh /tmp/mx/check.sh` (status, kills, failures, progress per
  pool); /tmp/mx/status.log (every 10 min), watchdog.log, failures.log.
  The watchdog supervises driver3.sh, heavy4.sh and tqa2.sh (restarts them
  if they die) and re-queues given-up jobs hourly (at most twice per job).
State 11-10 09:18: heavy 92 / 384, light 1% 1594 / 2852, light 10% 1706 /
2208, real Aker (SPACEV) done; TriviaQA Zipf 0.6 prep in its last stage,
then 0.99 and 0.3, then their runs.

### 9c. TriviaQA with the inner-product transform (second variant; decided with Jialin)

DPR_web is an inner-product model, Aker searches by Euclidean distance. On a
2M-vector sample: the top-10 by plain Euclidean and by inner product share only
20% of their members, and Euclidean favors low-norm passages (median norm of
its top-10 14.13 vs 14.62 for inner product, database median 14.39). The
ratio d10 / d1000 is not a fair contrast measure after the transform (a
constant offset in the squared distance compresses it: .975), what matters is
the ranking.
Transform (`datasets/make_sphere_ip_transform.py`): append sqrt(M^2 - |x|^2)
to every database vector (M = 16.746, the largest norm; norms 13.4-16.7,
median 14.39) and 0 to every query; Euclidean neighbors then equal
inner-product neighbors (checked: identical top-10 for 50 of 50 queries over
300k rows). Written as datasets/aker/sphere-ip-10m.npy (769-d float32, 31 GB,
memory-mapped via a -f32 symlink) and triviaqa-ip-sim-100k-<z>.npy; workloads
triviaqa-ip-<z>. TriviaQA now has two variants through the whole pipeline:
dpr (Euclidean on the DPR_web vectors, as Aker) and ip (transformed).
`run_matrix_triviaqa.sh [dpr|ip]`: both launchers share one prep lock
(/tmp/mx/prep.lock), the "already done" check is made after taking the lock,
job lists are named by variant; watchdog supervises both. Each variant costs
3 top-500 preps (~2 h each) and 6 data-change preps (~2-2.5 h each), strictly
one at a time.

### 9d. Quiet-machine guard for timing runs

Timing results are only valid if nothing else competed for CPU, memory
bandwidth or the disk. `simulations/approximate_hits/timing_env.py`:
- before a timing run, measures for 10 s how much CPU and disk traffic OTHER
  processes cause (machine-wide busy time minus the timing process's own,
  /proc/stat vs /proc/<pid>/stat; block-device bytes minus its own,
  /proc/diskstats vs /proc/<pid>/io) and refuses to start unless others use
  <= 1 core and <= 50 MB/s (`--wait-quiet MIN` waits for quiet instead,
  `--allow-busy` runs anyway and records the contention);
- during the run, a separate sampler process (not a thread: diskannpy holds
  the GIL for minutes while loading an index and starved a first thread
  version) records the same quantities every 2 s; validated: 24 added
  busy processes show as +23 cores, the timing process's own load is excluded,
  124 of 124 expected samples;
- the result JSON gets "environment": verdict quiet / contended, start check
  (busiest processes), mean / p95 / max load of others during the run, load
  averages, host and CPU model. Wired into `eval_timing.py` (both paths) and
  `aker_real/eval_aker_real.py --timing`.
- `python simulations/approximate_hits/timing_env.py audit` lists the verdict
  of every timing result. All 20 existing ones predate the guard ("no record";
  by file timestamps nothing else wrote results during their windows). The
  headline timing numbers will be re-run under the guard once the campaign
  has drained; timing under shifts / data changes only runs under it.

### 9e. Headline figure, draft (simulations/approximate_hits/plot_headline.py)

Design: one panel per workload; x = cache size (log), y = hit rate; one line per
technique, each point = the best hit rate among that technique's settings that
reach recall 0.9 (within .01). Techniques with no such setting are drawn faded
with a dotted line and their best recall printed. A strip under each panel shows
recall - 0.9; the right axis converts hit rate to the speedup bound 1/(1-h).
Ours / LRU: mean of 2 seeds; Aker (real code) / QVCache: best of the tuned grid on
seed 0. Drafts: results/summary/headline_draft_v1.png, _v2.png (not committed).
Finding from the draft (checked in the logs): our cache is frozen after warm-up
(e.g. SPACEV Zipf 0.99, 10%: 2,811 entries before and after; 82-85% of the budget
used) while LRU (our hit rule + LRU population) keeps admitting every miss (2,811 ->
9,356-10,183 entries). With a full budget at 1% ours wins or ties; at 10% LRU beats
ours by 5-30 points (SPACEV Zipf 0.3/0.6/0.99: 27.5/36.9/67.1 vs 14.6/24.5/60.0;
TriviaQA dpr Zipf 0.6: 88.7 vs 57.7). Open design question: online admission for
ours, or treat LRU population + calibrated hit rule as part of our method.

### 9f. Why LRU beats ours at 10% (analysis of existing runs, 11-10; CORRECTED 11-10)

"LRU" = our calibrated hit rule + LRU population (admit every miss, evict the least
recently used entry). Ours = greedy selection from the 20k warm-up requests, then frozen
(stable streams never alarm, so no rebuild).
What limits the warm-up cache (code: online_sim.warmup / greedy_entries):
- The warm-up splits the 20k requests into 4k candidate queries, 8k scoring queries, 4k
  to fit the model, 4k to calibrate (--warm-split), and greedy fills at most f = 85% of
  the budget (the rest is for the shell sample); it also stops when no candidate helps
  any scoring query. `--c-max 3000` / `--s-max 25000` apply ONLY to shadow rebuilds
  (after an alarm: at most 3000 new candidate queries, each needing a top-500 backend
  lookup and evaluation at up to six ball sizes, and 25000 scoring queries); they were
  written into the first simulator as cost bounds, with no recorded experiment behind
  the values. An earlier version of this section wrongly blamed c_max for the warm-up
  cache.
- Budget use at 10% (greedy allowance 850k vectors): SPACEV Zipf 0.3 / 0.6 / 0.99 use
  100 / 100 / 97%, TriviaQA dpr Zipf 0.6 / 0.99 87 / 61%. So unused budget is real only
  on TriviaQA; on SPACEV ours fills its allowance and LRU still wins with FEWER cached
  vectors (573k / 622k / 725k).
- Ball size: ours spends the budget on about 3,000 large balls (280 vectors per entry on
  SPACEV, 185-204 on TriviaQA); LRU holds 13-16k small balls (K = 20: 37-78 vectors per
  entry). Greedy maximizes hits on an 8,000-query scoring sample, which favours large balls
  around the sample's popular queries; when 62-98% of later requests are queries never seen
  before, covering the tail needs many small balls near many different past queries.
- Eviction: at 1% LRU evicts constantly and loses to ours in every 10k-request window; at
  10% on SPACEV it fills its budget and evicts yet still wins; on TriviaQA Zipf 0.6 it
  never fills (.84 -> .96), so it never evicts and still wins by 30 points.
- Learning after warm-up: popularity is stationary, so little is left to learn about the
  distribution, but LRU's hit rate still climbs over the stream (SPACEV 0.6, 10%: 27 ->
  39% per 10k requests; ours flat at 23-26%). TriviaQA groups (a question and its 50
  paraphrases): 84% of post-warm-up traffic goes to groups present in the warm-up
  (92-97% to groups seen at any earlier time), so one entry per warm-up group could reach
  about 84%; ours holds 3,645 entries for 7,729 warm-up groups. The remaining 4-5 points of
  LRU's lead come from groups first seen after warm-up.
- Exact repeats are not the explanation: only 1.6-11.5% of post-warm-up requests exactly
  repeat a post-warm-up request.
- Stationary traces have no "different windows": popularity does not change; the windows
  differ only in which new queries appear.
Probes (probe_jobs.py, probe_summary.py; 10% budget, seed 0; compared on requests
60,000-100,000): more candidates (--warm-split), small balls only (--max-ball-k), a
longer warm-up, and LRU with the same warm-ups. Results: see 9h.

### 9g. Does real Aker reproduce the paper? (11-10)

Real Aker on the RELEASED traces and Aker's own ground truth (gt_ids from the .pkl;
`aker_real/prep_aker_released.py`, results/aker_paper_check/), the paper's pools 1% / 5%,
reserve 5. Hit %: after our 20k warm-up / Aker's own count over all requests; recall:
over hits / over all requests (misses answered exactly here; the paper's pgvector
backend had ~.9 on misses). Paper values read off its Figure 9 (+- .02):
| workload | pool | paper | alpha_T .25 (paper) | alpha_T .9 (config file) |
|---|---|---|---|---|
| SPACEV 0.3 / 0.6 / 0.99 | 1% | 3 / 6 / 31 | 2.3 / 4.8 / 27.9 (recall all .99 / .99 / .98) | 2.4 / 5.4 / 30.6 |
| SPACEV 0.3 / 0.6 / 0.99 | 5% | 7 / 12 / 40 | 7.2 / 10.9 / 39.2 (.98 / .98 / .98) | 7.8 / 12.5 / 41.8 |
| TriviaQA 0.3 / 0.6 / 0.99 | 1% | 25 / 29 / 58 | 20.8 / 25.6 / 54.5 (.90 / .89 / .89) | 24.6 / 31.6 / 60.9 (.88 / .86 / .81) |
| TriviaQA 0.3 / 0.6 / 0.99 | 5% | 34 / 40 / 64 | 34.3 / 39.6 / 64.3 (.87 / .87 / .86) | 46.3 / 51.6 / 70.3 (.79 / .78 / .75) |
- The hit ratios reproduce the paper (SPACEV both settings; TriviaQA at the paper's
  alpha_T .25 within 1-4 points at 1% and exactly at 5%), and so does its recall claim
  (>= .8 over all requests).
- The gap to our numbers is the metric: the paper averages recall over ALL requests
  (misses get the backend's recall) with target .8; we average over HITS with target .9.
  Over hits alone, real Aker keeps .76-.95 on SPACEV and .53-.81 on TriviaQA.
- The config file's alpha_T .9 gives more hits but fails the paper's own .8 bar on
  TriviaQA at 5% (.75-.79); our earlier grid includes both settings.
- TriviaQA here = the released (mismatched-encoder) queries, i.e. the paper's setup;
  our triviaqa-dpr / -ip workloads re-embed the queries and are not comparable to it.

### 9i. Queue overload and global memory admission (11-10 evening)

What went wrong: per-pool slot caps do not add up. The TriviaQA launchers start one new
pool per trace as its prep lands, the hourly re-queue starts another pool each hour, and the
LRU probes and real-Aker paper check added their own; about 70 jobs ran at once (26 of them
ours / LRU at a 10% budget). The watchdog then killed young jobs in a loop (273 kills) and
333 jobs ran out of retries.
Memory facts (private memory, /proc/<pid>/smaps_rollup; RSS overstates it because every
process maps the shared database, SPHERE 31 GB): ours at a 10% budget peaks ~17 GB while
building its warm-up cache, then holds ~9-14 GB; LRU at 10% grows slowly (5-7 GB after 5 h);
other jobs 1-5 GB; a TriviaQA prep step 35-90 GB.
Fix (`mx_run_job.sh` v3): GLOBAL admission under a lock. A job starts only if committed
memory + its peak estimate <= 320 GB and MemAvailable >= its estimate + 30 GB. Committed =
running jobs at max(class peak, private) in their first 30 min, then 1.2 x private; a prep
step at max(90, private). A cap of 350 GB caused two dips to 6-10 GB (kernel OOM kills), so
320 + headroom. The 16 youngest heavy jobs (< 60 min old) and 58 idle old-version wrappers
were stopped; `mx_requeue.sh` now re-queues every job that started (log exists) but has no
result and no traceback, from every queue list (cap 2 per job); the watchdog's emergency
threshold is 15 GB.

Decisions (11-10, Jialin): DiskANN for the real-backend (timing) plot is tuned to recall@10
0.95 per dataset, the same index and search list for every technique (SPACEV: answers at 0.95
exist; the real query log reaches only .886, so .95 is unreachable there; TriviaQA indexes queued:
`run_diskann_triviaqa.sh`, one index per database variant, built under the prep lock). The
small-ball probe (--max-ball-k) is a diagnostic only, never a method: K <= 20 / 50 gave 11.3 /
16.0% vs 24.5% default on SPACEV 0.6 at 10%. Planned Aker tuning to reach the target: Standard
mode with alpha_T down to 0.01 and alpha_L 1.0, and Proximity mode (the authors' compile option,
one global threshold) swept down until recall 0.9.
Queue state 11-10 22:30: 8,676 distinct jobs in all lists; 6,482 done; 2,194 unfinished (357
started without a result, 1,837 never started because pools were cut short) now in one sweep
queue (/tmp/mx/sweep.sh, 60 slots, global admission).

### 9j. Scope cut to the two paper plots; Aker tuned to reach 0.9 (11-10 night)

Decision (Jialin): only the headline hit-rate plot (exact backend) and the timing plot (real
DiskANN at recall@10 0.95) are run now. Everything else is dropped from the queue:
`/tmp/mx/skip.txt` lists 2,142 jobs (448 of the detector factorial on TriviaQA, 1,694 for other
figures: data changes, inexact simulated backends, shifts). `mx_run_job.sh` exits for a listed
job, at start and before every admission attempt / retry (the in-loop re-check added 23:15, after
58 listed jobs admitted earlier were still running or waiting; they were stopped).
TriviaQA: headline-only launcher `run_triviaqa_headline.sh <variant>`.

Aker tuned to reach mean retention of hits 0.9 (`aker_real_jobs.py --tuned`, 140 jobs):
- Standard mode with alpha_T 0.1 / 0.01 x alpha_L 1.0 / 1.1 (top_delta 5);
- Proximity mode (the authors' compile option: one fixed global threshold instead of
  per-entry adaptive ones; separate build `~/third_party/aker_prox`, `libakershim_prox.so`,
  threshold via env AKER_GLOBAL_THRESH), `eval_aker_real.py --prox-frac f`: threshold = f x the
  median squared distance from a trace query to its exact nearest neighbor, f in .01-.3.
First results (23:00, about a quarter done): Proximity mode reaches 0.9 where Standard mode
cannot, at low hit rates (SPACEV 1%: Zipf 0.3 0.8% at prox .05, Zipf 0.6 3.8% at prox .1).

### 9k. Timing harness: every technique in front of the same DiskANN (eval_timing_online.py)

Design (agreed with Jialin): DiskANN on SPACEV 10M, search list 160 (recall@10 .954, tuned to
0.95), the same index and settings for all techniques; one request at a time, one thread,
quiet-machine guard. Per request, timed together: the cache lookup; on a hit the cached answer
(ours / LRU: with probability 0.1 DiskANN is also asked, to label the hit for calibration); on a
miss DiskANN's answer and the cache's update. Then the same 2,000 requests on DiskANN alone and on
DiskANN with its node cache given the technique's memory. Reported: mean / p50 / p99 over all
requests, hits and misses apart, speedup vs DiskANN, the 1 / (1 - h + audited share) limit,
stage times, recall vs exact (all requests and hits) and the simulator's retention (share of
DiskANN's answer that is cached) over hits.
Techniques:
- ours: the greedy cache built in warm-up from DiskANN's answers (online_sim --save-state with
  --backend-file ...diskann.npz --backend-target 0.95, shell scope = 10 best-fitting balls),
  static afterwards; hit path = FAISS HNSW (M 32, ef 32) over the cached vectors + intruder count
  over the 10 best-fitting balls; recalibration every 1,000 requests on the last 20,000.
- LRU: the same warm-up state, then every miss becomes an entry (K 20, shell from DiskANN's
  top-500 sampled at the state's rate p), least recently used entries evicted; the same hit path,
  maintained incrementally (vectors added to the HNSW graph, efConstruction 40 for inserts, 2x
  cheaper than 200; evicted vectors masked; graph rebuilt with 32 threads when > 30% are dead,
  counted as background work). Distinct shell samples are counted once (memory matches the
  simulator: 805,552 vs 805,526 vector units at 10%, Zipf 0.99).
  LRU needs DiskANN's top-500 on every miss (k 500, search list 500: ~3x a top-10 search). Two
  modes, `--maintenance sync` (the miss is answered by the top-500 search, the insert on the
  request's path) and `background` (answered by the normal top-10 search; the top-500 search,
  insert and evictions are background work, their time reported apart). Both are run.
- Aker: the real library (Standard or Proximity build), warmed up on 20,000 requests with
  DiskANN's answers (untimed).
- QVCache: as its paper describes it (C++ with Python bindings, FreshVamana mini-indexes):
  mini-indexes = diskannpy DynamicMemoryIndex (FreshVamana, C++), region thresholds over a PCA
  projection as in online_baselines.QVCache, warmed up like Aker. The simulator's QVCache
  searches its mini-indexes exactly; here they are graph searches.
Aker / QVCache run the setting with the best hit rate among those reaching retention >= 0.89 in
the headline results (`timing_jobs.py`, via plot_headline.collect).
Smoke tests (busy machine, 300 requests, numbers not usable): all four run; ours at 10% Zipf .99
hit rate .563, retention over hits .905 (simulator definition); LRU .567 / .905; under load the
LRU insert costs 15-70 ms per miss (HNSW inserts, single thread), so its sync miss latency is
DiskANN's top-500 search plus the insert.
Batch: `run_timing_batch.sh` (detached, log /tmp/mx/timing.log) waits for the tuned-Aker runs
and the warm-up states, writes /tmp/mx/timing.txt (SPACEV Zipf .99 / .6 / .3 x 1% / 10% x ours,
LRU sync, LRU background, Aker, QVCache = 30 jobs, ~8-10 min each), takes the prep lock, holds
the queue (HOLD), waits until no simulator / Aker / prep / index-build process is left, runs the
jobs one at a time under the guard, then releases. Results:
`results/timing_v2/spacev-aker-<z>/<technique>_b<b>.json`.

### 9l. Timing batch results and decisions (11-11)

- Batch: blocked 02:37-12:38 (its drain check matched the queued wrappers, which wait on HOLD:
  deadlock; fixed, `d26c7e9`), then ran all 30 jobs 12:38-19:14. Guard verdicts "contended" on
  9 runs come from ~1 core of interactive sessions (p95 1.02 cores, mean .27, I/O < 5 MB/s on 96
  cores); timings are usable.
- Result: in every run, speedup over DiskANN alone = 1 / (1 - h + audited share) within ~3%
  (a hit costs ~0.03-0.9 ms without an audit vs ~63 ms per DiskANN search). Ours' and LRU's
  mean hit cost is 6-8 ms because 10% of hits are audited (one DiskANN search each).
  DiskANN given the cache's memory as node cache: 1.05-1.17x.
- Caveat: the timed window is the 2,000 requests right after warm-up, so its hit rates differ
  from the headline (whole-run) ones; e.g. at 10% LRU and ours are both ~57% (Zipf .99), while
  the headline has LRU 67.0 vs ours 59.7 (LRU's advantage builds up later).
- Decision (Jialin): LRU's cache update is charged as background work (plot: LRU = lru_bg;
  sync runs in the table only). Sync costs: top-500 search ~190 ms per miss + insert.
- TriviaQA primary variant: ip (inner-product transform), since DPR_web is trained and meant to
  be searched by inner product; plain Euclidean shares only 20% of the top-10 and favors
  low-norm passages. The dpr DiskANN index is built (sphere10m; recall@10 .24 at search list
  128); its tuning was stopped, the ip index is building (`run_diskann_triviaqa_ip.sh`).
- SPACEV real query log: not a hard limit; tuning stopped at search list 1000 (.886, still
  rising ~.024 per step). Real-log queries are far from the base and have little contrast:
  median distance to the nearest neighbor 125.7 vs 81.2 for the Aker-trace queries, and d10 / d1
  1.038 vs 1.105 (d100 / d10 1.035 vs 1.086). Extended curve (search lists 1000-6400) running:
  `results/backends/spacev-real-hiL/log_curve.txt`.

### 9m. Timing v3: whole stream, sampled timing; LRU fidelity fixes (11-12 night)

Decisions (Jialin, 11-11): time the same requests as the headline (the whole stream after
warm-up); all cache updates are background work; stationary workloads, so ours runs static (no
recalibration, audits or drift tests: in the headline runs no test ever fired, and the static
cache meets the target with the same hit rates); LRU keeps recalibrating (its cache changes on
every miss). Check sample rates 1 in 20 / 10 / 5.
- Replay: DiskANN is deterministic: its stored answers equal live searches exactly (300 queries,
  top-10 / 15 / 20 at search list 160 and top-500 at 500, ids and order). eval_timing_online.py
  now drives the cache through every request; every --sample-every-th request is timed end to end
  with live DiskANN calls, the others use the stored answers; DiskANN alone / with equal memory
  run on the timed sample; per-request latencies saved (<out>_lat.npz).
- LRU fixes. (1) The warm-up state now carries the warm-up's calibration labels (online_sim
  save_state cal_*; states regenerated in results/timing_v3); the calibrator is seeded with
  them, as the simulator keeps them (before: the first recalibration used ~1,000 labels and raised
  the cutoff: 4.7% vs the simulator's 12.2%). (2) Shell samples exclude cached vectors (the
  simulator's shell = union of shells minus cached vectors; also in the memory count). (3) New
  shell vectors quantized to 4 bits like the simulator's sample (warm-up sample's range).
  Check, SPACEV Zipf .6, 1%, requests 20k-24k, identical settings (DiskANN answers, shell scope 10
  balls): harness 9.3% vs simulator 10.1% (per-request vs batch-of-1,000 decisions, HNSW).
- Batch (`run_timing_batch.sh`, log /tmp/mx/timing3.log): 24 runs at 1 in 20 (SPACEV Zipf .99 /
  .6 / .3 x 1% / 10% x ours, LRU, Aker, QVCache) + sample-rate check (1 in 10 and 1 in 5 for ours
  and LRU on Zipf .6 1% and Zipf .99 10%). Plot: plot_timing.py -> timing_draft_v2.png (table
  includes the sample-rate comparison).
- The queue had run dry at ~22:00 11-11 (watchdog exited "all done"); ~6 h idle. New watchdog
  `mx_watchdog2.sh` supervises the current launchers; hourly re-queue of old lists disabled.
- TriviaQA ip: tuned Aker (60 + 12 jobs incl. Proximity fractions .003 / .001) running; the
  32-byte-PQ DiskANN index was unusable (recall@10 .11 at search list 1000; 769 dims at 24 dims
  per PQ byte vs ~3 for SPACEV): rebuilding with 256-byte PQ (`run_diskann_triviaqa_ip_pq.sh`),
  tuning to .95 after the timing batch (efforts up to 4000). Old index results moved to
  backends/triviaqa-ip-<z>-pq32.
- SPACEV real log, extended curve (5,000 tuning queries): recall@10 .9476 at search list 2000,
  .9597 at 2500 (batched 134 / 167 ms per query with 32 threads vs 12 ms at 160 for the Aker
  traces): .95 is reachable at ~2,100-2,500, at roughly 10x the per-query cost.
- Headline: TriviaQA row now ip (primary variant); extra cache sizes 0.3% and 3% (912 jobs:
  SPACEV traces, TriviaQA ip; ours, LRU, QVCache grid, real Aker standard + tuned) queued to start
  when the timing batch ends (/tmp/mx/hx.sh); plot_headline.py BUDGETS = .3, 1, 3, 10%;
  /tmp/mx/after_timing.sh redraws headline_draft_v3.png every 2 h and the timing plot at the end.

### 9n. LRU empty-search bug; chained batches; timing as a line plot (11-12 afternoon)

- LRU timing bug: once in ~15,000 lookups the incrementally built FAISS HNSW over the cached
  vectors returned no result (62k live vectors indexed); the score became infinite, one infinite
  label made every cutoff fail (lambda None), and LRU served nothing for 20,000 requests until the
  label aged out (Zipf .99, 1%: 33.9% vs the simulator's 46.8%). Fix: exact search over the live
  cached vectors when the graph search returns fewer than k (1 fallback in 40,000 requests);
  the calibrator drops non-finite labels; ours' static hit path refuses to serve on a short result.
  After the fix: 45.8% over 40,000 requests (simulator ~47%). All LRU timing runs are redone
  (old ones in results/timing_v3/old_lru_before_fix).
- TriviaQA 256-byte-PQ DiskANN: recall@10 .752 at search list 128 (32-byte PQ: .039); confirms the
  PQ diagnosis. No TriviaQA experiment ever used the 32-byte index (no TriviaQA real-backend run
  exists; its tuning never reached the target).
- Normalizing TriviaQA instead of the 769-d transform: cosine shares only 57% of the inner-product
  top-10 (2M-passage sample, 200 queries; Euclidean on raw 20%); norms 13.4-16.4 matter for DPR.
- Chain (run_timing_batch2.sh <set> <name> <flags>): TIMING3 (matrix + sample-rate check) ->
  TIMING3B (LRU re-time, 7 runs) -> extra headline sizes 0.3% / 3% (hx.sh) + warm-up states for
  0.3% / 3% + TriviaQA DiskANN tuning -> TIMING3C (timing at 0.3% / 3%, every technique).
  after_timing.sh redraws the plots as batches land.
- plot_timing.py is now a line plot like the headline (timing_draft_v3.png): speedup over DiskANN
  alone vs cache size per technique, DiskANN with the cache's memory as node cache (grey), and a
  recall-vs-exact strip; a marker is hollow when that run's own whole-stream retention of hits
  is < 0.89. QVCache's settings (picked on the exact-backend headline) miss with DiskANN at 1%
  (Zipf .99: retention .75; Zipf .6 / .3 also hollow).

### 9o. Retention is measured on the served answer; accurate in-cache search (11-12)

Targets are retention relative to the backend (share of the backend's top-k, averaged over hits),
not recall vs exact (kept: it is what the cache can observe and certify, index-agnostic; recall
vs exact is reported alongside). Found: the label "share of the backend's top-k that is cached"
is not what is served when the in-cache search is approximate. Timing runs (HNSW over the cached
vectors, search list 32) at 10%: cached share .91-.92 but served overlap with DiskANN's top-10
.86-.88 (in-cache 10-NN recall at search list 32: .83-.89; 128: .95-.97; 256: .975-.985; 512:
~.989, ~1 ms). Decisions (Jialin): both fixes.
- Label = served retention (online_sim --label served, the new default): share of the backend's
  top-k in the answer the cache would serve (its k nearest cached vectors). With the exact backend
  it equals the old label except for distance ties (SPACEV is 8-bit: one batch .9209 vs .9215 in a
  4-batch check; hits identical), so the headline needs no re-run.
- Timing harness: in-cache HNSW search list 256 (ours and LRU); labels on the served answer; the
  warm-up calibration set (state cal_q / cal_w, new) is relabeled on the harness's own serving
  path and the cutoff recomputed before timing (untimed); the plot judges every technique by the
  served answer's overlap with DiskANN's top-10.
- Validation (Zipf .99, 1%, 20,000 requests): ours hit 48.5%, served overlap .913, recall vs exact
  on hits .916; LRU 45.6% (when its cutoff holds), served .905.
- Open: LRU sometimes loses its cutoff near request 35,000 (2 of 7 runs; lambda None, no hits
  for 20,000 requests); not tied to the exact-search fallback alone (fallbacks without collapse in
  4 runs). Fixed-order selection stops at the first failing threshold, so a few bad labels among
  the highest scores void every cutoff. LRU jobs now dump their labels when the cutoff is lost
  (CAL_DEBUG, <run>_caldbg.npz); plot_timing flags such runs.
- The sample-rate check was stopped after 2 points (1 in 20 vs 1 in 10: speedup 1.130 vs 1.119);
  ours / LRU states regenerated (served labels), ours and LRU re-timed (set "retime"); earlier
  runs in results/timing_v3/old_before_served_labels.

### 9p. Calibration walk starts at the fit split's most certifiable threshold (11-12 night)

Problem: the fixed-order walk (select_weighted: strictest grid threshold first, stop at the first
failing one-sided 95% normal lower bound) tests the top 1% of scores first (~40 of 4,000 labels);
a few low labels there fail it and no cutoff survives. Seen: timing state Zipf .99, 10% (served
labels): mean retention .95-.99 in every score band, cutoff None, ours 0 hits; LRU's intermittent
collapses; and in the headline, 44 runs had batches with no cutoff (SPACEV Zipf .3 1% LRU 43-48 of
80 batches, ours 5-7; SPACEV real log ours 19-23 of 104, LRU up to 42).
Fix (decided with Jialin): `threshold_selection.start_from_fit`: on the FIT split (independent of
the calibration labels), the grid threshold with the highest fit-set lower bound; the calibration
walk starts there (stricter thresholds untested). Validity unchanged: the starting point does not
depend on the calibration labels. Rejected: a fixed minimum count (100 works on these states by
luck; 200 skips the correct cutoff of Zipf .3 1%, which admits 160 of 4,000; no count fits all
hit rates). Where nothing is certifiable (few good hits, modest margin) the method serves nothing
(safe; e.g. timing state Zipf .3 at 0.3%).
- online_sim --start fit (default), state saves fit_q / lam_start; the timing harness relabels the
  fit split on its serving path and uses its own start.
- New states: Zipf .99 10% cutoff -0.115 (admits 2,320 of 4,000); the other five unchanged
  (as expected: the walk result changes only where it failed at a skipped threshold).
- Re-runs: headline ours / LRU (128 jobs, /tmp/mx/hr.sh; old results in
  r2_*/old_before_fit_start); extra sizes (hx.sh, its ours / LRU killed and retried on the new
  code); timing "final" set (ours / LRU at 1% / 10%, every technique at 0.3% / 3%) after
  HX + HR + states (run_timing_batch2.sh final TIMING3C ...). Old timing ours / LRU in
  results/timing_v3/old_before_fit_start.

### 9q. Self-adjusting queue; papers' own baseline settings in both plots (11-13)

Queue (Jialin: make the fixes self-adaptive, not ad hoc):
- mx_run_job.sh v4: peak private memory of every run recorded per class (prep, budget, method) in
  /tmp/mx/peaks.tsv (sampled every 20 s); admission estimate = max(prior table, 1.3 x largest
  finished peak, 1.6 x largest killed peak). Measured: SPACEV LRU at 10% 12-29 GB (prior was 10).
  A run that dies without output and without a traceback (memory kill) is retried with back-off
  2-32 min, up to 20 attempts, so memory pressure costs time, not results.
- mx_complete.sh <list> <flag>: runs a list, then checks the OUTPUTS (JSON exists / skipped /
  traceback) and re-runs what is missing (up to 5 rounds) before setting the flag (flags used to
  be set when a list had been worked through; the 3% timing states were marked done while missing).
- run_timing_batch2.sh v5: no drain by process names; HOLD + SIGSTOP of all our other jobs
  (simulator, real Aker, index tuning, preps; also ones started later), resumed at the end; the
  per-run quiet guard (--wait-quiet) decides by measurement.
- 11-13 losses (old runner): 398 memory kills; headline re-runs 62 / 128, extra sizes 543 / 720,
  extra-size Aker 170 / 192, 3% timing states missing; all re-queued (run_headline_complete.sh).

How the papers set their parameters (neither tunes to a recall target):
- Aker: fixed default (alpha_T, alpha_L) = (0.25, 1.1), "prioritizes search accuracy", no reserve
  (Delta = 0); reports the resulting recall (>= .8 over all requests); backends tuned to recall
  .9 without the cache.
- QVCache (arXiv 2602.02057): alpha .9, 4 mini-indexes, PCA 16 dims x 8 buckets; D chosen by
  starting at 0 and raising it by .025 while the hit ratio keeps improving meaningfully
  ("D in [0, .5] is a practical regime"); recall is monitored, no target.
Plots: both now show, per baseline, the hand-tuned setting (best hit rate among grid settings
reaching .9, chosen after the fact on the same run) and the paper's setting (dashed): Aker
at0.25_d0 (real code); QVCache m4 / r16x8 with D from the paper's procedure over our D grid
(0 / .025 / .075 / .15 / .25 / .5; stop when the next D gains < 5% relative). headline_draft_v4:
the paper settings miss .9 almost everywhere (recall .13-.87). Timing "final" set: 60 runs
(paper settings at all four sizes; ours / LRU at 1% / 10%; all at 0.3% / 3%).
- TriviaQA DiskANN at recall .95 needs search list 2000: 3.3 s per query (one thread, loaded
  machine) vs 63 ms on SPACEV; full-stream timing at 1 in 20 would take 7-10 h per run.

### 9r. Why DiskANN looked so slow: EBS; TriviaQA index study; ESCI timing (11-13)

- Storage: the only disk is Amazon EBS (nvme0n1 "Amazon Elastic Block Store"; no local instance
  store, no root for a RAM disk). DiskANN reads 4 KB sectors with O_DIRECT. A random 4 KB read
  measured 6.5 ms (p50; p99 10 ms) while the TriviaQA tuning (24 threads) and two index builds ran:
  the volume was saturated, so the 3.3 s per TriviaQA search (search list 2000) was mostly
  contention. Quiet: the SPACEV timing runs imply ~0.7 ms per read (63 ms at search list 160,
  beam 2), ~7x local NVMe. Latency is only meaningful on a quiet machine (the timing runs' guard
  enforces it; ad-hoc measurements under load are not).
- Even quiet, TriviaQA at search list 2000 would be ~0.7 s (0.9 at 800: ~0.3 s), so the index
  itself is the problem: recall .752 at search list 128 (SPACEV: .95 at 160). Study
  (index_study.py, recall vs search list on 2,000 queries, no latency): base (256-byte PQ, degree 64,
  build list 100) vs pq768 (1 byte per dimension), r128 (degree 128, build list 200), mips (native
  inner product on the raw 768-d vectors), mem (in-memory graph, no PQ: graph quality alone).
  The tuning run on the old index (triviaqa-ip-0.3) was stopped (its answers would be obsolete).
- SPACEV real log: .95 needs search list ~2,100 (quiet ~0.7 s); .9 ~1,250; .85 800; ~100 ms
  allows search list ~280 (recall ~.72). Decision pending.
- ESCI added: prep_io.load_prep (trace and ESCI preps) for eval_timing_online.py and
  eval_aker_real.py; ESCI answers at .95 / .9 (eval_real_backend, index exists: .961 at search list
  128); headline ESCI now with real Aker (stable grid + tuned, random stream) at all four sizes
  (run_headline_esci.sh; plus 0.3% / 3% for ours / LRU / QVCache); timing states for ESCI and ESCI
  in the final timing set and as a fourth panel.

### 9s. Disk = EBS by design; beam width; quiet latency probe (11-13)

- Jialin: EBS counts as the disk (any slower layer below the cache); not an issue to discuss.
- The 3.3 s TriviaQA figure was an ad-hoc check run outside the harness while the queue was busy
  (no quiet guard); every timing run uses the guard (other processes <= 1 core and <= 50 MB/s,
  verdict recorded) and the batch suspends our other jobs; the SPACEV numbers are quiet.
- Why search list 2000 is slow on disk while cheap in memory: each expanded node is one random
  4 KB read (~0.7 ms quiet here); with beam width W, W reads per round trip, so latency ~
  (expanded nodes / W) x read latency (compute negligible). W = 2 (DiskANN default) was never tuned.
- latency_probe.py: on the quiet machine, recall and single-query latency per (search list,
  beam width 2 / 4 / 8 / 16) for SPACEV, SPACEV real log, ESCI, TriviaQA (current index and the
  study's disk variants); also the share of queries whose top-10 equals the beam-2 result (the
  replay's stored answers are beam-2 searches). Timing runs then use, per index, the fastest beam
  width at the tuned search list with >= 99% identical results and recall within .005
  (timing_jobs.beam_for); runs at another width go to results/timing_v3_w<W> and every technique
  is re-timed there. Chain: probe batch (after the index study and ESCI states) -> final batch.

### 9t. Queue v5: registry, machine-state admission, priorities, adaptive reservation (11-13)

Problem (Jialin asked whether the fixes were ad hoc; they partly were): admission and suspension
only saw processes they knew by name; work outside the queue (index builds, tuning, hand-run checks)
was invisible, and the watchdog killed queued plot jobs while the exploratory builds that caused
the pressure kept running.
- Registry (/tmp/mx/registry/<pid>: class, priority, estimate), written by the runner for every
  python process it starts. Priority via MX_PRIORITY: plot (default) or adhoc; timing runs are
  registered by the timing batch as "timing". oom_score_adj plot 300, adhoc 1000.
- Admission from the machine's state: used = MemTotal - MemAvailable (everything, ours or not) +
  F x expected growth of registered jobs (class estimate - private memory); start only if
  used + growth + estimate <= 345 GB and available - growth - estimate >= 20 GB.
- Estimates learned per class: max(prior, 1.2 x largest finished peak, 1.1 x largest killed peak)
  (1.6 x killed ratcheted estimates up under pressure: a kill follows a full machine, not this
  job's need).
- Adaptive reservation factor F (/tmp/mx/growth_factor, start .5): the watchdog adds .25 per memory
  kill (max 1) and subtracts .1 per hour without kills with >= 60 GB available (min .2).
- Watchdog v4 kill order below 15 GB: registered adhoc, then unregistered python of ours >= 2 GB
  (ad hoc by definition), then registered plot (youngest first; the runner retries it); timing never.
  First use: 16:03 it killed the ad-hoc pq768 index build (7 GB available) instead of plot jobs.
- 16:45: priors lowered to modest values (underestimates only cost a retry; learned peaks take over;
  high only for index builds and ours / LRU at 10%); F decays every 15 min (was hourly), since
  over-reservation by unseen classes held 200 GB idle.
- Timing batches suspend every python process of ours except registered timing runs.
- Jobs without a JSON output (index builds) also go through the runner (success = exit 0; kills
  retried); the index study runs fully managed (run_index_study_managed.sh), builds restarted from a
  clean folder.

### 9u. DiskANN tuned per workload (search list x beam width); Aker's DiskANN settings (11-13)

Quiet probe (latency_probe.py, one search at a time, 100 queries; v1 in results/latency_probe_v1):
beam width (reads in flight per round trip) was never tuned (2 everywhere) and dominates latency on
this disk, at equal or slightly higher recall: SPACEV search list 160: 64 ms (beam 2) / 22 (8) /
15 (16); search list 2000: 733 / 215 / 126 ms; ESCI 128: 50 / 17 / 11 ms; SPACEV real log 1250
(.91): 466 / 135 / 79 ms; TriviaQA (current index) 512 (.86): 198 / 65 / 42 ms. All timing so far
used beam 2, i.e. a backend ~4x slower than it needs to be: the final timing batch was stopped.
Aker (paper; the released code has no DiskANN integration): DiskANN built with degree 64 / build
list 100 and 1.1 GB in-memory PQ on SPACEV (~110 bytes per vector; ours 32), degree 96 / build
list 100 and 5.1 GB PQ on SPHERE / TriviaQA (~512 bytes; ours 256); search parameters tuned to
recall@10 0.9 without the cache; no beam width reported; 3.7 ms (SPACEV) and 21.1 ms (TriviaQA)
per query on a local NVMe SSD.
Redo (Jialin: go ahead), run_backend_redo.sh:
- builds with the paper's settings (index_study variants spacev_pq110, aker_tqa; plot priority),
  besides the TriviaQA study variants (pq768, r128, mem; ad hoc);
- quiet probe of every candidate index: search list x beam width 2-32, recall on 5,000 queries
  (batched) and single-query latency (100 queries);
- select_backend.py: per workload the index / search list / beam width with the lowest mean
  latency at recall@10 >= 0.95 (0.9 when 0.95 needs > 100 ms) -> backends/<prep>/choice.json;
- stored answers regenerated at exactly that setting (eval_real_backend.py --fixed target:L --beam
  W --index-dir; old files diskann_v1.*); the timing harness takes beam, index and target from them;
- warm-up states and timing in results/timing_v4; final set (112 runs, ordered): SPACEV x3 (4 sizes
  x ours / LRU / Aker / QVCache tuned, papers' settings at 1% / 10%), ESCI (same), TriviaQA x3
  (1% / 10%, four techniques), SPACEV real log (1% / 10%); plot_timing: 8 panels in 2 rows.

### 9v. Other workloads with repetition; per-workload backend tuning (11-13)

- Real or near-repeated query workloads in related work:
  - Proximity (EPFL; arXiv 2503.05530) MedRAG-Zipf: PubMed 23.9M snippets (MedCPT, 768-d),
    PubMedQA's 500 questions drawn 10,000 times with Zipf 0.8, each repeat reworded by an LLM
    (synthetic repetition, near-duplicates in space); also MMLU over wiki_dpr (21M passages), each
    question 4 times with small textual prefixes.
  - TripClick (arXiv 2103.07901): real click logs of the Trip Database health search engine,
    ~5.2M interactions 2013-2020 over ~1.5M MEDLINE articles; head / torso / tail test sets by real
    query frequency; text only (we would embed documents and queries).
  - Not verified here: ORCAS (MS MARCO clicks with query counts), LMSYS-Chat-1M (LLM prompts).
  Not added yet (asked Jialin).
- DiskANN selection was per dataset (recall from one trace's queries applied to all three Zipf
  traces). Now per workload: workload_recall.py measures recall@10 over each workload's own
  REQUESTS (5,000 sampled from the stream after warm-up, repeats included) for every probed
  (search list, beam width); latency stays per index (quiet probe); select_backend.py uses the
  per-workload recall. Added to run_backend_redo.sh between the probe and the selection.
- SPACEV with the Aker paper's 110-byte PQ (spacev_pq110): recall .9506 at search list 128, .9589
  at 160 (32-byte PQ: .943 / .953).

### 9w. New workloads: MedRAG-Zipf and TripClick (11-13, started)

- Authors: Proximity (arXiv 2503.05530, EuroMLSys '25) is EPFL SaCS (Kermarrec; Randl, de Vos,
  Pires, Petrescu) with Huawei Zurich (Zhang Ji); QVCache (arXiv 2602.02057, Feb 2026) is EPFL DIAS
  (Ailamaki; Goecer, Tsakalidou, Nicholson, Kim). Different groups; QVCache does not cite Proximity.
  Proximity: one global similarity threshold tau, hand-set ("a hyperparameter to be optimized during
  deployment"), flat or LSH lookup, in-memory FAISS backends (HNSW for wiki_dpr, Flat for PubMed).
- MedRAG-Zipf (Proximity): PubMed corpus with MedCPT (768-d); PubMedQA questions drawn with Zipf 0.8,
  each repeat reworded by an LLM. Corpus vectors exist: NCBI hosts MedCPT article embeddings of all
  PubMed (ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/, 38 chunks x ~1M, float32, plus PMIDs);
  MedRAG's own precomputed archive (UVA SharePoint) now returns 403. Downloading the NCBI chunks
  (datasets/medrag/download_medcpt.sh, rate-limited to 15 MB/s for the quiet timing guard).
  Queries: PubMedQA questions embedded with the MedCPT query encoder (cheap). Proximity did not
  release its reworded queries (the repository holds only the cache code).
- TripClick: text only (no vectors), access by a signed request form (non-commercial; data and
  derived models may not be shared publicly).

### 9x. TripClick workload: data, log profile, encoding (11-13)

- Access: Hugging Face dataset `TripClick/tripclick` (gated; token in ~/.cache/huggingface/token,
  mode 600). The TripClick-trained encoder (Navid/tripclick-bert-base-dot) is not found, so documents
  and queries use MedCPT (ncbi/MedCPT-Article-Encoder, [title, abstract], 512 tokens;
  ncbi/MedCPT-Query-Encoder, 64 tokens; CLS; inner product), the encoder family of the PubMed
  embeddings used for MedRAG. Downloaded by datasets/tripclick/download.py into datasets/tripclick/raw.
- Not shareable: everything under datasets/tripclick/ except the .py scripts is git-ignored, as are
  datasets/aker/ (vectors) and results/ except summary plots. Never commit TripClick-derived files.
- Log: 5,317,350 clicks -> 2,457,146 searches (one per run of consecutive same-query clicks in a
  session; query lowercased, whitespace-normalized), 1,671,169 distinct; 32% of all searches repeat
  an earlier query; top 1000 queries cover 7.3%; 2013-2020 at ~290-360k searches / year.
  Exact-repeat share of a window grows with its length (2018-01-01: 50k 13.2%, 100k 16.6%, 200k 19.9%,
  400k 22.1%; similar for 2016 and 2019 starts).
- Window used: 100k consecutive searches from 2018-01-01 (2018-01-01 to 2018-04-12, 83,367 distinct,
  16.6% exact repeats), matching the 100k requests of the other traces. Approximate hits can exceed
  the exact-repeat share (near-duplicate queries); workload name `tripclick-ip` in
  run_e2e_oracle_trace.WORKLOADS (base tripclick-ip-docs.npy, 1,523,878 x 769).
- Inner product -> Euclidean as for TriviaQA ip: documents get sqrt(M^2 - |x|^2) appended, queries 0.
- Encoding speed (CPU): fp32 48 threads 6.9 docs/s (61 h for 1.52M). bf16 autocast at 8 threads per
  process ~12.7 docs/s per process on a quiet machine (8.4 under load ~56), cosine to fp32 >= 0.99995
  (checked on 96 documents; inner-product relative error <= 1.3e-3). Mean 340 tokens, 11% truncated
  at 512. build_workload.py: bf16 autocast, length-sorted batches (order restored), 20k-document shards
  written atomically, `--part i/N` workers. Run as 6 parts x 8 threads through the queue
  (/tmp/mx/tripclick.sh: TC_ENC -> assemble + queries -> online_prepare -> TC_PREP.DONE), ~7 h.
