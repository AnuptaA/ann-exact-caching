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
Open: measure end to end directly with the final decision path (HNSW + 10
balls) and equal memory (DiskANN's node cache also charged the HNSW graph);
throughput under concurrency (QPS, as Aker reports); workloads with
repeated requests (SPACEV Zipf), where the hit rate is far higher (needs a
DiskANN index over 10M vectors).

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
