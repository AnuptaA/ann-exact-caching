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
- Encoder memory (11-13): the first 6 workers grew to ~36 GB each (216 GB). A 640-document test
  stays at 3.7 GB, so the growth is in the number of distinct batch shapes (length-sorted batches:
  hundreds), most likely per-shape state of the CPU bf16 kernels (e.g. a repacked weight copy).
  Fix: pad every batch to a multiple of 64 tokens (`pad_multiple=64`, at most 8 shapes; speed
  unchanged in the test). Restarted workers hold at ~7 GB. Shards of the first run (0-5) were kept.

### 9y. MedRAG-Zipf workload (11-13)

- Proximity's recipe (arXiv 2503.05530, 4.2): MIRAGE's 500 PubMedQA questions (`benchmark.json`,
  MIRAGE GitHub), 10,000 independent draws with Zipf 0.8 (no temporal locality), every occurrence
  reworded by GPT-4o, each version unique across the dataset (they checked the RAG answer is the same
  for all rewordings); corpus MedRAG PubMed, 23.9M snippets, MedCPT, FAISS-Flat (exact).
- Ours (decided with Jialin 11-13: local-LLM rewording): Qwen2.5-7B-Instruct, Q8_0 GGUF, served by
  llama.cpp (built in ~/third_party/llama.cpp, CPU, `-DLLAMA_UI_GZIP=OFF` since the web-UI asset
  step fails offline) on 127.0.0.1:18080, 64 parallel slots (`/tmp/mx/llama_server.sh`). Hugging Face
  transformers bf16 generation on this CPU was ~0.11 rewordings/s (bf16 matmuls not faster than
  fp32: no AMX path); llama.cpp batched: ~116 generated tokens/s at 64 parallel under load.
- Draw (`datasets/medrag/build_workload.py draw`, seed 0): top question 771 occurrences (paper ~700),
  median 10, min 1.
- Rewordings (`datasets/medrag/paraphrase.py`): sampled (temperature 1, top-p 0.95) with one of 10
  wording hints that all require the same meaning; for questions with >= 20 accepted versions, 15 of
  them are shown with "worded differently from these"; unique after normalization (lowercase,
  alphanumerics) against the original and all accepted versions; verified by the same model
  (greedy, "do A and B ask exactly the same thing, same yes/no/maybe answer?"; only Yes accepted).
  Stand-in for Proximity's same-answer check. Stats in `datasets/medrag/paraphrase_stats.json`.
- Corpus: NCBI MedCPT article embeddings (all PubMed, 38 chunks) restricted to the PMIDs of MedRAG's
  corpus (`medrag_pmids.py` streams the 70 GB MedRAG/pubmed JSONL and keeps only PMIDs), inner
  product -> Euclidean (769-d) -> `datasets/aker/medrag-ip-docs.npy` (~74 GB). Requests: rewordings
  in stream order (each question's versions shuffled), MedCPT query encoder, 0 appended.
- Workload `medrag-ip` in run_e2e_oracle_trace.WORKLOADS. Stream length 10,000, so the simulators'
  warm-up default became min(20000, a fifth of the FULL stream) (online_sim, eval_aker_real,
  eval_timing_online; every existing stream has >= 88k requests, so nothing else changes).
- Chain: `/tmp/mx/medrag.sh` (rewordings -> query vectors; downloads -> PMIDs -> base; prep ->
  `run_headline_new_workload.sh medrag-ip`).

### 9z. New workloads in the headline and timing plots (11-13)

- `run_headline_new_workload.sh <prep>`: the full TriviaQA-ip-0.6 job set (every line that ran for it:
  ours / LRU configurations, Aker / QVCache grids, real Aker paper and tuned settings, 0.3/1/3/10%),
  latest line per output JSON, workload name substituted (292 simulator + 68 real-Aker jobs).
- `run_timing_new_workload.sh <prep>`: DiskANN (degree 64, build list 100, 256-byte PQ) -> quiet
  probe (after TIMING4) -> workload recall -> select_backend -> answers -> warm-up states (warm-up
  min(20000, n/5), state run stops 1000 later) -> timing_jobs `--set new` -> plot_timing.
- plot_headline: TripClick in the third row, MedRAG-Zipf in a fourth row (legend beside it);
  plot_timing: two more panels.
- Quiet timing batches now also suspend our `llama-server` and `curl` processes (run_timing_batch2.sh),
  not only python: both would disturb latency (CPU; EBS / network).

### 9aa. Quiet-window budget (11-14, Jialin away ~24 h)

- The quiet latency probe's batched 5,000-query recall per (search list, beam width) dominated the
  quiet window (~2.5 min per cell on SPACEV, far more at TriviaQA's search lists; projected 16-25 h
  with everything suspended). Removed (`latency_probe.py --n-recall 0` default): workload_recall.py
  already measures recall over each workload's requests for every probed (L, W) without a quiet
  machine, and select_backend.py uses it (the probe's batched recall only as a fallback for old
  probes). The re-run probe of all 7 indexes took 56 min (SPACEV: 3 min instead of ~100). TriviaQA
  candidates are base and aker_tqa only (pq768 / r128 side studies were killed for memory, never built).
- TIMING4 (main timing batch): 112 jobs x ~26 min (timing3b: 12 jobs in 5 h 10 min) ~ 48 h with
  everything else suspended. Same total either way, so it waits behind a gate (`/tmp/mx/TIMING_GATE`,
  checked by run_timing_batch2.sh; `/tmp/mx/timing_gate.sh`) until both new workloads' headline runs
  and DiskANN builds are done, or 14:00 UTC 11-14. The final set is reordered (same 112 jobs):
  every workload's 1% / 10% main cells first, then the papers' settings, then 0.3% / 3%.
- Open question for Jialin: whether CPU-only background jobs pinned away from the timing core
  disturb EBS-bound DiskANN latency measurably; if not, the quiet window need not suspend them.
- AIO contexts (11-14): DiskANN's reader calls io_setup(1024) per search thread at index open
  (measured: +2048 in /proc/sys/fs/aio-nr for 2 threads); aio-max-nr is 65536 and cannot be raised
  (no root), i.e. 64 thread contexts for all DiskANN processes together. A workload-recall job
  aborted with "io_setup() failed with EAGAIN" (3 x 24-thread recall jobs + 2 x 16-thread index
  studies). Fix: `diskann_aio.open_disk_index` (all five open sites) waits under a file lock until
  enough events are free; timing runs wait at most 30 min (paused jobs keep their contexts during a
  quiet batch). The ad-hoc index studies pq768 / r128 (killed for memory and retried for ~10 h; the
  probe had already run without them) were stopped; their directories are kept for a later rerun.
- MedRAG base build: memory-mapped fancy indexing faulted 4 KB pages at ~10 MB/s on the busy EBS
  volume; rewritten as one sequential read per chunk (~1 chunk / min), checked on chunk 0 (rows equal
  to the source, every row norm M, only MedRAG PMIDs). Coverage: 23,888,832 of MedRAG's 23,895,135
  PMIDs have an NCBI embedding (99.97%; 35.9M NCBI articles in total).
- MedRAG rewordings done (11-14 07:00): 10,000 unique verified rewordings; 24,475 candidates
  generated, 12,024 duplicates (49%), 1,169 rejected by the same-meaning check, 14 malformed.
  Problems found and fixed on the way: (1) Qwen switched partly to Chinese in 50 accepted
  rewordings (one also appended a "(Note: ...)"): candidates with characters of another script or
  model chatter are now rejected, and accepted ones re-filtered on resume; (2) a short, generic
  16-occurrence question stayed 2 short after 120 rounds: the "worded differently from these" list
  only started at 20 accepted versions; now stalled questions (no gain in a round) always get it,
  plus +0.1 temperature per stalled round (max 1.5); (3) long example lists exceeded a slot's
  1024-token context (HTTP 400, retried pointlessly): lists capped at ~1800 characters, 400s are
  failed candidates. MedCPT cosine between rewordings of the same question 0.951 on average,
  between different questions 0.443.
- TripClick: assembled 11-14 05:14 (1,523,878 docs, M 10.203), prep 05:31, headline runs done 06:21
  (292 simulator + 68 real-Aker jobs; 1.5M base, so fast).
- TripClick headline finding (11-14, for discussion): at 1%, ours (greedy_none) hits 4.6% at
  retention 0.932, LRU (our hit policy, K 10) 7.0% at 0.930, real Aker tuned ~11% (Aker paper default:
  19% at retention 0.70; its exact-repeat hits alone are 6.2%). The ceiling of the cached contents
  ("oracle" in online_summary: best hit rate the contents allow at the target) is 7.2% for greedy vs
  8.8% for LRU: the gap is population (greedy fills the cache once from the warm-up; a 3-month real
  log drifts; recency follows it), not the hit policy; calibration is also conservative here (0.93
  realized vs 0.9 target). Same pattern on the SPACEV real log at 10%. Not acted on: a cache
  population question (the project's next phase), Jialin's call.
- MedRAG-Zipf headline (11-14): with 500 questions, 0.3% of the 23.9M corpus (~143 top-500
  entries) already holds the working set, so the standard sizes are flat. Smaller sizes run with
  the 0.3% job set (budget substituted, `/tmp/mx/medrag_small.sh`); the panel shows 0.01 / 0.03 /
  0.1 / 0.3% (per-panel sizes in plot_headline LAYOUT). Hit rate / retention, seed 0:
  ours (greedy_score_rebuild) 0.441/0.918, 0.577/0.915, 0.636/0.928, 0.658/0.919;
  LRU K10 0.208/0.885, 0.407/0.884, 0.681/0.892, 0.879/0.900; LRU K20 0.229/0.922, 0.412/0.903,
  0.717/0.904, 0.919/0.910. Ours wins clearly under tight memory; LRU once the working set fits
  (greedy fills from the 2,000-request warm-up only: 39k of 239k vectors used at 1%).
  Real Aker paper default ~46% at retention 0.76 (every request is a unique rewording: no exact hits).
- Backend selection v2 (11-14 17:37; recall over each workload's requests, latency from the quiet
  probe, single query, EBS):
  esci1m 0.95: esci L 100 W 32, recall 0.9549, 8.6 ms (p99 11.7);
  spacev-aker-0.3 0.95: spacev-pq110 L 100 W 32, 0.9545, 10.2 ms; -0.6 / -0.99: spacev L 100 W 32,
  0.9509 / 0.9612, 9.9 ms (p99 14.2);
  spacev-real 0.9 (0.95 not reached by any setting): spacev-real-pq110 L 1000 W 32, 0.9091, 66.9 ms;
  triviaqa-ip-0.3 / 0.6 / 0.99 0.95: aker_tqa (degree 96, 512-byte PQ) L 1000 W 32, 0.953, 75.9 ms
  (p99 85.9); the 256-byte-PQ base index tops out at 0.947 (L 2000) on the 0.99 trace.
  Beam width 32 everywhere: v1 used beam 2 (TriviaQA 0.95 needed L 2000, ~740 ms).
- AIO lock (11-14): the first diskann_aio held the lock while waiting; a waiter suspended by a quiet
  batch would have blocked every timing run on the lock. Now the lock is held only to check and open.
- Reboot (11-14 ~19:56): /tmp (RAM) was wiped, with all of /tmp/mx (queue registry, learned peaks,
  growth factor, completion flags, job lists, ad-hoc chain scripts); no results were lost (all on
  disk). Recovery 20:19: /tmp/mx is now a symlink to ~/mx-state (on disk; after a reboot only the
  symlink must be recreated: `ln -sfn /home/jialind/mx-state /tmp/mx`, and the registry cleared);
  flags of finished steps restored by hand (INDEX_AKER, TIMING_PROBE_V2, REQ, HE, TSTATES_ESCI,
  TIMING3C, HEADLINE_tripclick-ip, HEADLINE_medrag-ip); chain scripts that only lived in /tmp/mx are
  now in the repository (run_medrag_small_states.sh). Relaunched: mx_watchdog4.sh, run_backend_redo.sh
  (probe and workload recall skip as done; selection unchanged; answers restarted from scratch:
  in progress at the reboot), run_timing_new_workload.sh for both new workloads,
  run_medrag_small_states.sh. Learned memory peaks start again from the priors.
- Plot locations (11-14): plot_headline.py / plot_timing.py now write only the latest versions to
  `simulations/approximate_hits/results/latest/` (headline.png, timing.png, timing_table.md; not in
  git, as everything under results/ except summary/: the plots include TripClick-derived results).
  Old drafts moved to results/summary/old_drafts/; results/summary/ keeps the earlier curated figures.
- Drift diagnostics (11-14, Jialin's questions): the headline's "ours" is already the drift-handling
  form (greedy_score_rebuild: score-drift detection + rebuild); on TripClick 1% it rebuilt 8 times and
  still reached 4.2%. Queued (/tmp/mx/diag_drift.txt, ad-hoc priority, 80 runs, 1% and 10%, seeds 0
  and 1, TripClick and SPACEV real log): (a) the other detection / response forms on the natural
  order (greedy_all_rebuild, greedy_all_recal, greedy_retention_rebuild, greedy_hitrate_rebuild,
  greedy_score_recal); (b) ours and LRU on a random reordering of the same requests (stream random).

### 9ab. Why LRU beats ours (11-14 evening; decided: real logs always use the drift-handling form)

- Checked: the headline's "ours" is greedy_score_rebuild (drift detection + rebuild) on every
  workload (plot_headline.collect: first available of OURS_NAMES). "LRU" is LRU population with our
  calibrated hit test (lambda recalibrated every batch), so the comparison is greedy vs LRU
  POPULATION under the same guarantee.
- Not only real logs: LRU also wins at 10% on the stationary Zipf traces (SPACEV 0.6: 36.7 vs 24.1;
  TriviaQA 0.6: 71 vs 51); ours wins at small caches.
- Cause 1, candidate cap: the greedy cache picks entries only from the candidate share of the
  warm-up (20% of 20,000 requests = 4,000) and never admits new queries; at 10% it holds ~3,000
  entries of ~280 vectors (top-K + shell), LRU ~50,000 entries of K 20. On the SPACEV Zipf traces
  only 2-7% of post-warm-up requests repeat any earlier request, so LRU's lead there comes from
  approximate hits of its many entries. Rebuilds draw candidates from 20% of a window: even fewer
  entries (TripClick 10%: 764 after rebuilds vs 1,705 at warm-up), so rebuild HURTS at 10%
  (TripClick 13.8% vs 20.4% without detection).
- Cause 2, temporal locality on real logs: TripClick in random order, 1%: ours 10.1% vs LRU 4.4%
  (seeds 0 and 1 agree); natural order: ours 4.6% vs LRU 7.0%. No detection / response form fixes
  the natural order (score / retention / hitrate / all, recal / rebuild: 3.5-6.2%); the drift is
  gradual, the KS-type alarms fire, rebuilds have small candidate pools. SPACEV real log: ours wins
  at 1% (4.5 vs 3.2-3.7), LRU at 10% (10.2 vs 7.7: cause 1).
- Test of cause 1 running: --warm-split 0.6 0.2 0.1 0.1 and 0.4 0.3 0.15 0.15 (12k / 8k candidates),
  SPACEV 0.6, TriviaQA 0.6, TripClick, 1% and 10% (/tmp/mx/warm_split.txt). TripClick 1% unchanged
  (4.5-5.0%: cap not binding at 1%, 826 entries).
- Timing harness: ours is static or `--ours-recal` (recalibration + audits); detection and cache
  rebuilds are NOT implemented there. Main timing batch held (/tmp/mx/TIMING_GATE) until decided.

### 9ac. Decisions and plan (11-14 night, Jialin away ~24 h)

- Variants kept side by side; which one is called "ours" is decided later from multiple scenarios:
  ours-static, ours-drift (detection + shadow rebuild), LRU (our warm-up cache + calibrated hit test,
  every miss admitted at fixed K, least-recently-used eviction), and a new continuous variant
  (below). Real logs (TripClick, SPACEV real log) always use the drift-handling forms.
- Timing: drift handling must run its own logic live in the timing harness (no replay of
  simulator installs). Time both ours-static and ours-drift (plus LRU, Aker, QVCache).
- Periodic re-selection was tried before (5p-5q) and dropped; its problem was partly labeled rebuild
  data (hits labeled only when audited, up-weighted 1/a: unbiased but noisy exactly on the hot
  queries; and only labeled queries can be candidate centers). New continuous variant to prototype:
  every miss (fully labeled by the backend) becomes a candidate entry; entry size chosen by
  estimated utility (hits per vector); eviction by lowest estimated utility AND by LRU (Jialin
  expects these to be similar; test both).
- Plan: (1) drift handling in eval_timing_online; (2) release TIMING_GATE after checking it;
  (3) prototype the continuous variant in online_sim; (4) analyze capped-K (/tmp/mx/maxk.txt) and
  warm-up-length (/tmp/mx/warmup_len.txt) tests; check every 30 min.
- Entry size (11-14, `--max-ball-k 20`: the same greedy selection limited to top-20 entries, LRU's
  size; seed 0, hit % / retention): ESCI 10% 30.3 -> 16.4 (LRU 25.6), ESCI 1% 10.1 -> 7.2 (LRU 0.4),
  SPACEV Zipf 0.6 1% 13.2 -> 10.6 (LRU 11.6), Zipf 0.99 1% 50.6 -> 46.4 (LRU 47.3), TripClick 1%
  4.6 -> 4.3 (LRU 5.2). Larger entries are genuinely better when chosen by utility; on ESCI (no
  repeats) capped greedy falls below LRU at 10%. Greedy's average entry grows with the budget
  (SPACEV Zipf 0.6: ~30 / 76 / 134 / 275 distinct vectors per entry at 0.3 / 1 / 3 / 10%).
- Simulator fixes for the timing harness (11-14): `make_parser()` (parser shared with the harness),
  `Sim.cache_state()` (hit-path state of the current cache), `Sim.last_hold` (held-out calibration
  requests of an installed rebuild); with --backend-file the exact truth falls back to the prep's
  top500.npy when topU1000.npy is missing (TriviaQA / TripClick / MedRAG warm-up states would
  otherwise have crashed).
- Timing harness `--method ours_drift` (class OursDrift): the simulator's controller (Sim: label
  window, detect, alarm reset, post-alarm shadow-rebuild checks, calibrate) runs live on the run's
  own scores (harness hit path) and labels (backend answers); installs build a new hit path
  (background work), held-out calibration requests re-scored on it. Validation: tc-test (TripClick
  alias, exact answers as "backend"), harness vs simulator on the same 40k requests
  (results/timing_test/tc-test).
- ours_drift validation (tc-test, 1%, 40k requests after the 20k warm-up): simulator events
  alarm 21k, 24k, rebuild 29k, then alarms / rebuilds at 31k-60k; harness: identical through the
  29k rebuild, then a few thousand requests apart (the harness scores through its HNSW hit path,
  the simulator exactly), 4 rebuilds installed, 22.5 s background. Hit rate 2.8% vs 3.6%, retention
  0.927 vs 0.917. The harness needs scikit-learn in .venv-ann (installed 1.6.1, numpy 1.25.0 kept).
- Timing sets (11-14): final = 134 jobs: ours (static) and ours_drift on every stationary workload,
  ours_drift only on the SPACEV real log; "new" set: TripClick ours_drift only, MedRAG both.
  plot_timing shows ours_drift as its own line. TIMING_GATE removed (22:30): the batch starts when
  the backend answers and warm-up states are done.
- Continuous variant `online_sim --method ogreedy [--og-evict utility|lru] [--og-span N]` (11-14):
  after each batch, greedy (the offline rule: marginal hits per new vector, from the current
  cache) over candidates = this batch's misses at every K plus current entries grown to larger K;
  scoring = misses of the previous --og-span requests (default cal-span 20k; fully labeled);
  victims by credit per vector (served answer vectors supplied / k over the span) only while below
  the candidate's gain per vector, or LRU whenever room is needed; both windows seeded from the
  warm-up requests; memory trimmed to M. Bugs found in smoke tests and fixed: scoring candidates on
  their own misses (every miss served itself: ~90% of entries replaced per batch); unseeded credit
  window (582 of 826 warm-up entries evicted in batch one). TripClick 1%, requests 20k-30k:
  ogreedy-utility 8.0% / 0.907 vs ours 7.0% / 0.902, LRU K10 7.1% / 0.894. Full runs queued:
  /tmp/mx/ogreedy.txt (6 workloads x 1%, 10% x 2 eviction rules, seed 0).
- ogreedy fixes (11-14 23:24-23:38), after the first full results (now *_v1.json, void): (1) the
  exchange compared ORACLE gains of candidates with SERVED credits of incumbents: gains now scaled
  by rho = served / oracle hits over the span (warm-up in-sample rates as a prior); (2) main cause
  of collapse: an eviction was assumed to free all its members, but shared vectors stay cached, so
  greedy overspent and the final memory trim removed old entries with no value test (TripClick 10%:
  1,763 -> 483 entries, retention 0.81 in a window); now freed = vectors held by no other remaining
  entry, and victims are valued per such vector. Check, requests 20k-45k, 10%: TripClick 27.3% /
  0.913 (ours 22.5 / 0.907, LRU K20 23.5 / 0.918, v1 22.3 / 0.858); ESCI 32.4% / 0.906 (ours 30.2 /
  0.910, LRU K20 26.1 / 0.906, v1 26.6 / 0.874). Full set rerun with the fixed code.
- ogreedy (fixed) results, seed 0, hit % / retention / 5k windows below T (of 16), vs ours
  (greedy_none), ours-drift, LRU best: ESCI 1% 10.8/0.920/0 (ours 10.1, LRU 0.4); ESCI 10%
  32.8/0.914/0 (30.3, 25.6); SPACEV Zipf 0.6 1% 17.0/0.919/1 (13.2, 11.6); Zipf 0.99 1%
  53.3/0.918/0 (50.6, 47.3); SPACEV real 1% 5.7/0.935/1 (4.5, 3.2); TripClick 1% 7.4/0.927/5 (4.6,
  7.0); TripClick 10% 28.4/0.909/5 (20.4, 24.3); TriviaQA Zipf 0.6 1% 25.1/0.914/1 (19.8, 17.5).
  Utility eviction >> LRU eviction (og-lru: Zipf 0.99 1% 33.0, Zipf 0.6 1% 6.2). Caveat: windows
  below T on TripClick (5 of 16; LRU 1-4): the cache changes every batch, calibration lags.
- Warm-up length (11-15, /tmp/mx/warmup_len.txt; seed 0; hit % / retention on requests after 60k
  for every run): SPACEV Zipf 0.6 10%: ours 24.2 / 33.7 / 40.5 at warm-up 20k / 40k / 60k, LRU K20
  39.1 / 39.4 / 40.0; Zipf 0.6 3%: ours 18.0 / 23.9 / 30.1, LRU 19.4 / 24.0 / 27.5; Zipf 0.99 3%:
  54.7 / 60.0 / 62.3 vs 54.3 / 56.9 / 60.2; TriviaQA 0.6 3%: 31.8 / 43.5 / 51.9 vs 30.2 / 43.2 / 49.5.
  A larger candidate pool closes the large-cache gap on stationary traces (ours >= LRU at 60k);
  LRU also gains on some traces because it starts from the same greedy warm-up cache. The
  continuous variant gets a growing pool without a longer warm-up.
- ogreedy 10% (fixed): SPACEV Zipf 0.6 31.1 (ours 24.5, LRU K20 36.9), Zipf 0.99 66.2 (LRU 67.1),
  SPACEV real 10.5 (LRU 10.2). At Zipf 0.6 10% ogreedy keeps ~3,500 entries of ~275 vectors, LRU
  ~12,900 of ~49: ogreedy admits a new query only with support from earlier misses (exact repeats
  are 6.6% there), so new queries lose to growing entries; LRU bets on recency. Residual gap =
  evidence-driven vs recency-driven admission.
- Windows below T on TripClick are shared with LRU (both dip to ~0.88 at windows 10-12, requests
  ~70k-80k, then recover): a property of that log period. --og-split with maintained labels (fresh
  calibration against the current cache, build / calibration halves) did not reduce them (1%: 7 of
  16 windows below, 9.5% / 0.907); kept as an option, not the default.

- Headline values, continuous greedy (og, ogreedy_utility, seeds 0 and 1) vs ours vs LRU (hit %, retention), 11-15 03:00:
  spacev-aker-0.3 0.3%: og (3.4, 0.938) ours (3.0, 0.92) lru (1.6, 0.931) | 1%: og (6.1, 0.926) ours (4.4, 0.932) lru (4.6, 0.926) | 3%: og (10.1, 0.918) ours (8.0, 0.921) lru (10.6, 0.917) | 10%: og (19.7, 0.914) ours (14.3, 0.92) lru (27.3, 0.917)
  spacev-aker-0.6 0.3%: og (12.7, 0.92) ours (10.3, 0.922) lru (5.9, 0.925) | 1%: og (17.0, 0.919) ours (13.3, 0.92) lru (11.6, 0.919) | 3%: og (21.3, 0.92) ours (17.3, 0.919) lru (18.7, 0.917) | 10%: og (31.3, 0.919) ours (24.1, 0.917) lru (36.7, 0.914)
  spacev-aker-0.99 0.3%: og (49.6, 0.915) ours (47.4, 0.913) lru (42.6, 0.911) | 1%: og (53.7, 0.914) ours (50.5, 0.913) lru (47.3, 0.914) | 3%: og (58.2, 0.911) ours (54.7, 0.913) lru (53.9, 0.916) | 10%: og (66.1, 0.911) ours (59.6, 0.912) lru (67.0, 0.915)
  triviaqa-ip-0.3 0.3%: og (2.5, 0.922) ours (2.9, 0.916) lru (1.0, 0.931) | 1%: og (12.5, 0.917) ours (8.7, 0.916) lru (6.1, 0.908) | 3%: og (26.3, 0.918) ours (19.4, 0.914) lru (19.0, 0.916) | 10%: og (60.1, 0.909) ours (38.2, 0.909) lru (62.9, 0.908)
  triviaqa-ip-0.6 0.3%: og (10.7, 0.918) ours (11.7, 0.912) lru (6.2, 0.91) | 1%: og (25.5, 0.913) ours (19.9, 0.912) lru (17.7, 0.91) | 3%: og (38.7, 0.915) ours (31.9, 0.913) lru (29.9, 0.917) | 10%: og (68.4, 0.91) ours (50.8, 0.909) lru (71.3, 0.907)
  triviaqa-ip-0.99 0.3%: og (58.7, 0.913) ours (57.0, 0.912) lru (53.2, 0.912) | 1%: og (66.5, 0.912) ours (64.0, 0.914) lru (62.4, 0.911) | 3%: og (75.0, 0.91) ours (71.8, 0.909) lru (71.1, 0.911) | 10%: og (91.4, 0.913) ours (81.3, 0.908) lru (91.3, 0.913)
  esci1m 0.3%: og (6.8, 0.926) ours (6.1, 0.924) lru (0.1, 0.885) | 1%: og (10.9, 0.919) ours (9.8, 0.917) lru (0.1, 0.851) | 3%: og (17.4, 0.916) ours (15.4, 0.914) lru (10.0, 0.919) | 10%: og (33.0, 0.913) ours (29.8, 0.912) lru (25.0, 0.91)
  spacev-real 0.3%: og (4.5, 0.947) ours None lru None | 1%: og (5.7, 0.936) ours (4.6, 0.939) lru (3.5, 0.95) | 3%: og (7.4, 0.924) ours None lru None | 10%: og (10.5, 0.907) ours (7.6, 0.918) lru (10.1, 0.925)
  tripclick-ip 0.3%: og (5.4, 0.934) ours (2.2, 0.937) lru (2.4, 0.931) | 1%: og (7.9, 0.927) ours (4.4, 0.921) lru (7.0, 0.929) | 3%: og (13.8, 0.919) ours (7.3, 0.92) lru (12.4, 0.919) | 10%: og (28.1, 0.913) ours (14.1, 0.911) lru (25.0, 0.916)
  medrag-ip 0.01%: og (50.4, 0.918) ours (44.3, 0.921) lru (23.7, 0.918) | 0.03%: og (67.3, 0.913) ours (57.2, 0.919) lru (40.9, 0.904) | 0.1%: og (73.8, 0.912) ours (59.7, 0.923) lru (70.5, 0.909) | 0.3%: og (88.9, 0.919) ours (66.4, 0.919) lru (91.8, 0.91)
  og >= ours at every cell except TriviaQA 0.3% (0.3 / 0.6 Zipf: 2.5 vs 2.9, 10.7 vs 11.7); og >= LRU except the 10% cells of SPACEV 0.3 / 0.6 (19.7 vs 27.3, 31.3 vs 36.7), TriviaQA 0.3 / 0.6 10% (60.1 vs 62.9, 68.4 vs 71.3), SPACEV 0.99 10% (66.1 vs 67.0), SPACEV 0.3 3% (10.1 vs 10.6), MedRAG 0.3% (88.9 vs 91.8). Every og cell meets T.
- `--og-fresh` (11-15): fresh misses (no evidence of their own) admitted at the K whose hindsight
  value on this run is best (earlier half of the miss window, as if admitted at each K, scored on
  the later half; mean gain over the current cache per new vector, x rho, scaled to the credit
  span), while it beats the victims' credit per vector. Aim: the large-cache cells where LRU's
  recency bet wins (SPACEV / TriviaQA Zipf 0.3-0.6, 10%) without hurting ESCI. Test queued
  (/tmp/mx/og_fresh.txt, 12 cells, seed 0).
- `--og-fresh` result (seed 0): +1 point at 10% (SPACEV 0.3 20.9 vs 19.8, 0.6 32.1 vs 31.1, 0.99
  66.2 vs 66.2; LRU 26.7 / 36.9 / 67.1), but -6 at SPACEV 0.6 1% (10.9 vs 17.0) and -1 at SPACEV
  0.3 3%; neutral on ESCI / TripClick / MedRAG. Not adopted (kept as an option). The remaining
  large-cache gap on SPACEV / TriviaQA Zipf 0.3-0.6 is open.
- --og-span (window of misses and credits): 40k >= 20k (default) > 10k slightly (SPACEV 0.6 1%:
  17.4 / 17.0 / 14.3; TripClick 1%: 8.8 / 7.4 / 6.4; ESCI 10%: 33.0 / 32.8 / 33.2): not sensitive.
- Honesty fix (11-15 04:30): rho used every request's retention (log ret_hist), including
  unaudited hits, which a real cache cannot know. Now: served hits (known) / oracle hit rate
  estimated from the labeled requests of the span (misses weight 1, audited hits 1 / rate) x
  requests. Other ogreedy inputs checked (credits: served answers; scoring: misses' backend
  answers; candidates: backend top-R, counted; seed: warm-up, all labeled). TripClick 10% check:
  27.3% / 0.912 (rho 0.84-0.86) vs 27.3% / 0.913 before. All ogreedy headline runs (80) and the
  LRU-eviction runs (12) rerun with the fixed code; previous outputs archived as *_v3.json. The
  og-span / og-fresh / og-split diagnostics above used the previous rho (difference ~0).
- Timing harness `--method ours_online` (class OursOnline, on LRU's incremental cache): the
  simulator's ogreedy controller runs live on the run's scores, labels and served answers; after
  each batch the harness cache is changed to the controller's entry set (balls / shells from the
  backend's top-500 lists, the simulator's sampling draw); controller work + changes = background.
  Check (tc-test, 1%, 40k requests): harness 6.9% / 0.943 vs simulator 8.5% / 0.923 (the same
  harness-vs-simulator gap as ours_drift: approximate in-cache search, slightly lower scores);
  13,874 entry changes; 189 s background (~4.7 s per 1,000-request batch).
- Timing final set now 158 jobs (+24 ours_online, every main cell incl. the SPACEV real log);
  TripClick / MedRAG "new" set likewise. At ~26 min per job: ~68 h of quiet machine.
- Headline values after the honesty fix (rho from labeled requests only; 11-15 06:00), og vs ours vs LRU (hit %, retention); every cell moved <= 0.6 points:
  spacev-aker-0.3 0.3%: og (3.3, 0.94) ours (3.0, 0.92) lru (1.6, 0.931) | 1%: og (6.0, 0.926) ours (4.4, 0.932) lru (4.6, 0.926) | 3%: og (10.0, 0.92) ours (8.0, 0.921) lru (10.6, 0.917) | 10%: og (19.7, 0.913) ours (14.3, 0.92) lru (27.3, 0.917)
  spacev-aker-0.6 0.3%: og (12.7, 0.921) ours (10.3, 0.922) lru (5.9, 0.925) | 1%: og (17.1, 0.918) ours (13.3, 0.92) lru (11.6, 0.919) | 3%: og (21.2, 0.922) ours (17.3, 0.919) lru (18.7, 0.917) | 10%: og (31.5, 0.917) ours (24.1, 0.917) lru (36.7, 0.914)
  spacev-aker-0.99 0.3%: og (49.5, 0.916) ours (47.4, 0.913) lru (42.6, 0.911) | 1%: og (53.7, 0.913) ours (50.5, 0.913) lru (47.3, 0.914) | 3%: og (58.2, 0.91) ours (54.7, 0.913) lru (53.9, 0.916) | 10%: og (66.1, 0.911) ours (59.6, 0.912) lru (67.0, 0.915)
  triviaqa-ip-0.3 0.3%: og (2.6, 0.921) ours (2.9, 0.916) lru (1.0, 0.931) | 1%: og (12.4, 0.917) ours (8.7, 0.916) lru (6.1, 0.908) | 3%: og (26.5, 0.917) ours (19.4, 0.914) lru (19.0, 0.916) | 10%: og (59.9, 0.91) ours (38.2, 0.909) lru (62.9, 0.908)
  triviaqa-ip-0.6 0.3%: og (10.8, 0.918) ours (11.7, 0.912) lru (6.2, 0.91) | 1%: og (25.4, 0.914) ours (19.9, 0.912) lru (17.7, 0.91) | 3%: og (38.8, 0.915) ours (31.9, 0.913) lru (29.9, 0.917) | 10%: og (68.4, 0.91) ours (50.8, 0.909) lru (71.3, 0.907)
  triviaqa-ip-0.99 0.3%: og (58.8, 0.913) ours (57.0, 0.912) lru (53.2, 0.912) | 1%: og (66.6, 0.911) ours (64.0, 0.914) lru (62.4, 0.911) | 3%: og (75.0, 0.91) ours (71.8, 0.909) lru (71.1, 0.911) | 10%: og (91.4, 0.913) ours (81.3, 0.908) lru (91.3, 0.913)
  esci1m 0.3%: og (6.8, 0.926) ours (6.1, 0.924) lru (0.1, 0.885) | 1%: og (10.8, 0.919) ours (9.8, 0.917) lru (0.1, 0.851) | 3%: og (17.5, 0.916) ours (15.4, 0.914) lru (10.0, 0.919) | 10%: og (33.1, 0.914) ours (29.8, 0.912) lru (25.0, 0.91)
  spacev-real 0.3%: og (4.5, 0.946) ours None lru None | 1%: og (5.7, 0.937) ours (4.6, 0.939) lru (3.5, 0.95) | 3%: og (7.4, 0.922) ours None lru None | 10%: og (10.5, 0.909) ours (7.6, 0.918) lru (10.1, 0.925)
  tripclick-ip 0.3%: og (5.2, 0.937) ours (2.2, 0.937) lru (2.4, 0.931) | 1%: og (8.5, 0.925) ours (4.4, 0.921) lru (7.0, 0.929) | 3%: og (13.2, 0.918) ours (7.3, 0.92) lru (12.4, 0.919) | 10%: og (28.4, 0.911) ours (14.1, 0.911) lru (25.0, 0.916)
  medrag-ip 0.01%: og (50.3, 0.918) ours (44.3, 0.921) lru (23.7, 0.918) | 0.03%: og (67.3, 0.913) ours (57.2, 0.919) lru (40.9, 0.904) | 0.1%: og (73.9, 0.912) ours (59.7, 0.923) lru (70.5, 0.909) | 0.3%: og (88.9, 0.919) ours (66.4, 0.919) lru (91.8, 0.91)
- Entry-size cap on ogreedy at 10% (seed 0; --max-ball-k): SPACEV 0.3 19.8 (3,509 entries) ->
  K<=100 34.6 (12,781) / K<=50 27.9 (14,075), LRU K20 26.7; SPACEV 0.6 31.2 (3,558) -> 42.2
  (12,422) / 36.0 (11,718), LRU 36.9. At large budgets greedy over-chooses top-200 / 500 entries
  whose coverage of past misses does not recur (a generalization bias of its objective); at small
  budgets large entries win (the top-20 cap test). Self-adaptive version `--og-kcal`: per entry
  size K, phi_K = realized hits (credits) / predicted gain of this run's own entries of size K
  (>= 20 entries, >= half a span old; else 1); candidates ranked by phi_K x gain per vector; no
  cap. Test queued (/tmp/mx/og_kcal.txt, 12 cells, seed 0).
- --og-kcal v1 had no effect (phi = 1 everywhere): credits count every served vector an entry
  holds, shared ones included, while predictions are marginal hits; realized >= predicted. v2:
  temporal cross-check in marginal hits on both sides (span misses split earlier / later; this
  batch's candidates scored against each half over the current cache; per K, among the top fifth
  by earlier-half gain per vector, phi_K = later gain / earlier gain), every 5 batches. v1 outputs
  archived (*_kcal_v1.json); v2 rerun on the same 12 cells.
- --og-kcal v2 results (seed 0): phi falls with SMALLER K (TripClick 1%: K10 0.19 ... K500 0.48;
  SPACEV 0.99 1%: 0.24 ... 0.51): small entries' predicted gains hold up worse on later requests,
  so it does not steer toward small entries. Hits: TripClick 1% 9.8 (vs 7.3), 10% 29.0 (28.5);
  SPACEV 0.3 / 0.6 10% 20.2 / 31.2 (19.8 / 31.2; cap K<=100: 34.6 / 42.2); TriviaQA 0.6 10%
  69.3 (68.9; cap 69.0); others within 0.3. Not adopted as default.
- Open question (for Jialin): at 10% on SPACEV Zipf 0.3 / 0.6, capping ogreedy's entries at K 100
  gives 34.6 / 42.2 vs LRU 26.7 / 36.9 (uncapped 19.8 / 31.2), but the cap is a hand setting and
  does not help TriviaQA 0.6. The per-size cross-check says big entries generalize at least as
  well candidate by candidate, so the cap's gain is likely a dynamic (myopia) effect: growing an
  existing entry wins each batch's comparison but uses memory that many later-arriving queries
  would use better. Not resolved.

### 9ad. Timing cost cuts (11-15, approved by Jialin)

- Breakdown of the v3 timing jobs: ~half of each job re-timed DiskANN alone and DiskANN with equal
  memory (~8 min per job at 62 ms), identical for every technique of a workload. Re-estimate of the
  158-job batch with the tuned backends: ~24 h (SPACEV / ESCI ~5 min per job, TriviaQA ~18, real
  log ~15), not 68.
- Cut 1: `eval_timing_backend.py` times DiskANN alone once per workload and DiskANN with equal
  memory once per cache size (node cache = memory of the static ours cache at that size; every
  technique has the same vector budget), on the same timed requests; technique runs use
  `eval_timing_online.py --no-backend-arms`; plot_timing fills backend / backend_mem / speedup
  from timing_v4/<prep>/backend_s<every>.json.
- Cut 2: 1 request in 50 timed (was 20); hit rates still over the whole stream. Check cells in the
  same batch: SPACEV 0.6 and TriviaQA 0.6, 1%, ours and LRU, also at 1 in 20 (with their own arms).
- Estimated batch: ~8-10 h. Proposed, not done (Jialin to decide): compose per-request latency from
  cache-only time + the request's DiskANN time measured once (technique runs would no longer hit the
  disk while timed); build the shared warm-up cache once instead of per technique; static ours'
  untimed decisions outside the quiet window.
- Batch status 11-15 21:30 (TIMING4, 172 jobs, started 16:38): the 10 DiskANN-arm jobs took ~3.4 h
  (~21 min each: the equal-memory arm opens DiskANN with 4 node-cache sizes); per-technique jobs
  (SPACEV): ours ~2-3 min, drift 2-6, continuous 5-14 (controller work 3-9 min is wall time, counted
  as background), LRU 6-11, Aker / QVCache 1-2. One cell ran "contended" (my own commands during the
  batch): oursdrift SPACEV 0.6 1%, moved to timing_v4/contended_old and rerun. Batch loop stopped
  at 21:30 after job ~29 (done jobs kept; the rest skip on restart) to edit the harness and run the
  composed-timing check on a quiet machine.
- Cost items (Jialin 11-15): (2) warm-up product shared: Sim.warm_product / restore_warm, the first
  drift / continuous run of a cell stores timing_v4/<prep>/warm_b<b>.pkl, later ones load it (the
  simulator's warm-up before replay took ~4 min per drift / online run; ~1.6 h over the batch).
  (3) static ours untimed pass outside the quiet window: dropped as small (ours jobs are 2-3 min;
  ~35 min in total) and superseded by item 1 if composed timing validates (then no technique run
  needs a quiet machine). (1) composed timing: `--compose-backend backend_s50_lat.npy`: timed
  requests use stored answers; a request that needs the backend (miss, audited hit) is charged that
  request's DiskANN-alone latency measured by eval_timing_backend.py. Check on a quiet machine:
  run_compose_check.sh (SPACEV Zipf 0.99 1% and TriviaQA Zipf 0.6 1%, six techniques): live vs
  composed sequential vs composed with 12 jobs at once, 4 cores each.
- Composed-timing check (11-16 00:00-00:25; SPACEV Zipf 0.99 and TriviaQA Zipf 0.6, 1%, six techniques):
  hit rates equal (<= 0.8 points; FAISS's multi-threaded HNSW build is nondeterministic). TriviaQA:
  composed = live within 1.5% (sequential and 12 jobs at once, 4 cores each: parallel does not
  distort). SPACEV: composed 8-9.6 ms vs live 5.3-6.4 ms. Cause: DiskANN's latency depends on request
  spacing. Misses inside technique runs cost ~10 ms, the DiskANN-alone arm (back to back) ~15 ms:
  the volume sustains ~15,000 reads/s (measured 14,996/s during the answer jobs); a SPACEV search
  (~200 reads) run back to back is read-budget bound (~13 ms), spread out it takes ~10 ms.
  Gap test (spacev-aker-0.99, 300 requests): gap 0 -> 14.7 ms; 10 / 30 / 60 / 120 ms -> 9.9 / 9.7 /
  9.8 / 9.8 ms. ESCI 0 / 10 / 60 ms: 10.7 / 9.0 / 8.8. TriviaQA 0.6 (~1,000 reads per search): 77.6 /
  75.0 / 76.4 (no effect: read-bound either way). The original method had the same mismatch (alone
  back to back, techniques' misses spread out), so SPACEV / ESCI speedups were inflated.
  Fix: eval_timing_backend.py --gap-ms (default 10; also time_backend(gap_ms)); arms to be rerun.
- Composed timing validated (11-16 01:30). DiskANN-alone arm with the 10 ms gap, SPACEV Zipf 0.99:
  alone 14.72 -> 9.53 ms; equal memory (0.3 / 1 / 3 / 10%) 8.96 / 8.19 / 7.22 / 6.13 -> 6.74 / 6.54 /
  6.57 / 6.40 (old files kept as backend_s50_nogap.json). SPACEV 0.99 1%, composed vs live mean ms:
  Aker 6.26 / 6.10, LRU 6.17 / 6.16, ours 5.46 / 5.31, ours-drift 5.97 / 5.76, continuous 5.70 / 5.59,
  QVCache 6.23 / 6.40 (<= 3.7%); six at once = one at a time within 2%; p99 within 7%. TriviaQA 0.6
  (no gap needed): LRU 66.5 / 66.8 (par / live), continuous 61.9 / 62.4. The TriviaQA live runs are
  the real results of those cells.
- Estimate for the rest of the final set (138 technique jobs): live ~12 h sequential vs composed ~2-3 h
  at 12 at once; baselines (~3.4 h) needed either way. Launched 01:33: run_timing_final.sh = the 7
  remaining DiskANN-only arms (quiet, sequential) then run_timing_composed.py (12 workers x 4 cores).
  Final timing results mix 29 live runs (SPACEV 0.99, 0.6 partly, TriviaQA 0.6 1%) and composed
  runs; the JSON args record --compose-backend. Then: TripClick / MedRAG chains (their waiting
  scripts need TIMING4.DONE, which the killed live batch never set).
- In-cache search list found too small on TriviaQA at 10% (11-16 06:30): the partial timing plot showed
  static ours at TriviaQA Zipf 0.99 10% with hit rate 0.052 (simulator 0.81; 1%: 0.598 vs 0.64),
  speedup 1.02. Cause: the harness's hit path searches the cached vectors with HNSW (search list 256, fixed
  from the SPACEV check, 0.975-0.985 neighbor recall); on TriviaQA 10% (769-d, 696k vectors) recall@10
  is 0.733 / 0.901 / 0.945 / 0.971 / 0.988 at search list 64 / 256 / 512 / 1024 / 2048 (1.6 / 2.9 / 5.5
  / 11 ms per lookup beyond 256: 1.6, 2.9, 5.5, 11.1 ms). The hit score is built from the neighbors'
  distances, so approximate neighbors mean low scores and a strict cutoff. With 1024: hit rate 0.736.
  Fix (all HNSW techniques: ours, drift, continuous, LRU): `tune_ef` picks, per cache, the smallest
  of 256 / 512 / ... / 4096 with neighbor recall@10 >= 0.975 on 300 warm-up requests (brute force over
  the cached vectors, untimed); recorded as in_cache_search in every result. All ours / drift /
  continuous / LRU results so far (96 files, composed and the live ones) archived in
  timing_v4/ef256_old/ and rerun composed (Aker and QVCache use their own libraries: unchanged).
  The larger search list costs real hit-path time at big caches; that is part of the result.
- Partial timing plot (before the fix): SPACEV Zipf 0.3 / 0.6 speedups ~0.9-1.1 (DiskANN with the
  cache's memory as a node cache is 1.4-1.5 there, the gap-baseline effect plus memory), SPACEV 0.99
  1.3-1.8, ESCI ~1.0 (LRU 0.74-0.87: its cache updates), TriviaQA 0.99 1% up to 2.4.
- Final-batch bug (11-16 10:40): `tune_ef` ran after the harness's warm-up calibration on the
  serving path, so the cutoff was calibrated with the 256-search-list path and then used with the
  larger list: static ours at TriviaQA 0.6 10% served 7.7% at retention 0.995 (simulator 52.6% at
  0.9; cutoff 3.2 vs the simulator's 1.06). Fixed (tuning in the constructors, before the warm-up
  calibration); the 41 runs with a list > 256 (ours, drift, continuous, LRU; mostly 10% cells, TriviaQA
  1%) archived in timing_v4/ef_order_old and rerun. Runs with list 256 unaffected.
  Search lists chosen: 256 (53 runs), 512 (25), 1024 (1), 2048 (14), 4096 (1).
- Reading the first complete timing plot (before this rerun): (a) at SPACEV Zipf 0.3 / 0.6 and ESCI
  every technique is at 0.8-1.1x: hit rates are 10-30%, a hit still costs 1-3 ms of in-cache search
  vs a 9.6 ms DiskANN search, and a miss pays the failed lookup too (10.5-11.7 ms vs 9.6); (b) DiskANN
  with the cache's memory as a node cache (grey dashed) is 1.4-1.5x there, above every technique;
  (c) SPACEV Zipf 0.99: 1.5-1.8x; TriviaQA 0.99 1%: 2.0-2.4x.

### 9ae. Plan after the first full timing plot (11-16 evening, decided with Jialin)

Findings that drive it: (1) speedup = baseline / mean latency: ours' hit path (HNSW kNN over cached
vectors) costs every request 1-24 ms (kNN 0.5-23.4 ms; balls 0.1-0.2; intruder count 0.1-0.75; model
~0) against 9.6 ms (SPACEV) / 74 ms (TriviaQA) per DiskANN search, so ours needs hit rate > lookup /
DiskANN time (12% SPACEV, 32% TriviaQA 10%) just to break even; Aker / QVCache pay ~0. (2) The hit
decision depends only on the kNN radius (and the intruder count measured against it), not on the
neighbor identities; the served answer (the 10 points) matters for retention. SPACEV 0.6 10%, 1,500
requests: search list 64 (neighbor recall 0.908) gives median radius ratio 1.0000 (p95 1.015), 99.8%
equal decisions, retention of exact-hit queries 0.911 vs 0.924; list 128 (0.949): 100%, 0.922.
The 0.975 neighbor-recall target was my choice from an earlier SPACEV check, not derived.
(3) DiskANN with a node cache: SPACEV 9.6 -> 6.8 ms with 54k nodes (0.5%), 6.5 at 1.6M; ESCI 8.5 ->
6.5 (3.2k nodes) ... 5.4 (100k); TriviaQA 74.0 -> 73.6 / 72.7 (100k / 884k): hub nodes at the search
start, independent of repetition (same at every Zipf); latency-bound (few sequential reads) vs
read-rate-bound (TriviaQA ~1,000 reads at ~15k reads/s).

Decided:
a. Audits: not charged to request latency; audit time measured separately and reported separately
   (affects drift / continuous / LRU runs).
b. Stage-1 hit path tuned ADAPTIVELY: starting from a high-accuracy setting, lower the search list and
   then the vector precision (fp32 -> 8-bit -> coarser / PQ) while the radius estimates and hit decisions
   stay accurate against exact search on warm-up requests; keep the last accurate setting. Stage 2:
   served answer for hits only, with its own accuracy rule (retention of hits).
c. Node cache: a fixed small node cache in the DiskANN baseline and under every technique's misses
   (outside the cache budget), size chosen per workload by doubling until the latency gain saturates.
d. The 100 ms rule is no longer a hard limit. BACKEND-RECALL SENSITIVITY TEST (must not be dropped):
   rerun the comparison with DiskANN tuned to higher recall (0.97 / 0.98 / 0.99) at least on SPACEV and
   ESCI (probe: SPACEV 0.95 / 0.97 / 0.98 = 10 / 20 / 29 ms; ESCI 9 / 14 ms; TriviaQA 76 / 186 ms),
   reported as a sensitivity result; main numbers stay at 0.95.
e. No heuristic pre-check with new parameters. Exact bound (skip the kNN when the best-case score
   cannot reach the cutoff) kept as an idea, only if it prunes enough (to be measured).

### 9af. Phase 2 results and the v5 timing harness (11-16 night)

Goal reminder (Jialin): every caching technique was losing to "DiskANN + node cache" in some cells,
which looks bad. v5 therefore gives every arm (baseline, every technique's misses) a base node cache
outside the cache budget; speedups shrink, and can later be raised again by tuning the backend to a
higher recall (dearer misses). The two-stage hit path removes ours' main handicap: the every-request
kNN over the cache.

Node cache per workload (node_cache_curve.py, quiet, 10 ms gap, 400 requests; start 0.03% of points,
x4 per step, stop at the first step that gains < 3%; backends/<prep>/node_cache.json):
| workload | none | chosen | ms |
| SPACEV Zipf 0.6 | 9.69 | 11,996 nodes (4.3 MB) | 7.08 |
| SPACEV Zipf 0.3 (pq110 index) | 9.99 | 11,996 | 7.59 |
| SPACEV real log | 64.98 | 2,999 | 61.35 |
| ESCI | 8.67 | 300 | 6.82 |
| TriviaQA 0.6 | 75.42 | 2,999 (10 MB) | 73.79 |
| TripClick | 11.39 | 1,828 | 9.42 |
| MedRAG | 13.24 | 7,166 (24 MB) | 11.48 |
SPACEV Zipf 0.99 uses 0.6's choice, TriviaQA 0.3 / 0.99 use 0.6's (same index; eval_timing.base_node_cache).
OPEN: the stop rule ends at the first flat step. SPACEV real gains little from 3k to 12k nodes but a
lot further out (57.7 ms at 54k, 45.6 ms at 10% of points, doc 9ae), so its base cache is probably
too small: the equal-memory arm can still beat the techniques there. Needs Jialin's call (a large
base node cache outside the budget is a design choice).

Stage-1 / stage-2 tuning (stage1_tune.py; 11-16 fixes: index builds were single-threaded, ~2 h per
TriviaQA 10% build -> --threads; the stage-2 reference was brute force, which SPACEV's integer vectors
(exact distance ties) make unreachable at any search list (gap 0.012 at 2048) -> reference = search list
2048, brute-force retention still recorded; tuning requests: n doubled from 600 until >= 200 exact hits
(SPACEV Zipf 0.3 1% / 3% had 0 hits in 600: vacuous stage 1, no stage 2); --precisions):
- Quantized graphs (sq8 / sq6 / sq4, exact re-rank of 5x candidates) were never clearly cheaper than
  fp32 at a lower search list on the 6 cells tuned with all precisions (differences < 0.1 ms, within
  noise of 6 concurrent runs); the full set is tuned fp32-only.
- Stage 1 (hit decision): search list 16 passes nearly everywhere (agreement >= 0.99, |dhit| <= 0.005),
  0.1-0.5 ms per request; TriviaQA 0.6 needs 64 (1%) / 128 (10%), SPACEV real 10% 256.
- Stage 2 (served points on hits): 32-128 on ESCI / TripClick / MedRAG / SPACEV Zipf, 256-512 at
  SPACEV 0.6 3-10% and SPACEV real, 512 TriviaQA 1%, 2048 TriviaQA 10% (1024 loses 0.006).
- (Tables in timing_v4/<prep>/stage1_b<b>.json; the 600-request versions in stage1_first_version/.)

v5 harness (eval_timing_online.py / eval_timing.py / eval_timing_backend.py):
- --two-stage auto: stage 1 = lowest accepted fp32 search list, stage 2 = its choice. Stage 1 decides;
  on a hit, stage 2 is searched (timed, stage_ms["serve"]) and its points are served; on a miss the
  stage-2 answer is computed only when a label needs it (served-answer labels, doc 9o), as background
  work. Same in all three hit paths (FastHitPath: ours / drift; LRU._knn: LRU / continuous greedy).
- --node-cache auto: base node cache in the technique run's DiskANN and in both DiskANN-only arms; the
  equal-memory arm = base + the cache's memory.
- Audits: their backend call (composed: that request's DiskANN latency) and label are timed apart,
  never charged to a request; reported as audits_timed {n, total_s, mean_ms_per_audit,
  ms_per_timed_request}.
- Check (SPACEV 0.6 1%, 3,000 requests, busy machine): static ours one-stage (search list 512) kNN 0.97
  ms per request, hit rate 0.108, retention of hits 0.935; two-stage (16 / 32) kNN 0.08 + serve 0.01 ms
  per request, hit rate 0.111, retention 0.932. LRU / drift / continuous run (0.087 / 0.114 / 0.116 hit
  rate, retention 0.91-0.94). Audit accounting checked with audit rate 0.5 (12 audits, 9.7 ms each,
  excluded; hits 0.52 ms).
- timing_jobs.py --set v5 (results/timing_v5/<prep>/, states from timing_v4): the final set's workloads
  and techniques plus TripClick (real log: no static ours) and MedRAG (0.01-0.3%); 10 arm jobs + 202
  technique runs. run_timing_v5.sh: tuning (run_stage1_all.sh) -> quiet arms one at a time -> composed
  runs 12 at once (run_timing_composed.py --set v5 --version timing_v5). Flag /tmp/mx/TIMING_V5.DONE.
- Not yet: plot_timing.py still reads timing_v4; backend-recall sensitivity test (9ae d) after v5.
- Tuning, final (22:15): >= 200 exact hits in every cell (n 600-4,800 requests). Stage 1 = 16 everywhere
  except TriviaQA 10% (128 / 128 / 64 at Zipf 0.3 / 0.6 / 0.99), SPACEV real (64 at 1%, 512 at 10%).
  Stage 2 = 32-128 (ESCI, TripClick, MedRAG, SPACEV Zipf), 256 (SPACEV 0.6 3-10%), 512 (TriviaQA 1%,
  SPACEV real 1%), 1024 (SPACEV real 10%), 2048 (TriviaQA 10%). With 600 requests TriviaQA 0.6 1% had
  picked 64 for stage 1; with 1,200 (304 hits) 16 passes.
- SPACEV Zipf 0.3 1% / 3%: the state has no cutoff (no feasible one at the end of the warm-up: static
  ours serves nothing there; v4 hit rate 0.000), so no request is an exact hit at any n. The
  recalibrating controllers pick cutoffs later (v4 traces, median: 1%: drift 2.04, LRU 2.40, continuous
  1.79; 3%: 2.39 / 1.70 / 1.64). Tuned with stage1_tune.py --lam <those three>: stage 1 must pass at
  every cutoff, stage 2 checked on the hits of the lowest. Result: stage 1 16, stage 2 64.
- The SPACEV Zipf 0.99 v5 arm (first arm, 22:10) overlapped these two tunings for ~1-2 min: check its
  environment verdict; rerun it if contended.
- plot_timing.py: TIMING_VERSION=timing_v5 selects the v5 results.

#### 9af (cont.). v5 first results and two harness bugs (11-17 early morning, Jialin away)

Bugs found and fixed while the batch ran:
- AIO limit: with a node cache, open_diskann warms the cache with 32 threads = 32 AIO contexts per
  run; 12 composed runs at once exceeded the machine's 65,536 (io_setup EAGAIN, 5 runs crashed).
  Composed runs make no timed DiskANN call (charged the baseline latency, which includes the base node
  cache), so they now open DiskANN without node cache (2 contexts), as in v4. Crashed runs rerun by
  run_timing_v5_after.sh.
- Audit double subtraction: for a composed audit, the DiskANN charge was moved to audit time AND
  subtracted from the request's wall time, which never contained it: hits came out too cheap (SPACEV
  real drift: mean hit latency negative), speedups of drift / continuous / LRU too high (SPACEV 0.99
  ~10%). Fixed (audits_timed.accounting = 2). move_audit_bug.py moves the earlier audited runs to
  timing_v5/<prep>/audit_bug/ for rerun; until then v5_summary.py corrects them (adds the charges back:
  audits_timed.total_s = charges + microseconds of label work), marked *.

First v5 results (05:40; * = corrected pre-fix run). Speedup over DiskANN + base node cache; "node$"
= DiskANN given the cache's memory as extra node cache:
- SPACEV Zipf 0.99: 1%: ours cont.* 1.86, drift* 1.79, static 1.77, LRU* 1.64, Aker 1.52, node$ 1.07;
  10%: cont.* 2.04, drift* 1.94, static 1.91, Aker 1.80, QVCache 1.64, node$ 1.15. Ours wins clearly.
- TriviaQA 0.99: 1%: drift* 2.49, static 2.47, QVCache 2.25, LRU* 2.23, Aker 1.78; 10%: cont.* 3.15,
  LRU* 3.09, static 2.60, drift* 2.58, Aker / QVCache 2.55; node$ 1.00. Ours wins.
- TriviaQA 0.6: 1%: cont.* 1.24, static 1.17, Aker 1.07, QVCache 0.99; 10%: LRU* 1.85, static 1.46,
  Aker 1.35, QVCache 1.29; node$ 1.00. Ours wins.
- SPACEV Zipf 0.6: 1%: cont.* 1.09, static 1.06, Aker 1.02 (retention 0.89), node$ 1.05; 10%: cont.*
  1.19, static / drift* / Aker 1.13, node$ 1.11. Ours wins, narrowly.
- SPACEV Zipf 0.3: 1%: everything <= 1.00 (static ours has no cutoff: 0 hits, pays the lookup:
  0.97), node$ 1.07; 10%: static 1.08, Aker 1.07, node$ 1.11. node$ wins (low reuse).
- ESCI: 1%: static 1.04, drift* 1.03, Aker 0.99, node$ 1.09; 10%: static 1.16, cont.* 1.14, node$ 1.26.
  node$ wins: hit rates 9-24% bound any cache at ~1.25x; the extra node cache gains 26% at 10%.
- SPACEV real log (DiskANN recall 0.9): 1%: Aker 1.04, cont.* 1.03, node$ 1.05; 10%: Aker 1.09, drift*
  0.99, node$ 1.33. Aker / node$ win.
- TripClick: 1%: Aker 1.10, cont.* 1.00; 10%: Aker 1.20, cont.* 1.11 (hit rate 0.246 vs Aker 0.183,
  but its lookup costs ~1 ms per request: balls 0.31 + shell 0.43 + kNN 0.25 + serve, vs 9.4 ms
  DiskANN; Aker's lookup ~0.04 ms). node$ < 1. Aker wins. In the simulator (headline) Aker also beats
  ours at TripClick 1% (11.1% vs ogreedy 8.5%).

Why SPACEV real is weak (not a two-stage effect; v4 had the same 3.7%): the headline plot uses an
exact backend, the timing states a DiskANN backend at recall 0.9 (0.95 = 132 ms was over the old 100 ms
rule). Labels are the served answer's overlap with DiskANN's top-10; the cache serves truer neighbors
than a 0.9-recall DiskANN, so labels drop, the warm-up cutoff rises (state 1.71 vs ~1.0 in the
exact-backend simulator run at 10%) and is lost 29% of the time: drift ours serves 3-6% in the
stationary part vs ~10% in the simulator. Same mechanism, weaker, at recall 0.95 elsewhere. A SPACEV
real run at 0.95 (search list 2000, beam 16, 0.953) would cut it and make misses dearer, but its answers
take ~10 h (the 0.9 ones took 9.5 h, O_DIRECT reads at ~15k/s): not started, Jialin to decide.

Where the remaining lookup cost is (two-stage made kNN small): balls (distance to every entry center)
and shell (intruder count over 10 balls' samples) now dominate on many-entry caches (TripClick 10%:
0.31 + 0.43 ms; LRU's many small entries: balls 1.24 ms). An index over entry centers with an exact
bound (fit = (r_nn + d)/r >= (r_nn + d)/r_max per radius group) would cut balls without a new
tolerance: idea, not implemented. Also: with no cutoff (cal.lam None) no request can hit, so the score
is only needed for labels and could be background work (static SPACEV 0.3 1%: 0.97 -> ~1.0): not done.

#### 9af (cont.). v5 complete (11-17 09:20): 202 runs, all audited runs rerun after the fix

results/latest/timing.png, timing_table.md, timing_v5_summary.txt (v5_summary.py); v4 plot kept as
timing_v4.png / timing_table_v4.md. Best "ours" variant (static / drift / continuous) vs the best
other technique (Aker / QVCache tuned, meeting the target) vs DiskANN given the cache's memory as
extra node cache ("node$"), speedup over DiskANN + base node cache:
| workload | 1% (MedRAG 0.03%) | 10% (MedRAG 0.3%) | verdict |
| SPACEV Zipf 0.99 | ours 1.82 / Aker 1.52 / node$ 1.07 | 1.98 / 1.80 / 1.15 | ours wins at every size |
| TriviaQA 0.99 | 2.56 / QVCache 2.25 / 1.00 | 3.20 / 2.55 / 1.01 | ours wins |
| TriviaQA 0.6 | 1.24 / Aker 1.07 / 0.99 | 1.80 (LRU 1.85) / 1.35 / 1.00 | ours wins |
| TriviaQA 0.3 | 1.10 / 1.03 / 1.01 | 1.42 (LRU 1.66) / 1.19 / 1.01 | ours wins |
| MedRAG-Zipf | 2.78 / QVCache 1.19 / 1.00 | 2.91 (LRU 3.99) / 1.71 / 1.01 | ours wins |
| SPACEV Zipf 0.6 | 1.08 / 1.02 (ret 0.89) / 1.05 | 1.20 / 1.13 / 1.11 | ours wins, small margins |
| SPACEV Zipf 0.3 | 0.98 / 1.00 / 1.07 | 1.11 / 1.07 / 1.11 | node$ wins (ties at 10%) |
| ESCI (no repeats) | 1.04 / 0.99 / 1.09 | 1.16 / 0.99 / 1.26 | node$ wins; ours beats Aker / QVCache |
| SPACEV real log (DiskANN 0.9) | 1.03 / Aker 1.04 / 1.05 | 1.00 / 1.09 / 1.33 | Aker / node$ win |
| TripClick log | 1.00 / Aker 1.10 / 0.99 | 1.11 / 1.20 / 0.93 | Aker wins |
Ours (some variant) beats every other technique AND node$ on 6 of 10 workloads (all with repetition
skew: SPACEV 0.99 / 0.6, TriviaQA x3, MedRAG); continuous greedy is the best ours almost everywhere.
Loses: low-reuse SPACEV 0.3 and ESCI to node$ (a cache cannot reach 1.26x at 9-24% hits), and the two
real logs to Aker (SPACEV real: label vs a 0.9-recall backend, above; TripClick: ~1 ms lookup vs 9.4 ms
DiskANN, and fewer hits than Aker at 0.3-3%).
In flight: SPACEV real at DiskANN recall 0.95 (search list 2000, beam 16): answers started 05:31
(backends/spacev-real95/, --share-deep: one search per query; check 200/200 identical), ~10 h. Then
it needs states, tuning, a quiet arm and composed runs as workload alias spacev-real95.

#### 9af (cont.). SPACEV real log at DiskANN recall 0.95 (alias spacev-real95, 11-17 12:00)

Answers: search list 2000, beam 16, recall@10 0.953 over all 114,978 queries (4.6 h with --share-deep;
recall@500 of the deep lists 0.865). Base node cache 2,999 nodes (spacev-real's). DiskANN 129.5 ms.
Warm-up cutoffs: 1% 1.16, 10% 1.11 (at recall 0.9: 1.00 / 1.71). Tuning: stage 1 64 / 256, stage 2
512 / 2048. Results (speedup; hit rate):
- 1%: ours cont. 1.04 (4.4%), Aker 1.04 (3.3%), drift 1.03 (3.8%), LRU 1.01, QVCache 1.01; node$ 1.02.
- 10%: Aker 1.09 (8.3%), QVCache 1.07 (retention 0.72), cont. 1.06 (7.9%), LRU 1.05 (8.6%), drift
  1.04 (5.4%); node$ 1.05 (vs 1.33 at recall 0.9: at the larger search list the node cache matters less).
At 130 ms per DiskANN search the lookup is negligible; speedup ~ 1 / (1 - hit rate). Ours' hit
rates rose (drift 3.6 -> 5.4%, cont. 5.3 -> 7.9%) but stay at or below Aker's; in the exact-backend
simulator cont. reaches 10.5% vs Aker 8.3%, so the remaining gap is the DiskANN-relative labels.
Our hits cost ~9-11 ms (stage 2 at search list 2048 over 850k vectors: ~8 ms), irrelevant here.
Plot: 11th panel "SPACEV real log, DiskANN recall 0.95".

#### 9ae d. Backend-recall sensitivity, first cell: ESCI at DiskANN recall 0.97 (alias esci1m-r97, 11-17)

run_recall_variant.sh (generic alias pipeline). ESCI: search list 160, beam 16, recall@10 0.974 over
all 130,652 queries (answers 1.8 h). DiskANN + base node cache 11.69 ms (6.73 at 0.95). The arm's
guard said "contended" (3.1% of samples over the 1-core limit, mean other load 0.25 cores): kept.
| ESCI | 0.95: best ours / Aker / node$ | 0.97: best ours / Aker / node$ |
| 1% | 1.04 / 0.99 / 1.09 | static 1.08 (ret 0.896), cont. 1.05 / 1.00 / 1.05 |
| 10% | 1.16 / 0.99 / 1.26 | static 1.23 (ret 0.896), cont. 1.21 / 1.00 / 1.13 |
With dearer misses ours overtakes node$ on ESCI (the node cache's gain shrinks with the larger search
list: 1.26 -> 1.13 at 10%; SPACEV real 0.9 -> 0.95: 1.33 -> 1.05). Aker never hits ESCI (no repeats).
Next: SPACEV Zipf 0.6 at 0.97 (search list 256, beam 16, spacev10m index; alias spacev-aker-0.6-r97),
started 14:25. 0.98 / 0.99 not started (settings not probed for ESCI beyond 0.9775 at search list 200).

### 9ag. Why ours loses to Aker on TripClick (11-17 afternoon, with Jialin)

Decided: SPACEV real's main numbers use DiskANN recall 0.95 (spacev-real95; the plot's SPACEV real panel
now shows it). Sensitivity so far: only ESCI and SPACEV Zipf 0.6 at 0.97, 1% and 10%, every main
technique (ours x3, LRU, Aker / QVCache with the settings picked for 0.95, not retuned); no more queued.
SPACEV Zipf 0.6 at 0.97 (search list 256, beam 16, DiskANN 18.2 ms): 1%: cont. 1.13, static / drift
1.09, LRU 1.07, Aker 1.03 (ret 0.89), node$ 1.02; 10%: LRU 1.28, cont. 1.27, static 1.19, Aker 1.13,
node$ 1.09.

Cost side (aker_gap.py): DiskANN is fast on TripClick (9.4 ms) because 0.95 needs only search list 128
(beam 32), like ESCI / SPACEV Zipf (100), not because of the node cache (1,828 nodes; 11.4 ms without).
Our lookup costs twice as much there (1.01 ms per request at 10%: kNN 0.26, balls 0.32, shell 0.43) as on
SPACEV (0.47) since vectors are 769-d float32 (SPACEV 100-d int8, ESCI 384-d), and Aker gets many hits
(18.3% vs ours 25.2% at 10%: ratio 1.4, vs 2.2 on SPACEV 0.6 and ~inf on ESCI). LRU (our rule) gets
1.5-1.8x Aker's hits at every size but paid ~1.2-1.5 ms per request outside the stage timers. Diagnosis
(TripClick 10%, LRU, run alone): batch run hits 4.82 / misses 13.48 ms (0.87x) -> alone 2.71 / 11.45
(1.08x) -> alone, gc disabled 1.99 / 10.89 (1.16x; Aker 1.20). Python cyclic GC ~0.6 ms per request;
12 concurrent composed runs with heavy background HNSW inserts ~2 ms per miss (memory bandwidth; the
11-15 composed-vs-alone check covered static runs only). NO_GC=1 env switch added (diagnostic).

Hit side. TripClick timed requests: 68% new, 14% near-duplicates, 18% exact repeats (6.6% of requests
repeat within 2,000 requests, 2.8% within 10). Hit rate per class at 1%: Aker recent repeats 88%, near-
dups 29%; ours drift 11% / 4%; cont. 36% / 11%; LRU (our rule) 80% / 61% (all: 16.8% vs Aker 10.1%).
Offline / windowed population (greedy + rebuild, continuous greedy) misses short-term repetition; LRU
admission catches it. Why the headline (simulator) disagrees (LRU 7.0% vs harness 16.6%): the simulator
updates LRU / ogreedy once per 1,000-request batch, so a repeat within the batch can never hit; 5.5% of
TripClick requests repeat a query from the same batch (other workloads <= 0.7%). Simulator LRU 1%:
batch 1000 7.0%, batch 100 9.9% (retention 0.92-0.93). Drift detection on LRU: 15 alarms in 80 batches
(each discards all labels -> next batch no cutoff, then a strict cutoff from one batch of labels); LRU's
own churn moves the score / hit-rate distributions, so the tests mostly false-alarm (headline runs have
detection off for LRU / ogreedy).
Fix 1 (in progress): online_sim --update-every U (admission / eviction every U requests; calibration and
drift per batch). Continuous greedy scores a candidate on earlier misses only, so a first occurrence is
not admitted even per request (admitted after its 2nd, hits from its 3rd).

Plot: dotted lines "Ours, best case": the best (target-meeting) ours / drift / continuous / LRU run's
same timed requests and hits, misses charged exactly their DiskANN time, (a) hits at their measured
stage-2 serve time, (b) hits free. Reading (v5): TripClick best case 1.15-1.43x > Aker 1.09-1.20x at every
size (current LRU hits suffice); SPACEV real 0.95 best case = Aker (1.04 / 1.09): hit rate, not cost;
ESCI 0.95 best case ~= node$ (ties), at 0.97 above it; SPACEV Zipf 0.3 1-3% best case <= node$ (low reuse).

### 9ah. Refactor: one cache implementation, request path in C++ (11-17 evening, decided with Jialin)

Why: the simulator (online_sim.Cache, batch-vectorized, cache updated once per 1,000-request batch with
full rebuilds) and the timing harness (eval_timing_online: own Ours / LRU / OursDrift / OursOnline,
incremental, per request) were two implementations of the same cache; their differences caused the
recent mismatches (TripClick LRU 7.0% vs 16.6%, labels, drift alarms, two-stage only in timing) and the
Python request path costs ~1 ms per request on fast backends (interpreter work, cyclic GC).
Decided: (1) the request path (cached-vector index, stage 1 / 2 search, ball fit, intruder count,
score, serve; incremental add / evict; LRU bookkeeping) in C++ (pybind11), used by both headline and
timing; calibration, drift detection, greedy / continuous greedy / LRU population, drivers stay in
Python (background work, still changing); (2) graph library hnswlib 0.8.0 (vendored headers,
Apache-2.0: markDelete / unmarkDelete for evictions; FAISS vs hnswlib benchmark only as a background
sanity check); (3) headline runs keep exact (brute-force) search over the cached vectors, timing the
two-stage graph path; headline backend stays exact; (4) QVCache's request path also to C++ (fairness);
(5) a workload registry replaces the symlink aliases and per-workload special cases; (6) one replay
driver, per-request admission, calibration / drift on the batch schedule; old code kept as the
reference until parity tests pass, then deleted.
Environment: Python 3.9 has no headers installed (no sudo), so the new code runs in .venv-core
(Python 3.11; numpy 1.25.0, scipy 1.13.1, scikit-learn 1.6.1, faiss-cpu 1.8.0, diskannpy 0.7.0,
pybind11 3.1.0, hnswlib, matplotlib, pytest).

Milestone 1 (done): simulations/approximate_hits/cachecore/ (cachecore.cpp, build.sh, __init__.py).
State: cached vectors refcounted by id (dense store + hnswlib graph; evicted -> markDelete, revived ->
unmarkDelete; rebuild_index frees evicted slots), shell samples refcounted by id (stored as given, i.e.
already quantized; a sample is an intruder candidate only while its id is not cached), entries (center,
radius, ball ids, sample ids, last use; swap-remove rows), holders (vector -> entries, for LRU marks).
lookup(x, lambda, ef1, ef2): ef1 = 0 exact scan, else graph at ef1; shell over the whole sample
(shell_balls 0) or the n best-fitting entries' shells; stage-2 serve at ef2 on hits; per-stage times.
lookup_batch (no cache change in between): parallel over queries, stage 2 as a second pass (hnswlib's
search list is global). Exact distances: float32, `omp simd` reduction (-fopenmp-simd).
Tests: tests/test_cachecore.py (8, vs a NumPy reference: scores, intruders, neighbors, add / remove round
trip, empty cache, graph vs exact at high ef and after rebuild, batch = sequential, LRU marks).
Parity on the saved SPACEV Zipf 0.6 1% state (parity_static.py, 1,500 requests): exact / whole sample
vs eval_timing.HitPath and exact / 10 balls vs FastHitPath brute+balls10: 100% equal decisions, scores
within 1e-6; graph ef 64 (hnswlib) vs FastHitPath hnsw64 (FAISS): 99.6-99.9% equal decisions, equal hit
rate. Single thread: exact 4.9 ms (NumPy BLAS 4.3), exact 10 balls 1.9 (2.0), graph 0.16-0.18 ms (FAISS
path 0.25).
Not yet: database changes (inserts / deletes, the data-change experiments) are not supported by the
core; those stay on the old simulator. Quantization range is fixed per run (from the warm-up sample)
instead of recomputed at every rebuild as online_sim.prepare does.

#### 9ah (cont.). Node-cache full curves; decision (11-17 19:10)

node_cache_curve.py --full (to 10% of the points, quiet): ESCI 0 / 300 / 1,200 / 4,800 / 19,200 /
76,800 nodes = 8.69 / 6.80 / 6.58 / 6.47 / 6.15 / 5.62 ms: no saturation below the cache budget (within 3%
of the best only at 76,800 nodes = 7.7% of the points, ~140 MB). SPACEV real at recall 0.95 (search list
2000): 129.7 ms without, 127-130 ms at every size up to 1.9%: the node cache does not matter there.
Decided (Jialin): keep the small base node caches (ESCI 300 nodes); the equal-memory line shows what the
rest of the memory would buy DiskANN.
Milestones 2-5 code (committed): core_cache.CoreCache (online_sim --cache-impl core), per-request
admission (--update-every 1), timed per-request mode (--timing-compose / --timing-every, --core-ef1 /
--core-ef2), QVCacheCore (simulator QVCache on cachecore.QVCore, per-request inserts; timed mode with
hnswlib mini-indexes, search list 64), workloads.py registry. Parity / validation runs queued:
run_core_parity.sh (TripClick 1%, python vs core: greedy, LRU, continuous greedy; LRU per request),
run_core_timing_check.sh (timed mode vs v5 harness: SPACEV 0.6 1% static / LRU; TripClick 10% LRU /
continuous / drift).

#### 9ah (cont.). Parity, validation and the v6 batch (11-17 evening)

Parity (online_sim --cache-impl python vs core, TripClick 1%, 20 batches): greedy + rebuild identical
(20 / 20 batches); LRU / continuous greedy within 0.002 hit rate, 0.006 retention: with --bits 32 (no
quantization) LRU is identical (20 / 20, same memory and entries), so the only difference is the fixed
quantization range (by design). Per-request LRU (core, --update-every 1), TripClick 1%, full stream:
15.1% hits, retention 0.912 (per-batch headline LRU 7.0%; Aker 11.1%).
hnswlib vs FAISS (bench_hnsw_libs.py, 10% caches of SPACEV / TripClick / TriviaQA): within ~10% in
time and recall at every search list; hnswlib slightly faster at the small stage-1 lists.
Core per stage, 10% caches (state of the static warm-up cache, one thread): TripClick ef1 16: kNN 0.13,
balls 0.23, shell 0.29, serve 0.16 ms per request (v5 Python: 0.26 / 0.32 / 0.43, serve 0.71 per hit);
SPACEV 0.6: 0.08 / 0.13 / 0.004 / 0.06.
Timed-mode validation vs v5 (same answers, baseline latencies, search lists): SPACEV 0.6 1% static:
hit 0.108 both, 6.69 vs 6.73 ms (1.064x vs 1.057x); TripClick 10% drift: 1.02x vs 0.94x (hits 1.38 vs
2.06 ms, misses 10.35 vs 10.97); SPACEV 0.6 1% LRU: 0.98x vs 0.99x; TripClick 10% LRU: 1.045x vs 0.87x
(with the 36 tuning jobs running), 1.125x alone (hits 2.00, misses 11.27; Aker 1.20x). LRU's remaining cost
is algorithmic: ~4,000 small entries at 10%, balls 0.75-1.09 ms and shell 0.77-0.89 ms per request
(entries x 769-d, memory-bandwidth bound, so also sensitive to concurrent jobs).
Fixes found on the way: lambda -inf now = every request takes the hit path (labels / tuning got
stage-1 answers before); exact mode keeps no graph (was ~4x slower, 2.8 GB reserved at TriviaQA 10%);
CoreCache bookkeeping incremental (Cache recomputed refcounts / shell union per call: per-request LRU
10k requests > 25 min -> ~2 min); incremental graph inserts at efConstruction 40 (as the harness);
continuous greedy keeps per-batch policy updates (its update re-scores 20,000 requests' misses, ~4 s per
call; per-request admission for it needs an incremental implementation: open), lookups per request.
Tuning on the core (tune_core.py, 36 cells): stage-1 lists differ from the FAISS tuning, which re-ranked
5x candidates exactly (the harness never did): e.g. TripClick 10% 64 (FAISS tuning 16), stage 2 128 (same).
v6 batch (jobs_v6.py, run_jobs.py, 8 at a time, 4 cores each): 340 timed runs: ours static / drift /
continuous, LRU (k 20, per request), QVCache (core, hnswlib mini-indexes) and real Aker, Aker / QVCache
at up to 3 candidate settings (best in the headline among those meeting the target) so the plot picks the
best meeting the target in the timed run; arms from v5. First Aker runs failed (library path) and are
rerun by run_v6_after.sh, which then times the two heaviest LRU cells alone, runs the headline per-request
reruns (LRU k10 / k20 x 2 seeds, QVCache 16x8 grid, on TripClick, SPACEV real, SPACEV 0.99, TriviaQA 0.99:
same-batch repeats 5.5% / 0.7% / 4.1% / 10.8% of requests; others <= 0.12%) and the plots.

### 9ai. Cheaper ball fit / intruder count; which "ours" wins (11-18 early morning)

Variants of our hit rule: static (greedy at warm-up, fixed), drift handling (greedy + score drift test +
shadow rebuild; headline greedy_score_rebuild), continuous greedy (ogreedy, utility eviction, per-batch
updates), LRU with our rule. Best variant meeting the target, v6 timing / headline (headline: per-request
reruns where done): continuous greedy wins 27 / 36 timing cells and 22 / 32 headline cells; static ties
it within 0.01x in 4 timing cells; LRU wins TripClick 0.3-3% in both (short-term repetition: 14.7-23.9%
hits vs 5-13%) and the 10% caches in the headline (SPACEV 0.3 / 0.6 / 0.99, TriviaQA 0.3 / 0.6, MedRAG
0.3%), in timing only TriviaQA 0.3 10% and MedRAG 0.3%. Drift handling never wins. Open: per-request
admission for continuous greedy (the TripClick small-cache gap).

Untimed checks (ballfit_experiments.py, 10% warm-up caches, 5,000 test requests):
- A: score from the 3 NEAREST balls (and intruders over the 10 nearest), model refit + recalibrated:
  hit rate vs best-fit balls: SPACEV 0.6 21.5 vs 21.9%, TripClick 27.5 vs 27.4, TriviaQA 0.6 41.1 vs 42.7.
- B: best fit among the m nearest centers, state model / cutoff: decision agreement >= 0.99 down to m 64
  (SPACEV, TripClick), 128 (TriviaQA); smaller m serves more at lower retention (missed best balls ->
  fewer intruders), so m must be tuned or the path recalibrated.
- C: 1-bit RaBitQ intruder counts (state model): decisions agree 0.995 / 0.93 / 0.87; D: with the model
  refit and cutoff recalibrated on RaBitQ counts: SPACEV 22.0 vs 21.9%, TripClick 26.7% (ret 0.902) vs
  27.4% (0.894): the model tolerates the count error. Decided: packed 4-bit codes now, RaBitQ later as a
  separate step (it also changes the sample's memory charge).
Core changes (cachecore.cpp): packed 4-bit shell codes (two per byte, per-dimension lo / step; distance
||x||^2 + ||v||^2 - 2 x.lo - 2 sum x_i step_i c_i with ||v||^2 stored: identical counts; parity with the
Python simulator 20 / 20 batches), center index (hnswlib over entry centers) with ball modes fit-m / near,
near-scan (nearest by full scan), near scans caches of <= 1,000 entries (speed only). Interleaved
microbenchmark (TripClick static warm-up caches, loaded machine, ratios fair): 10%: ball finding 0.88 ->
0.35 ms (near, center index), intruder count 0.66 -> 0.38 (packed); 1%: balls 0.23 -> 0.21, count 0.16 ->
0.095. A first end-to-end TripClick run (timing_v6b) ran beside heavy headline jobs and is not usable for
timing (hit rates / retention unchanged).
Pool statistics (online_sim --pool-stats; jobs_v6 --set pool, 60 runs): per hit, entries holding served
vectors, greedy cover size, best single entry's share of the served top-k, best single-entry retention vs
the backend's top-k. Test (TripClick 1%, 5 batches, greedy): 3.1 entries hold served vectors, cover 1.42,
best entry 94% of the served top-k, single-entry retention 0.846 vs pool 0.882.
Scheduling (03:00): headline per-request reruns ~12 / h (exact scans per request, 8 threads each), 65 / 264
done; paused with SIGSTOP (run_quiet_now.sh, /tmp/mx/paused_pids.txt) for the quiet checks (solo LRU cells,
then TripClick near / fit-packed runs, 4 at a time, timing_v6c), resumed automatically afterwards.

#### 9ai (cont.). Quiet TripClick checks (timing_v6c, 11-18 03:32-04:38)
18 timed runs, 4 at a time on an otherwise idle machine. `fitpacked_` = default ball fit + packed 4-bit
shell codes, `near_` = nearest-center ball (center index) + packed codes. Same composed timing as v6
(each miss charged that request's measured DiskANN latency). Speedup over DiskANN alone (mean latency;
v6 = float codes, best fit, 8 runs at once):

| budget | variant | v6 | fit + packed | near + packed | best Aker meeting the target |
|---|---|---|---|---|---|
| 0.3% | LRU, our rule | 1.127 (hit 0.146) | - | 1.126 (0.142) | 1.084 |
| 1% | LRU, our rule | 1.119 (0.174) | 1.142 (0.174) | 1.138 (0.174) | 1.101 |
| 1% | continuous greedy | 1.032 | 1.038 | 1.038 | |
| 10% | LRU, our rule | 1.012 (0.326) | 1.037 (0.326) | 1.165 (0.335) | 1.201 (hit 0.183, ret 0.937) |
| 10% | continuous greedy | 1.119 (0.251) | 1.138 (0.248) | 1.125 (0.245) | |
| 10% | drift handling | 0.988 | 1.016 | 0.988 | |

- Sanity: fit + packed hit rates match v6 within 0.001 for LRU and within 0.004 for continuous greedy
  (packed codes move a few intruder counts; drift handling moves more, 0.134 -> 0.124 at 10%, because
  its drift test and rebuilds amplify small differences). Retention stays at 0.909-0.934 everywhere.
- Mean stage costs per request, TripClick 10% LRU: ball finding 1.29 ms (v6) -> 1.16 (fit, quiet)
  -> 0.27 (near); intruder count 0.84 -> 0.76 -> 0.73; first-stage kNN 0.43. With `near`, the intruder
  count is now the largest term. The packed codes alone save ~0.1 ms here; most of the v6 -> quiet gain
  at 1% comes from the quiet machine.
- Result: `near` lifts TripClick 10% LRU from 1.04x to 1.17x but it is still below Aker's 1.20x: we
  serve 33.5% of requests vs Aker's 18.3%, yet our hit path costs ~1.6 ms per request on average vs
  Aker's ~0.04 ms lookup. Below 10% we beat Aker at every size (0.3%: 1.13 vs 1.08, 1%: 1.14 vs 1.10).
  `near` does not help continuous greedy (its balls are already few; near picks balls with larger
  shells, intruder count 0.20 -> 0.30 ms).
- Next lever for the 10% cells: the intruder count (fewer shell samples per ball, or 1-bit RaBitQ codes,
  9ai D), not the ball search.
- Bug found: QVCacheCore with --qv-graph and no --timing-compose tried to load the timing file (TypeError
  at start); fixed (load only in timed mode). Only the 4 QVCache graph twins (below) hit it; rerun.

#### 9ai (cont.). Cost of the graph path in the headline (graph vs exact twins, 11-18 05:30)
The headline per-request reruns now search the cache with the two-stage graph path (hnswlib; QVCache:
mini-index graphs, --qv-graph) instead of exact scans. To measure what that costs, 8 TripClick cells
already finished with exact scans were rerun with the graph path (results/online_v6_graphcheck). The
backend answers are exact in both. Hit rate (retention of hits):

| budget | method | exact scans | graph path | change in hit rate |
|---|---|---|---|---|
| 0.3% | LRU, our rule, k 10 | 11.38% (0.916) | 10.79% (0.919) | -0.59 pts (-5.2%) |
| 0.3% | LRU, our rule, k 20 (best) | 14.94% (0.913) | 14.60% (0.914) | -0.34 pts (-2.3%) |
| 3% | LRU, our rule, k 10 | 19.96% (0.908) | 19.22% (0.908) | -0.74 pts (-3.7%) |
| 3% | LRU, our rule, k 20 (best) | 23.89% (0.912) | 22.82% (0.914) | -1.07 pts (-4.5%) |
| 0.3% | QVCache D 0, m 8 | 8.00% (0.837) | 6.63% (0.798) | -1.37 pts (-17%) |
| 3% | QVCache D 0, m 8 | 12.68% (0.840) | 10.38% (0.807) | -2.30 pts (-18%) |
| 0.3% | QVCache D 0.075, m 4 | 17.92% (0.423) | 17.99% (0.418) | +0.07 pts |
| 3% | QVCache D 0.075, m 4 | 25.42% (0.466) | 25.37% (0.465) | -0.05 pts |

- Ours (LRU with our rule) loses 2-5% of its hits, and retention is unchanged, because the calibration
  sees the same graph path it is served with. Graph misses mean some useful entries are not found.
- QVCache with a tight threshold (D 0) loses ~17% of its hits and ~0.035 retention. No QVCache setting
  here meets the 0.9 target in either mode, so its headline point (best below target) moves little.
- Aker (real library) uses its own index in both modes, so the graph path costs only ours and QVCache:
  the headline comparison against Aker is conservative by about 2-5% of our hit rate on TripClick.
- Only TripClick was checked. The loss may differ on other workloads (e.g., larger caches, other
  dimensions); add twins elsewhere if a headline cell is close.

### 9aj. Status check, plot agreement, wins and losses (11-18 12:40, Jialin back)

State: headline per-request reruns done except 1 run; pool statistics 57 / 60; plots redrawn 12:37
(results/latest/headline.png, timing.png). New queues (all untimed, run beside each other):
- v6_twins (61 runs): graph-path twins of the 65 exact-search headline runs already done (all TripClick:
  11 LRU, 54 QVCache), into results/online_v6_graphcheck. Measures what dropping brute force cost.
- attrib (24 runs, results/attrib): why timing hit rates are below headline hit rates (next section).
- v6_headline_exact (jobs_v6.py --set headline-exact; 199 runs + 65 copies of exact runs already done,
  results/online_v6_exact): the 264 headline reruns with exact search over the cache, 8 threads each.
  Starts after every other queue (run_exact_last.sh), main sizes first.

Do the two plots agree? Aker: yes, within 0.3 points in every cell (same code, same stream; headline
Aker uses the exact backend, timing the DiskANN answers, and it does not matter for Aker). TripClick LRU
and QVCache: within 0.1-1.9 points (both are per-request on the graph path). Ours and LRU on all other
workloads: no. The timing hit rate is lower than the headline's by 1-4 points on SPACEV, 3-11 on
TriviaQA, 2-10 on ESCI (timing > headline on MedRAG LRU, +1 to +16, whose headline LRU is still the old
per-batch run). Differences between the two kinds of runs (args diff, TriviaQA 0.3 10%): timing uses the
DiskANN recall-0.95 answers as backend (headline: exact answers), shell samples drawn from the 10
nearest balls only (headline: 3 balls, all intruders), the graph search lists tuned on the core, LRU
k 20 (headline: best of 10 / 20) and the headline numbers are the mean of 2 seeds. See the attribution
table below for which of these costs how much.

Wins and losses against the best alternative (best of Aker, QVCache, and for timing also DiskANN with
the cache's memory as extra node cache), best variant of ours (greedy / continuous / drift / LRU with
our rule) picked after the fact, as the alternatives are:
- Headline, per-request reruns (hit rate at recall 0.9): ours >= best alternative in all 40 cells
  (smallest ratios: TriviaQA 0.99 1%: 1.12x, SPACEV real 10%: 1.27x, SPACEV 0.99 1%: 1.30x, TripClick
  0.3 / 1 / 3%: LRU 14.7 / 18.0 / 23.9% vs Aker 9.2 / 11.1 / 13.6%, 10%: 34.2 vs 18.3%). Before the
  per-request LRU rerun, TripClick 0.3-3% were losses (ours 5.2 / 8.5 / 13.2% vs Aker 9.2 / 11.1 / 13.6).
- Timing (speedup over DiskANN alone): ours wins 27 of 36 cells. The 9: SPACEV Zipf 0.3 at 0.3 / 1 / 3%
  (node$ 1.02 / 1.07 / 1.08 vs ours 1.00 / 1.00 / 1.04; low reuse), ESCI at all four sizes (node$ 1.05 /
  1.09 / 1.17 / 1.26 vs ours 1.05 / 1.06 / 1.08 / 1.17), SPACEV real 10% (Aker 1.088 vs 1.057) and
  TripClick 10% (Aker 1.20 vs 1.12 in v6; 1.165 with the new ball finder, quiet run). Closest wins:
  TripClick 3% +0.6%, SPACEV real 1% +0.8%, TripClick 1% +1.6%.

### 9ak. Exact-repeat fast path, RaBitQ shell codes, fewer shell samples (11-18 afternoon, approved by Jialin)

Goal: cut our per-hit cost on TripClick 10% (hit path ~1.6 ms per request vs Aker's ~0.04 ms lookup; free
hit decision would give 1.40x vs Aker 1.20x).

Exact-repeat fast path (`--core-repeat-fast`, core `set_repeat_fast`): the core keeps a hash of every entry's
center (FNV-style over the vector's bytes; exact compare on a match). A request equal to a cached center is
served that entry's own k nearest ball members, skipping kNN, ball finding, shell count and model. With an
exact backend, that is the backend's top-k for that query (retention 1). With the DiskANN backend, it is the
first k of the backend's deep list for it, close to its top-k answer. The request scores FAST_SCORE (1e9):
it is served at every cutoff, so calibration counts it as a hit at every cutoff. Its label comes from
audits, as for any hit. Batch lookups (fit / calibration) still compute its full features; the model is
fit on the normal features. Not served before a lambda exists. TripClick: 18% of timed requests are exact
repeats (9ag). Only our cache (ours / LRU with our rule) has it: Aker already serves exact repeats cheaply.
QVCache could get the same shortcut.

RaBitQ shell codes (`--core-shell-code rabitq`, core `set_rabitq`): 1-bit RaBitQ as the untimed check (9ai C/D,
ballfit_experiments.RaBitQ). The centroid is the first sample's mean, with a dense random orthogonal rotation
(numpy QR, seed 0; the core gets P^T). Each sample stores sign bits of P^T (o - c) / ||o - c||, plus
||o - c|| and <x, u>. Codes come from the raw vectors, not the 4-bit ones. A request builds per-byte lookup
tables (d / 8 x 256 partial sums of its rotated vector, ~25k adds + one d x d rotation), then each sample
costs d / 8 table adds instead of d multiply-adds on 4-bit codes. The model is fit at warm-up on the RaBitQ
counts (as check D). Memory is still charged at 4 bits per dimension (conservative: the real size is
1 bit + 8 bytes, ~27% of the 4-bit size at 769-d); giving the freed memory to more samples or more cache is
a follow-up. Tests: the core's counts match the NumPy reference (78 / 80 within 1); on 16-d Gaussian data the
estimate is biased up near the boundary (many samples just outside), but tracks the exact count
(correlation > 0.5).

Fewer shell samples: no code; `--f 0.95` (shell 5% of memory instead of 15%) and `--shell-balls 3` (count over
the 3 nearest balls' shells instead of 10).

Runs (timing_v6c/tripclick-ip, quiet window via run_quiet2.sh, 4 at a time, untimed queues paused): LRU and
continuous greedy x {fast, rabitq, fastrq, fastf95, fastsb3} x 10% / 1% / 3% / 0.3%, against the near_ runs of
9ai (same quiet conditions). Untimed smoke runs first (results/smoke9ak, TripClick 1% LRU).

Smoke runs (untimed, TripClick 1% LRU k 20, per request, near + timing settings; results/smoke9ak):
baseline (near_, 9ai) 17.39% / 0.914; fast path 17.49% / 0.914; RaBitQ 18.01% / 0.913; both 17.96% / 0.915.
The quiet window started 15:37 (13 processes paused).

Graph vs exact twins, all 65 (TripClick, 9aj): LRU with our rule loses 1.5-5.9% of its hits (best setting
k 20: 1% -1.5 / -3.1% for seeds 0 / 1, 0.3% -2.3 / -2.4%, 3% -4.5 / -5.9%), retention unchanged. QVCache:
D 0 loses 17-19% of its hits and ~0.035 retention; every other D is within 0.8%. No QVCache setting meets
0.9 on TripClick in either mode.

#### 9ak (cont.). First quiet results, TripClick 10% and 1% (11-18 17:00)
Speedup over DiskANN alone (hit rate, retention), all with the nearest-center ball finder and packed codes
unless RaBitQ. Mean stage costs per request in ms (knn / balls / shell):

| | LRU 10% | cont. greedy 10% | LRU 1% | cont. greedy 1% |
|---|---|---|---|---|
| near (9ai) | 1.165 (33.5%, .910) .43/.27/.73 | 1.125 (24.5%) | 1.138 (17.4%) | 1.038 (6.9%) |
| + fast path | 1.184 (33.6%) .41/.26/.70 | 1.151 (25.2%) | 1.136 (17.4%) | 1.033 (6.6%) |
| RaBitQ | 1.165 (31.8%, .918) shell .59 | 1.118 (24.0%) | 1.135 (18.0%) shell .18 | 1.056 (9.7%) |
| fast + RaBitQ | 1.177 (31.9%) | 1.126 (24.0%) | 1.138 (18.0%) | 1.059 (9.7%) |
| fast + f 0.95 | **1.267** (33.1%, .913) shell .21 | 1.141 (22.6%) | **1.144** (17.3%) shell .03 | 1.030 (5.8%) |
| fast + 3 shell balls | 1.252 (32.9%, .913) shell .23 | 1.167 (24.0%) | | 1.044 (6.7%) |
| Aker (best meeting 0.9) | 1.201 (18.3%, .937) | | 1.101 (11.1%) | |

- TripClick 10% now beats Aker: 1.267x (LRU, fast path, shell 5% of memory) vs 1.201x. Fewer shell samples
  are the lever: the intruder count drops from 0.70 to 0.21 ms per request at a 0.5-point hit-rate cost.
- The fast path alone saves ~0.13 ms per request at 10% (1.165 -> 1.184), little at 1%.
- RaBitQ as implemented is not faster: the per-request dense rotation (d^2 = 591k multiply-adds at 769-d)
  and table build cost ~0.1+ ms, more than the per-sample savings on small shells (1%: 0.107 -> 0.181 ms).
  A fast structured rotation (random signs + Hadamard, d log d) would remove most of that. Its counts change
  hit rates: higher at 1% (LRU 17.4 -> 18.0%, continuous greedy 6.9 -> 9.7%), lower at 10% (33.5 -> 31.8%).
- Caveat: f 0.95 was picked after seeing TripClick. Under the no-per-workload-tuning rule, the shell size
  must come from one setting or rule for every workload (9ae: the dev-chosen split, eval_shell_budget), and
  the other workloads must be checked with it before it replaces f 0.85.

#### 9ak (cont.). Corrections and checks (11-18 evening)
- QVCache / Aker lookup cost IS in the timing: every timed request's latency is the measured lookup plus,
  on a miss, the composed DiskANN time. Only the per-stage breakdown (stage_ms) is ours-only. Mean latency
  on hits (v6, 10%): QVCache 0.30-1.95 ms, Aker 0.02-0.05 ms, ours 0.48-0.69 ms (ESCI), 1.54 (TripClick
  continuous greedy), 6.56 ms (SPACEV real continuous greedy: v6 ball finding scans every entry; the f95
  rerun uses the nearest-center finder). On a miss, ours adds 0.3-2.8 ms to DiskANN's time; QVCache adds
  0.0-2.0 ms.
- SPACEV real: 7.32% of requests after warm-up are exact repeats of earlier queries (TripClick 18.8%,
  SPACEV Zipf 0.3 2.4%, ESCI 0). Aker's 8.25% hits are almost all of them, and ours (8.4% in timing,
  10.5% headline) are mostly the same requests: this log has little near-neighbor reuse beyond exact
  repeats.
- ESCI node cache (node_cache_curve, 9ah): 8.69 ms with no node cache, 6.80 at 300 nodes, 5.62 at 76,800;
  DiskANN alone in the arms (300 nodes) 6.73, node$ at 10% 5.35 ms. Not yet explained; check DiskANN's
  per-search I/O counts with and without the node cache.

### 9al. Hit-rate ladder on static caches (11-18 evening, decided with Jialin)

Direction change: instead of changing the implementation and re-timing to find what saves time at the same
hit rate, measure first what each approximation costs in hit rate, on static warm-up caches, and only time
the approximations that cost little. Then validate on a few targeted dynamic (per-request) runs.

hit_table.py (one table per workload x budget x cache variant x backend): the simulator's warm-up cache,
then for every distinct request (warm-up fit / calibration splits, and up to 50,000 post-warm-up requests)
the raw ingredients of the hit decision: exact kNN over the cached vectors (nr, served answer, retention vs
the backend's and the exact top-k), the 10 best-fitting and 10 nearest entries, single-entry serving
(best-fit, nearest, best of 10), intruder counts for scope {3, 10} x order {fit, near} x sampling rate
{4, 2, 1, 0.5, 0.25, 0.1} x p0 (nested draws: one pass) in float and 4-bit, RaBitQ at p0, the whole sample
at p0, the vectors examined (cost proxy), and graph variants from the core (stage-1 nr at ef 16 / 64 / 256
and the tuned ef, fit order, deployed counts, stage-2 served answers). Exact parts are batched matrix
products. Validated: features equal the simulator's (max difference 0.0002), labels identical.
Cache variants: default (f 0.85, K 10..500), maxk10 (K = k: past top-k only, like Aker / QVCache), f95,
f100 (no shell); backends: exact, diskann (balls / shells / labels from DiskANN's lists and answers).
hit_ladder.py: per variant, on random 30 / 30 / 40% train / calibration / test splits of the post-warm-up
requests (5 draws): oracle hit rate (given the served answers), model with the best cutoff (ranking
quality), model + calibrated lambda (online_sim's rule; every request labeled, as --oracle-cal), plus
training-size and calibration-size sweeps and the as-deployed warm-up splits. --summary: every variant
relative to the reference (3 best-fit balls + count over the whole 4-bit sample, exact search) across
cells, and the cross-table comparisons (backend, cache variants, oracle whole cache vs single entry).
First look (TripClick 1%, test with 1,000 requests only): on a static cache graph search costs nothing
(median stage-1 nr ratio 1.001, served retention 0.1596 vs 0.1597): the 1.5-5.9% loss of the dynamic
graph twins (9aj) must come from the online graph (incremental inserts at efConstruction 40, deletions).
Cache composition there: 55% of cached vectors are in their entry's first k; entries K 10: 484, 20: 250,
50: 75, 100: 15, 200: 2; shell 15% of memory.
Queue: 200 jobs (5 cache / backend variants x 40 cells: 10 main workloads x 4 headline sizes), 12 at a
time, after the f95 timing rerun (run_tables.sh). The exact-search headline rerun is cancelled (not
started; its job file /tmp/mx/v6_headline_exact.txt is kept).
Planned dynamic checks after the ladder: per-request LRU / continuous greedy on the online graph vs exact
(graph degradation), the chosen cuts per request, and the 2 x 2 (past top-k vs expanded balls) x (single
entry vs whole cache) on dynamic caches.

#### 9al (cont.). Online graph check and first ladder results (11-18 23:30)
Online graph (TripClick LRU k 20, per request, seed 0; hit rate / retention):

| | 1% | 3% |
|---|---|---|
| exact search | 18.20% / .912 | 23.89% / .912 |
| graph, tuned lists, inserts at efConstruction 40 (v6) | 17.94% / .910 | 22.82% / .914 |
| graph, tuned lists, inserts at 200 | 18.21% / .910 | 23.62% / .914 |
| graph, lists 64 / 256, inserts at 200 | 18.18% / .911 | 23.75% / .913 |

The graph loss of the dynamic runs (9aj) is mostly the cheap incremental inserts (`--core-insert-ef`, new;
inserts are background work). Decision candidate: inserts at 200 by default.

f 0.95 timing rerun, first 19 of 72 cells (then held for the tables): ESCI fails (1%: 0% hits for LRU and
continuous greedy, v6 4.8 / 8.9%; 10%: LRU 15.6 -> 9.5%); elsewhere small gains (SPACEV 0.99 10% LRU 1.80 ->
1.93x, TriviaQA 1% +0.03x). One fixed shell size does not fit every workload.

Ladder, default cache, exact backend, all 40 cells (5 draws each; hit_tables/summary.md). Calibrated hit
rate relative to the reference (3 best-fit balls + count over the whole 4-bit sample, exact search):
- The reference gets 79% of the oracle on average (by workload 0.58 TriviaQA 0.3, 0.62 SPACEV 0.3, 0.71 ESCI /
  TripClick, 0.78-0.83 TriviaQA 0.6 / SPACEV 0.6 / SPACEV real, 0.95-0.97 Zipf 0.99 and MedRAG).
- Features: 2 balls lose nothing (mean 1.007, min 0.978); 1 ball 0.996 mean but 0.92 on SPACEV 0.3 (min
  0.69); without the intruder count 0.85 (min 0.41): the shell count matters, the 3rd ball does not.
- Count precision: float counts beat the 4-bit codes by 5% on average (TriviaQA 0.3 +29%, SPACEV 0.3 / TripClick
  +6%): the 4-bit scalar quantization with the sample's range is lossy. RaBitQ (1 bit of the raw vectors) is
  about as good as or better than the 4-bit codes over the same shells (scope 10 balls: 0.974 vs 0.939).
- Shell scope / size: counting over the shells of the 10 best-fitting balls instead of the whole sample:
  0.939 (TriviaQA 0.3 0.85, SPACEV 0.3 0.87); 3 balls 0.918; a quarter of the samples 0.88; more samples help
  (p x4 over 10 balls: 0.993).
- Ball finder: the 3 nearest centers instead of the best-fitting: ESCI 0.47, SPACEV 0.3 0.76, others 0.94-1.0.
  Bad where requests are not repeats.
- Graph search (static cache, tuned lists): 0.93-1.02 by workload (SPACEV real / TriviaQA 0.3 0.93); fixed small
  stage-2 lists collapse on large caches (stage 2 must be tuned per cache).
- Serving from one entry instead of the whole cache: 0.64 mean (ESCI 0.08, TriviaQA 0.3 / 0.6 ~0.5, MedRAG 0.58,
  Zipf 0.99 0.92). The oracle of single-entry serving is also far lower (ESCI 10%: 23.1 vs 37.1%).
- Calibration size: 2,000 requests 0.97, 1,000 0.93, 250 0.80 (min 0.20); 8,000-16,000 +4%. Training size barely
  matters above 1,000 (0.99).
- As deployed (warm-up splits) 1.05 of the reference, but retention < T - 0.01 in 4 of 40 cells (the warm-up
  does not represent later requests in those cells).
- DiskANN backend (cells done so far): -0 to -10% (ESCI 10% 30.1 -> 27.0%), SPACEV 0.3 0.3%: 2.0 -> 0.6%.

#### 9al (cont.). Backend, candidates, count codes (11-19 01:20; 110 / 200 tables)
- 4-bit codes over a robust range (1st / 99th percentile per dimension) match float counts: 1.056 of the
  reference (float 1.051, 8-bit 1.047): the 4-bit loss is the min / max range (outliers), not the width.
  New core option `--shell-range robust` (CoreCache), to validate per request.
- DiskANN backend (all 40 cells), reference variant: 0.93 of exact on average (min 0.31); SPACEV 0.3 0.75,
  SPACEV real 0.87, TriviaQA 0.6 0.90, ESCI 0.93, the rest 0.96-1.0.
- Past top-k only (maxk10, 27 of 40 cells so far): 0.50 of the default cache on average (SPACEV 0.3 0.04,
  TriviaQA 0.3 0.23, SPACEV 0.6 / TriviaQA 0.6 ~0.34, ESCI 0.44, SPACEV real 0.54, MedRAG 0.67, Zipf 0.99
  0.79-0.87, TripClick 0.92). Expanded balls roughly double the hit rate except where hits are repeats.
- Cache composition at 1%: only 7-16% of cached vectors are in their entry's first k (SPACEV, TriviaQA,
  SPACEV real), 41% ESCI, 55% TripClick; the shell sample is 15% of memory.
- Whole cache vs one entry: the oracle of serving the best of the 10 best-fitting entries is 0.83 of the
  whole cache's on average (ESCI 0.63, TriviaQA 0.3 0.71, SPACEV real 0.76); calibrated, one entry gets 0.64.
  So of the two differences to Aker / QVCache, expanded balls matter most (x2), whole-cache serving next
  (x1.2 at the oracle, x1.6 calibrated).
- The remaining f95 timing runs were cancelled at 00:28 (19 of 72 cells kept): one shell size does not fit
  every workload (ESCI fails at 1%), and the ladder will choose the configuration to time.
- TripClick upload to Google Drive: README drafted (datasets/tripclick/README_upload.md); not uploaded, pending
  Jialin's confirmation that the Drive folder is private (TripClick terms forbid public sharing / hosting of
  any part, derived resources included).

#### 9al (cont.). Headroom, calibration miss, consistent comparisons (11-19 02:00)
- Upload: TripClick vectors, requests, texts, raw shards, README and LICENSE to gdrive:Datasets/tripclick
  (Jialin confirmed; private use only).
- Oracle gap, split (mean over sizes; oracle | model's best cutoff | calibrated): TriviaQA 0.3 26.4 | 19.1
  (0.72) | 17.2; SPACEV 0.3 10.6 | 8.3 (0.79) | 7.0; ESCI 20.9 | 16.9 (0.81) | 15.4; TripClick 11.7 | 9.9 (0.85)
  | 8.9; TriviaQA 0.6 36.4 | 30.2 (0.83) | 28.6; Zipf 0.99 / MedRAG 0.97-0.99 | ~0.98. Calibration costs
  only 1-16% of the best cutoff; the model's ranking is the gap.
- Model class: gradient boosting (depth 4) = logistic regression on the same features (TriviaQA 0.3 10%:
  38.4 vs 38.9). Richer features (10 best-fit balls, counts at every sampling rate in float, RaBitQ, all
  samples) help: TriviaQA 0.3 38.9 -> 42.3, ESCI 30.5 -> 32.0, SPACEV 0.3 14.0 -> 14.6, TripClick 19.9 ->
  20.5, TriviaQA 0.6 50.8 -> 52.2; the oracle is 57.6 / 37.0 / 20.8 / 24.0 / 66.0. The rest is information the
  features do not carry; hit_table_aug2.py adds the exact intruder count (from the query's own top-500) to
  see how much of it is the sample's noise (queued, 40 default tables, with ladder reruns that also add
  1-nearest / 2-nearest / best-fit-among-m-nearest ball variants and the rich-feature probe).
- Calibration miss (deployed warm-up splits, later requests): only TripClick, all four sizes (retention
  0.877-0.888). Random splits of the later requests are fine (0 of 200 draws below T - 0.01). Cause: the
  warm-up requests are easier than the later ones for a cache built from them (TripClick 1%: mean retention
  of the cache's answer 0.19 / 0.175 on the fit / cal splits vs 0.099 later; exact repeats of a cached
  center 2.6-2.8% vs 1.6%). Other workloads: equal. A shift, not the normal approximation.
- Calibration rule (threshold_selection.py, normal_fixed; weighted for audited hits): per threshold in a grid of
  100 fit-score quantiles, from the fit split's start threshold toward lenient: among calibration requests
  scoring >= lambda (n of them) lower bound = mean - z * sd / sqrt(n), z = 1.645 (delta 0.05); lambda = the last
  threshold before the first one whose bound is below T (fixed order: no multiplicity correction). Needs: (1)
  the normal approximation of a mean of n values in [0, 1] (n = admitted calibration requests, ~200 at a
  5% hit rate), (2) calibration requests drawn like the future ones. Chosen by us (SUPG / BARGAIN); Hoeffding
  (width sqrt(ln(1/delta) / 2n) = 0.039 at n = 1,000 vs 0.006 for sd 0.12) and empirical Bernstein were
  compared on the detection tests (normal >> kl > hoeffding ~ eb) and eb admits nothing below ~10-20k.
- Consistent comparisons (ratio of mean hit rates over a workload's sizes; the earlier mix of directions
  and of oracle vs calibrated numbers was confusing): whole cache vs best-fit entry only, oracle / calibrated:
  ESCI 0.64 / 0.12, TriviaQA 0.3 0.63 / 0.52, TriviaQA 0.6 0.72 / 0.55, SPACEV real 0.72 / 0.71, TripClick
  0.81 / 0.59, SPACEV 0.3 0.82 / 0.74, MedRAG 0.96 / 0.61, Zipf 0.99 0.89-0.96 / 0.77-0.92. Past top-k only vs
  default, calibrated / oracle: SPACEV 0.3 0.02 / 0.20, TriviaQA 0.3 0.23 / 0.45, TriviaQA 0.6 0.33 / 0.54, SPACEV
  0.6 0.32 / 0.38, ESCI 0.35 / 0.50, SPACEV real 0.46 / 0.46, MedRAG 0.66 / 0.71, Zipf 0.99 0.78-0.87 / 0.79-0.88,
  TripClick 0.89 / 0.87.
- DiskANN backend vs exact (ratio of means over sizes), oracle / calibrated: SPACEV real 0.89 / 0.86, SPACEV 0.3
  1.00 / 0.88, ESCI 0.96 / 0.91, TriviaQA 0.6 0.97 / 0.90, TriviaQA 0.3 0.99 / 0.92, others 0.97-1.00 / 0.96-1.00. The
  oracle barely moves while the calibrated rate falls: the loss is in the features (balls and shells built
  from DiskANN's lists, deep-500 recall 0.89 on TripClick), not in what is available to serve. Not yet split
  into labels-only vs structure-only.

#### 9al (cont.). Ball variants and the exact-count probe (11-19 02:15; 31 of 40 default tables)
Upload done 01:45 (88 objects, 9.0 GiB, sizes checked). Calibrated hit rate relative to the reference:
- 1 nearest center + count over all: 0.89 mean (ESCI 0.44, SPACEV 0.3 0.75, others 0.97-1.00); 2 nearest 0.90;
  3 nearest 0.91. Best fit among the 3 nearest centers: 1 ball 0.96, 3 balls 0.97 (ESCI 0.69-0.71); among the 10
  nearest: 1 ball 0.98, 3 balls 1.00 (ESCI 0.87, TriviaQA 0.3 1.16). So ESCI needs the radius-aware search over
  more than 10 centers; a center graph with fit among m = 10-64 is the candidate cut.
- Rich features (10 balls, counts at every sampling rate, float, RaBitQ): 1.11 mean (TriviaQA 0.3 1.65).
- Exact intruder count (not deployable, from the query's own top-500): 1.24 mean (TriviaQA 0.3 1.89, SPACEV 0.3 1.35,
  ESCI 1.28, TripClick 1.28, SPACEV real 1.11, Zipf 0.99 / MedRAG 1.01-1.02); one ball suffices with it. Most of the
  gap to the oracle is the sampled count's noise: better estimators of the intruder count (more or smarter
  samples, floats, unions of shells) are the main model-side lever.

#### 9al (cont.). All 200 tables and the dynamic validation done (11-19 12:50)
(I did not check the queues between 02:15 and 12:50; the machine was idle from the end of the dynamic runs,
roughly 06:00-12:50. Monitoring loops must stay inside long-running turns.)
- Shell share f (static, calibrated hit rate relative to f 0.85, ratio of means over sizes): f 0.95: TripClick
  0.83, TriviaQA 0.3 0.92, ESCI 0.94, TriviaQA 0.6 0.98, others 1.00-1.01; f 1.0 (no shell): TripClick 0.63,
  TriviaQA 0.3 0.78, ESCI 0.86, TriviaQA 0.6 0.90. One f does not fit every workload; f 0.95 costs TripClick 17%
  (the timing win at TripClick 10% came with 0.5 points less hit rate at that size only).
- Dynamic validation (per request, untimed, DiskANN backend, graph search; hit % / retention), v6 timing
  config vs V1 (4-bit robust range + inserts at 200) vs V2 (V1 + intruder count over the whole sample instead of
  the 10 best-fit balls' shells): ESCI 10% LRU 15.6 / 16.8 / 23.1; ESCI 10% continuous 23.7 / 24.7 / 30.5;
  SPACEV 0.3 1% continuous 2.2 / 3.8 / 5.6; TriviaQA 0.3 10% continuous 49.0 / 48.9 / 54.6; TripClick 10% LRU
  32.6 / 32.9 / 34.6, continuous 25.1 / 25.3 / 27.9; SPACEV real 10% LRU 8.6 / 8.5 / 9.1; SPACEV 0.99 1% +0.4 / +1.3.
  So the shell scope costs real hit rate dynamically (ESCI up to 8 points); the robust range and inserts at 200
  add 0.1-1.6 points. Retention stays 0.905-0.929.
- Timing batch 9al (timing_9al, 32 runs, 8 at a time, quiet machine, started 12:51): the eight cells above,
  LRU and continuous greedy, with fit-m (best fit among the 64 nearest centers), robust 4-bit range, inserts
  at 200 and the fast path; C1 counts over the whole sample, C2 over the 10 best-fit balls' shells.

#### 9al (cont.). First timing 9al results (11-19 14:15)
- C1 (count over the whole sample) is far too slow: hit path 6.4-13.6 ms per request vs 0.2-0.5 ms at 10 balls
  (SPACEV 0.99 1% LRU 0.41x vs 1.63x), though it gains hit rate. Unstarted C1 runs were skipped (marker only);
  C3 (intruder count over the shells of the 30 best-fit balls) replaces it in the full batch.
- C2 (10 balls; fit-m 64, robust range, inserts 200, fast path) vs v6, speedup (hit %): ESCI 1% continuous 1.061x
  (9.6) vs 1.049x (8.9); ESCI 10% continuous 1.228x (27.8) vs 1.172x (23.7); ESCI 10% LRU 1.056x (20.6) vs 0.953x
  (15.6); SPACEV 0.99 1% continuous 1.891x vs 1.854x; TripClick 1% continuous 1.052x (8.6) vs 1.032x (6.5); SPACEV 0.3 1%
  continuous 0.999x (3.8) vs 1.003x (2.2). Same speed, higher hit rate; ESCI 10% 1.228x still below node$ 1.258x.
- Queue: timing_9al.txt (32 runs, 22 done at 14:15), then timing9al_full.txt (128 runs: C2 for every other cell,
  C3 for all cells), quiet, 8 at a time.

#### 9al (cont.). Timing 9al, first batch complete (11-19 15:45)
Speedup (hit %), v6 vs C2 (fit-m 64, 10-ball count, robust range, inserts 200, fast path): TripClick 10% LRU
1.012x (32.6) -> 1.163x (33.6), 10% continuous 1.119x -> 1.136x, 1% LRU 1.119x -> 1.129x (18.0), 1% continuous
1.032x -> 1.052x (8.6); TriviaQA 0.3 10% LRU 1.648x (53.9) -> 1.792x (55.2), continuous 1.600x -> 1.631x; SPACEV real
10% LRU 1.056x -> 1.059x (9.0), continuous 1.057x -> 1.064x; ESCI 10% continuous 1.172x -> 1.228x, LRU 0.953x ->
1.056x. Hit path of SPACEV real is still 2.7-3.5 ms per request (not the count; to profile). C3 (30 balls) on SPACEV
0.6 1%: slower than v6 (0.932x LRU, 1.055x continuous) with +1.4 points of hit rate: the count is not worth
more than ~10 balls there. Two C1 runs (TriviaQA 0.3 10%, whole sample) stopped after 1.8 h by PID.
Full batch (timing9al_full.txt, 128 runs; C2 for the other cells, C3 for all) started 15:13, 8 at a time.

### 9am. Workload characterization (11-19 17:00)
Per workload (stream, requests after a 20,000-request warm-up; "near" = a first-time query whose exact top-10
shares >= 5 of its 10 ids with an earlier query's top-10; shares are of all requests after the warm-up):
| workload | DB | dim | requests / distinct | exact repeats | median repeat gap | near (>= 5 shared) |
|---|---|---|---|---|---|---|
| SPACEV Zipf 0.3 / 0.6 / 0.99 | 10M | 100 | ~100k / 98k, 94k, 65k | 2.4 / 6.6 / 37.6% | 30.6k / 24.0k / 4.5k | 54 / 52 / 28% |
| TriviaQA Zipf 0.3 / 0.6 / 0.99 | 10M (SPHERE) | 769 | ~100k / 89k, 81k, 47k | 13.1 / 22.0 / 56.5% | 28.6k / 21.0k / 3.0k | 50 / 45 / 21% |
| ESCI | 1M | 384 | 130,652 / 130,652 (random order) | 0% | - | 50% |
| SPACEV real log | 10M | 100 | 123k / 115k (Bing log, then official queries) | 7.3% | 9.0k | 3% |
| TripClick | 1.52M | 769 | 100k / 82.9k (2018 log) | 18.8% | 3.7k | 31% |
| MedRAG-Zipf | 23.9M | 769 | 10k / 10k (500 questions, each draw LLM-reworded) | 0% | - | 94% |

#### 9am (cont.). Why the oracle gap, and workload selection criteria (11-19 18:00)
- Oracle gap vs retention mix (default cache, mean over sizes; share of requests with retention >= 0.9 / 0.3-0.9 /
  < 0.3; calibrated / oracle): TriviaQA 0.3 .18 / .31 / .52, 0.58; SPACEV 0.3 .07 / .15 / .78, 0.62; ESCI .14 / .23 / .63,
  0.71; TripClick .08 / .13 / .80, 0.71; MedRAG .48 / .12 / .40, 0.97; SPACEV 0.99 .44 / .13 / .44, 0.96. The share of
  borderline requests does not predict the gap (correlation -0.04 over 40 cells; MedRAG has no exact repeats and
  sits at 0.97). What does: the exact intruder count closes the gap exactly where it is large (x1.77 TriviaQA 0.3,
  x1.35 SPACEV 0.3, x1.28 ESCI / TripClick, x1.01-1.03 MedRAG / Zipf 0.99): hits there are partial-overlap
  neighbors that only the count can tell from misses, and the sampled count is noisy.
- Every ladder variant refits the logistic model on its own features (and, for cache / backend variants, on its own
  cache) and recalibrates lambda on its own scores, per random split. The cache is static; requests are split
  iid at random from the post-warm-up stream and all are labeled.
- The 10-ball shells in the dynamic / timing runs are found by the core's fit-m mode: the 64 nearest centers (center
  graph), sorted by fit, best 10 (best 3 for the features). The static ladder's 10-ball scope took the best 10 of all
  entries; the combination with m = 64 is measured only in the timing / dynamic runs.
- Selection criteria for the paper's workloads (Jialin): Zipf settings of one family count as one workload; mix
  large wins (hit rate far above every alternative, speedup ~2x or more), moderate wins, and real-query
  workloads; 4 workloads are fine.

### 9an. Speedup against the best alternative, and cheap serving (11-19 18:10)
Ours / best alternative (Aker, QVCache or DiskANN with the cache's memory as node cache), best of our runs per cell
(v6 + 9al), speedup ratio: TriviaQA Zipf 1.09-1.53 (0.3: 1.09 / 1.51 at 1% / 10%; 0.6: 1.16 / 1.41; 0.99: 1.53 / 1.36); SPACEV
Zipf 0.94-1.29 (0.99: 1.07-1.29; 0.3 below 1 at 0.3-3%); TripClick 0.97-1.04; SPACEV real 0.98-1.01; ESCI 0.93-1.00 (v6;
1.23x vs node$ 1.26x at 10% with 9al); MedRAG 1.86 / 2.16 / 2.01 / 2.74 (0.01-0.3%). Only MedRAG exceeds 2x.
Where the time goes (TriviaQA 10%, DiskANN 74 ms): per hit 12-13 ms, of which stage-2 serving ~8-10 ms (graph search
at the tuned search list, 1024-2048, over a 1M-vector cache); per miss +2.5-4 ms (kNN + balls). SPACEV 0.99 10%:
per hit 0.84 ms, per miss +0.67 ms. Counterfactual TriviaQA 0.99 10% (86% hits): serving at ~1 ms would give ~5.4x
vs the best alternative 2.6x (2.1x). SPACEV 0.99 cannot reach 2x vs best alternative: its hit rate caps it near 2.7x
(Aker 1.8x).
Cheap serving: serve a hit from the exact k nearest among the members of the m best-fitting entries (at most m x K
vectors) instead of a search over the whole cache. Ladder column hit_table_aug3.py; TripClick 1% (calibrated hit
rate, whole cache 4.6%): m = 1 3.7% (27 vectors scanned), m = 3 4.3% (82), m = 10 4.6% (272). Core option
`set_serve_balls(n, ef1)`, `--core-serve-balls N`, unit test added (17 pass). aug3 runs on all 40 default tables
after the running timing jobs finish (timing runner held at 18:06, left held; its remaining C2 / C3 runs will be
replaced by a batch with union serving).

#### 9an (cont.). Union serving, ladder results (11-19 20:00; 34 of 40 tables for m = 30, 100)
Calibrated hit rate of serving from the union of the m best-fit balls, as a share of whole-cache serving (ratio,
mean over sizes; ESCI / MedRAG / SPACEV 0.3 / 0.6 / 0.99 / SPACEV real / TripClick / TriviaQA 0.3 / 0.6 / 0.99):
m = 1 .07 .58 .73 .86 .92 .74 .73 / m = 3 .42 .95 .80 .93 .95 .83 .86 .70-.93; m = 10 .68 1.00 .86 .94 .97 .89 .95 (all
cells .89); m = 30 .79 1.00 .89 .96 .98 .93 .97 .99-1.01 (.94); m = 100 .92 1.00 .90 .97 .99 .96 .99 .99-1.00 (.97).
Vectors scanned per request, m = 30 / 100: TripClick 1.7k / 5.4k, TriviaQA ~1.9k / 6k, ESCI 2.4k / 5.7k, MedRAG
2.7k / 7.8k, SPACEV 8.5k / 26.5k (100-d). At ~1 ms this replaces ~10 ms of stage-2 search on TriviaQA 10%.
Timing batch 9an (timing9an.txt, 128 runs): C4 = 9al config + serve from the 30 best-fit balls (fit-m 64), C5 = 100
balls (fit-m 128), LRU and continuous greedy, all 32 non-sensitivity cells, 10% first; starts when the aug4 jobs
finish; the held 9al full batch (28 of 128 done) is dropped.

#### 9an (cont.). Segfault in the first 9an batch, restart (11-19 23:05)
The first union-serving batch lost most runs silently: exit status 139 (SIGSEGV) during warm-up on larger caches.
Cause: my optimization of union_serve shared a stamp array between threads, and lookup_batch (features, warm-up)
runs requests in parallel. Fixed with thread-local stamps (a global atomic counter); tests: serving vs a NumPy
reference, and batch lookups with 8 threads vs the single path (18 pass). run_jobs prints "done" for any exit,
so a missing DONE marker is the failure signal. All C4 / C5 results before 23:05 are discarded and rerun.
First numbers before the restart (valid, single-request path): MedRAG 0.3% LRU C4 2.76x (hit path 3.3 ms; serve 2.7 ms
at 30 balls, 8.9 ms at 100 balls, vs 0.3 ms for the stage-2 graph on that small cache): union serving only pays on
large caches (graph serve ~9 ms at TriviaQA 10%); rule candidate: use the cheaper serving per cache size.
TriviaQA 0.99 1% LRU C4: 2.40x vs 2.24x (hit path 1.67 vs 3.26 ms).

#### 9an (cont.). First union-serving timing results (11-20 00:25)
TriviaQA 0.99 (speedup, hit %): 10% LRU: v6 3.47x (86.7), C2 (graph stage 2) 4.77x, C4 (30 balls) 4.42x (83.0, hit path
5.3 ms vs 12.1), C5 (100 balls) 2.91x (hit path 14.4 ms); 10% continuous C4 3.59x (82.0); 1% LRU 2.24x -> C4 2.43x,
C5 2.32x; 1% continuous 2.59x -> C4 2.73x, C5 2.51x. Union serving trades hit rate (86.7 -> 83.0%) for serve time and
is not a clear win over the C2 stage-2 graph; 100 balls is slower than the graph (14 ms). Best vs best alternative so
far: TriviaQA 0.99 10% 1.85x (C2 LRU 4.77x vs QVCache 2.59x), TriviaQA 0.6 10% 1.51x (C3 LRU 2.04x vs Aker 1.35x), TriviaQA
0.3 10% 1.51x, MedRAG 1.86-2.74x (v6), everything else <= 1.5x. Ceiling argument: a cell can only reach 2x vs the best
alternative if 1 / (1 - hit rate) of ours >= 2 x the alternative's measured speedup with the hit path ~free: TriviaQA 0.99
10% (91% hits vs 60%), 0.6 10% (71% vs 27%), MedRAG; not SPACEV (hit rate 63% -> 2.7x cap vs Aker 1.8x), TripClick, ESCI.

### 9ao. Why TripClick / ESCI cannot reach 2x, memory accounting, and config names (11-20 01:00)
- A result cache serves at most a share h of requests, so with free hit decisions and free serving its mean latency is
  (1 - h) x the backend's: speedup <= 1 / (1 - h), whatever the backend's latency. Versus an alternative with hit rate
  h_a (also free): ours / alternative <= (1 - h_a) / (1 - h). Caps from the headline hit rates (ours best / best alternative):
  TripClick 0.3 / 1 / 3 / 10%: 1.17 / 1.22 / 1.31 / 1.52x vs Aker 1.10 / 1.13 / 1.16 / 1.22x (ratio cap 1.06 / 1.08 / 1.13 / 1.24);
  ESCI: 1.07 / 1.12 / 1.21 / 1.49x vs DiskANN with the cache's memory as node cache 1.05 / 1.09 / 1.17 / 1.26x (ratio cap
  1.02 / 1.02 / 1.04 / 1.19). Doubling our hit rate over Aker's (34 vs 18% at TripClick 10%) is a 1.24x cap because the
  speedup is 1 / (1 - h), not h. A slower backend does not change the caps; it only lets the measured speedup approach them.
- Aker's memory: the paper (Sec. 4) says each neighbor object stores "the raw vector and database-specific metadata"
  and the cache size is controlled by capping the total number of neighbor objects; evaluated at 1% and 5% of the
  dataset's vector count. The code (third_party/aker, VectorSlot / VectorSlotPool) allocates a payload of the vector's size
  per pooled neighbor; the cache logic in src/core never reads the payload, only vector ids and the cached query
  vectors; the payload is exposed to the host through the C wrapper. So the hit path needs only ids: we charge Aker one
  full vector per neighbor object (pool = budget x N), which at equal BYTES under-fills it (ids 8 B vs 100 B SPACEV int8,
  3 KB TriviaQA / TripClick float32). Whether the authors intended the raw payload (host integration) is not stated.
  Our own accounting charges cached vectors at full size and the shell sample at b bits; entry centers, id lists, the
  graph and the model are not charged (< ~10%).
- Config names (plain words; codes are retired from reports): "v6" = first full timing run; "tuned" = v6 + robust 4-bit
  range, graph inserts at efConstruction 200, exact-repeat fast path, best fit among the 64 nearest centers, intruder
  count over the 10 best-fit balls (= C2); C3 = tuned with the count over 30 balls; C4 / C5 = tuned with a hit answered
  from the vectors of the 30 / 100 best-fit balls instead of a search over the whole cache.

#### 9ao (cont.). Metrics, bigger caches, Aker paper (11-20 02:00)
- The Aker paper never says cached vectors can be dropped: "result-centric caches store only the past query vectors
  and their top-k results"; neighbor objects store "the raw vector and database-specific metadata"; its memory
  claims are the sharing of neighbor objects between entries and a small query filter (26-276 MB, 2.5-5.4% of
  DiskANN's pinned index). Cache sizes of 1% / 5% count neighbor objects. Decision (Jialin): keep the budget in
  vectors (the paper's own unit); the paper must state that Aker's entries could be stored as ids only.
- Candidate metrics beyond mean speedup (tuned ours vs best alternative, TripClick 10% / 1%, ESCI 10%, TriviaQA 0.6 10%,
  SPACEV 0.99 10%): (a) memory the best alternative needs to match our hit rate: TripClick 13x / 9x at 0.3% / 1%, TriviaQA 0.6
  4.5x / 4.3x at 0.3% / 1%, ESCI / SPACEV / MedRAG never reach it within 10x-33x; (b) mean latency at a fixed offered load
  (backend as a queue, misses see (S + wait)): TripClick 10% vs Aker 1.18x / 1.48x / 1.69x at 50 / 80 / 90% of the uncached
  capacity (M/M/1; M/D/1 1.10 / 1.31 / 1.46), TripClick 1% 1.11 / 1.27 / 1.40, ESCI 10% vs node$ 1.09 / 1.22 / 1.31, TriviaQA 0.6
  10% 1.92 / 2.61 / 3.03, SPACEV 0.99 10% 1.40 / 1.61 / 1.70; vs no cache 1.7-15x. Hit-rate cap 1 / (1 - h) at zero load.
- Larger TripClick caches queued (20 / 30 / 50% of the dataset): static tables, Aker (4 proximity settings), ours LRU and
  continuous greedy per request (untimed); ESCI static tables at 30 / 50%.

### 9ap. Throughput (closed loop, real DiskANN misses) (11-20 04:30)
Question (Jialin): do we get larger gains in throughput than in workload time? What do Aker / QVCache measure?
- Aker (paper, Sec. 5): the client issues requests from a single thread ("The benchmark runs with a single
  thread"); QPS is therefore 1 / mean latency, the same as our workload-time speedup; concurrency is left to future work
  (global lock). QVCache (arXiv 2602.02057): hit ratio, hit latency, p50 latency (p99 omitted: "unless the hit ratio is above
  99%, P99 is dominated by queries that miss"), throughput ("we report the metric reflecting this benefit"), 10-recall@10,
  vectors retrieved; headline "p50 latency up to 40-1000x lower". Neither reports latency under load.
- Latency at fixed load: at load 0 it equals the workload-time speedup; both systems' waits grow toward saturation; ours vs an
  alternative grows with load when our hit rate is higher (waits scale with 1 / (1 - load x miss share)).
- Real DiskANN on this machine's volume is I/O bound: TripClick, search list 128 / beam 32, base node cache, batch_search
  from 1 to 32 threads: 87.7 queries/s at every thread count (~170 reads per query at ~15,000 reads/s). DiskANN's
  Python binding holds the GIL (search: a Python loop keeps 34% of its speed during searches; batch_search: ~1%), so the
  replay below uses backend worker processes.
- throughput_replay.py: replays a timed cache run's per-request decisions (hit or miss) and measured cache-path times (online_sim
  --timing-every 1 now records dt_ms) in closed loop: one cache thread (serial, busy-waits the measured time), misses go
  to 8 DiskANN worker processes (real searches, no node cache, same for every arm), at most C requests in flight.
  TripClick 10%, 20,000 requests, queries per second (hit rate): no cache 67.0; ours (tuned LRU) 99.8-100.5 (33-34%);
  Aker (proximity 0.05) 80.5-80.6 (17%); identical at C = 4 / 16 / 64. Ours / Aker = 1.25x (cap 1.24x), ours / none 1.50x
  (cap 1.52x), vs 0.97x and 1.16x in single-request mean latency. p50 latency at C = 4: 54.2 / 57.5 / 59.1 ms (ours / Aker / none).
  ESCI 10%, C = 16: no cache 75.9; ours (tuned continuous greedy, 27.9% hits) 104.6 = 1.38x (cap 1.39x).
- DiskANN with the cache's memory as node cache, same method (batch_search, 16 threads, 3,000 queries): TripClick 87.7 (base 1,828
  nodes) -> 116.6 queries/s (154,753 nodes) = 1.33x; ESCI measured next.

ESCI node cache, same method (16 threads): base 300 nodes 95.7 queries/s -> 100,479 nodes (the cache's memory at 10%) 184.0 =
1.92x, against ours 1.38x in the replay: on ESCI the larger node cache beats our result cache in throughput by more than
in single-request latency (1.26x vs 1.17-1.23x). TripClick: node$ 1.33x < ours 1.50x. Caveats of the replay: cache-path
times are replayed from a separate run (no contention with backend processes), one cache thread, backend workers without
node cache, hit / miss decisions fixed from the recorded run (no feedback from completion order).

#### 9ap (cont.). Larger TripClick caches, throughput queue (11-20 14:30)
- TripClick at 20 / 30 / 50% (untimed, per request, tuned settings, graph search lists of the 10% cell): ours LRU 44.6 / 56.0 / 75.1%
  (retention .911 / .912 / .909), continuous greedy 36.2 / 48.4 / 70.7%; Aker best (proximity 0.05) 21.9 / 24.2 / 24.7% (saturates
  near 25%). Hit-rate caps 1/(1-h): ours 1.81 / 2.27 / 4.02x vs Aker 1.28 / 1.32 / 1.33x (ratio caps 1.4 / 1.7 / 3.0x). Static tables:
  TripClick calibrated 32.5 / 40.2 / 52.5%; ESCI at 30 / 50%: 57.2 / 78.1% (caps 2.34x / 4.57x).
- Latency terms: single-request mean latency (unloaded) ours 1.163x vs Aker 1.201x speedup at TripClick 10% (ours 3% slower);
  in the closed-loop replay at 4 clients, p50 54.2 vs 57.5 ms and mean 39.6 vs 49.7 ms are queueing-dominated latencies
  (4 requests always in flight), a different quantity.
- Queue (started 14:20): 81 timed recording runs (every request timed; ours LRU and continuous greedy tuned, best Aker,
  best QVCache; 21 cells: TripClick 1 / 3 / 10 / 20 / 30 / 50%, TriviaQA 0.3 / 0.6 / 0.99 at 1 / 10%, SPACEV 0.3 / 0.6 / 0.99 at 1 / 10%,
  MedRAG 0.03 / 0.1 / 0.3%) 8 at a time, then throughput_all.py: closed-loop replays (16 clients, 12,000 requests) and the
  equal-memory node-cache arm for each cell, sequential on a quiet machine.
