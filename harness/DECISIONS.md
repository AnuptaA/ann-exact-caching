# Q1 decisions, deviations and findings

Last updated with the EC2 preparation. Differences from the Aker paper and the library are in
`harness/aker/DIFFERENCES.md`; local library patches are in `third_party/aker/PATCHES.md`.

## Decided by the project owner
- Aker `release` revision (e7639a5), C++ replay driver, squared L2 into Aker, plain Euclidean everywhere else.
- Proximity = Aker's fixed-threshold mode at the 0.1% and 1% percentiles of inter-query distances
  (squared L2, distinct queries of the first 30K requests of each stream).
- Budgets are exactly 1%, 5% or 10% of N in pooled-vector units; any other value fails.
- K = 10; a returned vector is correct if its distance is at most the exact 10th distance * (1 + 1e-6);
  recall always divides by 10.
- TriviaQA queries use Sphere's DPR query encoder (not normalized); the evidence log is informational.
- Equality hit (strict): served by an entry stored under this exact vector id. Filter hits and linked
  repeats are approximate.
- Stale exact-caching code lives in `stale/`. Native PostgreSQL 16 + pgvector v0.8.0 on EC2.
- The list length L (neighbors stored per query, `Q1_LIST_SIZE`, default 50) is provisional: it is decided
  after a microbenchmark of entry sizes {1,2,5} x K versus {1,2,5,10} x K versus the old K sweep.
  Changing L reruns only the `answer_table` and `backend_recall` stages and the cheap `q1-run-*` cells.

## Decided during implementation (approved unless noted)
- Drivers are built by `harness/cpp/CMakeLists.txt` (two static executables, Release, warnings off,
  `ak_sequence_gen` excluded). Pool slots hold real float32 vectors. Linking after an approximate hit is
  reproduced in standard mode (a library no-op in Proximity).
- Ground truth: GPU candidates top-(100+28), CPU float32 recompute, sort by (distance, id). With more than 28
  exact ties at the 100th distance the kept ids can differ; distances stay exact.
- Answer tables are re-sorted by recomputed float32 distance.
- Universe members are not re-normalized after interpolation (Aker's procedure; ESCI member norms are about
  0.84 on average; the L2 ranking over unit-norm products does not depend on the query norm).
- The Zipf rank permutation depends only on the stream seed, so the same groups are popular for all exponents.
- ESCI official queries are the 130,193 distinct query texts (130,652 ids); the base is all 1,814,924 products.
- TriviaQA: train + validation + test, de-duplicated by stripped text: 96,005 questions.
- `ef_search` is tuned on a 2K sample of official queries (ground truth reused), grid 10..400, rule
  max(smallest ef reaching recall 0.9, L).
- Proximity thresholds are computed once per (dataset, stream, seed) and shared across budgets; the
  universe-wide percentile is stored as a diagnostic.
- PostgreSQL: shared_buffers about 25% of RAM, wal_level minimal, parallel maintenance workers 7;
  `maintenance_work_mem` is sized per build so the HNSW graph fits (the build fails on pgvector's
  "graph no longer fits" notice). Binary COPY for the load.
- Results store, per request, the outcome code, equality and approximate flags and recall (not the returned ids).
  Results go to `$Q1_DATA/results/<commit>/`; a result from another commit is redone.
- The local macOS build used FAISS 1.15.1 headers; the EC2 script uses the system package or v1.8.0.

## Findings
- Upstream `StackFloatBuffer` freed its `alloca` buffer on constructor return, so converted query vectors were
  garbage and approximate lookups never hit with clang on macOS (patch 5). Linux impact unknown.
- The query filter's efSearch/efConstruction (8/16) are swapped relative to the paper (16/8); now configurable.
- The paper's pinned commit has no compile-time Proximity option (runtime flags not checked); `release` does.
- Linked hits exist only in standard mode.
- pgvector 0.8.0's HNSW L2 distance function is `vector_l2_squared_distance` (confirmed on a real server).
- Aker's TriviaQA payloads are bge-base-en-v1.5 raw CLS; sphere-10m is not bge. Hermes' TriviaQA file has
  96,005 rows, equal to our distinct question count.

## Not verified yet
Linux build and `setup_instance.sh`, CUDA paths, Python 3.10, behavior and memory at 1M-10M scale, agreement
with the paper (`make q1-fidelity`), DPR_web context encoder reproducing the sphere vectors, Hermes'
TriviaQA encoder, whether L = 50 is enough for the final learned system.

## L microbenchmark (2026-11-07, ESCI 50K, K=10, T=0.9, 5K candidates, 20K scoring queries, seed 42)
`populate esci` then `eval final` (7-feature logistic model, calibrated threshold), one run at a time.
- Entry sizes {1,2,5} x K (10,20,50) versus {1,2,5,10} x K (10,20,50,100): oracle and 7-feature hit rates differ
  by at most 0.003 at matched cache sizes up to 30% (noise); only the balls-only baseline gains about 0.01.
  Populate 9.2 min / 1.4 GB versus 16.0 min / 1.6 GB; eval final about 4 min each.
- The old set (10..500, 30K candidates) was stopped after 70 minutes without finishing.
- Shell depth R (needs L >= R) on {1,2,5} x K, hit rate of the 7-feature model at matched cache size:
  10% cache: R=50 0.240, 100 0.255, 200 0.264, 500 0.264; 20%: 0.357, 0.385, 0.397, 0.406; 1%: 0.091, 0.101, 0.100, 0.110
  (balls-only 0.192 at 10%). Memory-fair (shell sample charged to the cache): R=200 and R=500 gain about the
  same over balls-only at equal total memory (10%: +5.5 and +4.4 points, 20%: +6.8 and +6.9); R=50 gains less
  above 5% (20%: +4.4).
- Reading: {1,2,5} x K is enough for entries; the shell depth is what sets L. R=200 captures the memory-fair
  gain of R=500 on this workload, so L = 200 looks sufficient; L = 500 buys nothing memory-fair. One workload and
  one seed: confirm on SIFT1M before fixing `Q1_LIST_SIZE`.
