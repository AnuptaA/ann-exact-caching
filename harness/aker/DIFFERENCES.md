# Aker as run here versus the paper and the released code

Library: sjoon-oh/aker `release` e7639a5, vendored in third_party/aker (patches in
third_party/aker/PATCHES.md). Driver: harness/cpp/ak_replay.cc.

## Revision
- The paper's pinned commit 9f0c6c2 predates the refactor: it has no Proximity compile mode
  and the C wrapper has an old name. `release` has both. The behavior of the two was not
  compared line by line; the rules that matter for Q1 are covered by tests
  (tests/harness/test_aker_rules.py).

## Rules (verified by test on hand-made geometry)
- Standard mode: per-entry threshold starts at minD / 4 (hard-coded in the library), the hit test
  uses min(minD, threshold), an approximate hit multiplies it by alpha_T, an equality hit by
  alpha_L, both capped at minD. Matches the paper (Sec. 4.1). `global_thresh` is unused.
- Proximity mode: hit iff distance < global_thresh; no minD cap, no adaptation.
- Units: squared L2. Slot distances are squared Euclidean (from the answer table), the hit test
  uses the library's own squared L2, so minD/4, the caps and global_thresh are squared. The
  paper's pgvector integration also passes squared L2 (pgvector's L2 HNSW distance; no sqrt in
  its patch). Percentile thresholds for Proximity are percentiles of pairwise squared distances.

## Differences from the paper (configuration or patch)
- Query filter: hard-coded m 4, efSearch 8, efConstruction 16 in the library (paper: efConstruction 8,
  efSearch 16). Now configuration (patch 3); the harness sets the paper's values.
- alpha values: repository bootstrap configs use alpha_tighten 0.9; the harness uses the paper's
  alpha_T 0.25 and alpha_L 1.1. top_delta 0, FIFO eviction.
- StackFloatBuffer freed its alloca buffer on constructor return (patch 5): with clang on macOS the
  converted query vector was garbage, so approximate lookups never hit. Whether upstream's Linux
  builds were affected, and by how much, is unknown.
- std::random_shuffle replaced by std::shuffle (patch 4, unused code path).
- ak_ycsb_gen and the YCSB-C based sequence generator are not built (streams come from workloads).

## Behavior of the library that shapes results
- Pool: capacity.pool_size counts pooled vectors (neighbor objects), reference counted by vector
  id, so a vector shared by several entries counts once. Representative query vectors are separate
  clones and are not counted. Eviction (FIFO over entries) triggers when current size + incoming list
  size > pool_size (it assumes every incoming vector is new, so it evicts early when vectors are
  shared) and counts only vectors actually freed, so it can evict more entries than strictly needed.
  The budget rule is exact: pool_size must be 1%, 5% or 10% of N (harness.config.pool_size_for).
- Linked hits exist only in standard mode: after an approximate hit the driver creates a dummy entry
  and links it to the representative (as the paper's integration does), so a repeat of that query is
  an exact (linked) hit. In Proximity mode linking is a no-op, so repeats are approximate hits again.
- On a hit the answer is the matched entry's stored list (no re-ranking by distance to the new
  query); it is scored by its true recall against the exact top-10, never by the system's claim.
- Equality versus approximate is classified by our definition (the request's vector id appeared in
  an earlier request); the library's own exact/linked flag is recorded alongside and the count of
  disagreements is stored in every result header (`kind_mismatch`).
- Not used: the write log (risk_thresh), Potluck mode, dropout.
- The driver stores real float32 vectors in the pool slots (the library needs payload bytes equal to
  vector_in_bytes), so the base must be readable; it is memory mapped.
