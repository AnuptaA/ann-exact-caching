#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .distance_metrics import DistanceMetric
from .kv_cache import CacheEntry

#-------------------------------------------------------------------------------

@dataclass
class CandidateEntry:
    query: np.ndarray
    r_Q: float
    alpha: Optional[float]             # None for point-like and ball-like proposals
    r_c: float                         # used as cache radius at hit time
    R_c: object                        # bool array (len n_base) when store_rc=True, else None
    rc_size: int                       # R_c.sum(); always set regardless of store_rc
    covered_train_indices: object      # bool array (len n_train_queries) — observed indices satisfying CIG
    covered_test_indices: object       # bool array (len n_test_queries) — held-out test indices satisfying CIG
    top_k_vectors: List[np.ndarray]
    top_k_indices: List[int]
    half_gap: float
    source_index: Optional[int] = None  # index into observed_queries this candidate was proposed from
    n: Optional[int] = None             # neighbor rank (1..n_max) this candidate's sphere was built from

    def to_cache_entry(self) -> CacheEntry:
        # CIG soundness requires the stored set to be the COMPLETE set of base vectors
        # within r_c of the query, i.e. R_c -- not the (possibly much smaller) top_k set
        # loaded at the eval K.
        if self.R_c is None:
            raise ValueError(
                "to_cache_entry requires R_c; use store_rc=True in propose_candidates"
            )
        # int32 (not a Python list of boxed ints): at real scale (tens of
        # thousands of entries with ~44% R_c fill over a 50k+ base) a list
        # of boxed ints costs ~9x more memory than a compact int array and
        # was a direct contributor to OOM kills at high budgets.
        indices = np.flatnonzero(self.R_c).astype(np.int32)
        # Index-only: no per-entry top_k_vectors/top_k_matrix copy. Candidates'
        # R_c sets frequently overlap heavily (by design, for nested spheres),
        # so a private per-entry copy would duplicate the same base vectors
        # across many entries. Distance computation and result-vector lookup
        # instead go through a shared base_hat/base_vectors matrix passed at
        # hit-check time (see CacheSimulator, lemma1_no_union_sigma/
        # lemma2_no_union_sigma).
        return CacheEntry(
            query=self.query,
            top_k_vectors=[],
            r_Q=self.r_c,
            half_gap=self.half_gap,
            top_k_indices=indices,
            top_k_matrix=None,
        )

#-------------------------------------------------------------------------------
# Shared helpers
#-------------------------------------------------------------------------------

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def _cig_mask(
    q_hat: np.ndarray,
    Q_hat: np.ndarray,
    r_t: np.ndarray,
    r_c: float,
    metric: DistanceMetric,
) -> np.ndarray:
    if metric == "angular":
        dots = np.clip(Q_hat @ q_hat, -1.0, 1.0)
        dists = np.arccos(dots) / np.pi
    else:
        dists = np.linalg.norm(Q_hat - q_hat, axis=1)
    return (dists + r_t) <= r_c


def _raw_pairwise_dists(vecs_hat: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Full (n, n) distance matrix among pre-normalized vecs_hat, diagonal
    left as the real (near-zero) self-distance, not masked.

    Shared by _pairwise_cache_dists (which sorts and masks the diagonal to
    inf) and precompute_ball_invariants (which reuses this directly as
    train_dist_only when train_queries is observed_queries) so there's one
    canonical value for this matrix instead of two independently-rounded
    matmuls that could disagree by a few ULPs at a certification boundary.
    """
    if metric == "angular":
        sims = np.clip(vecs_hat @ vecs_hat.T, -1.0, 1.0)
        return np.arccos(sims) / np.pi
    n = len(vecs_hat)
    dists = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dists[i] = np.linalg.norm(vecs_hat - vecs_hat[i], axis=1)
    return dists


def _pairwise_cache_dists(
    cache_queries: np.ndarray,
    metric: DistanceMetric,
) -> tuple:
    """Pairwise distances among cache queries, sorted per row, self excluded.

    Returns (sorted_dists, sorted_indices) both shape (n_cache, n_cache - 1).
    sorted_dists[c, n-1] = distance from query c to its n-th nearest neighbor.
    sorted_indices[c, n-1] = index of that neighbor.
    """
    cache_hat = _normalize(cache_queries.copy())
    dists = _raw_pairwise_dists(cache_hat, metric).copy()
    np.fill_diagonal(dists, np.inf)
    order = np.argsort(dists, axis=1)
    sorted_dists = np.take_along_axis(dists, order, axis=1)
    # Remove the inf column (self): after sort, inf is last; take all but last
    sorted_dists = sorted_dists[:, :-1]
    sorted_indices = order[:, :-1]
    return sorted_dists, sorted_indices


def _base_dists_for_query(
    q_hat: np.ndarray,
    base_hat: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    if metric == "angular":
        return np.arccos(np.clip(base_hat @ q_hat, -1.0, 1.0)) / np.pi
    return np.linalg.norm(base_hat - q_hat, axis=1)


def _precompute_kth_base_dist(
    cache_hat: np.ndarray,
    base_hat: np.ndarray,
    k: int,
    metric: DistanceMetric,
    batch_size: int = 32,
) -> np.ndarray:
    """k-th nearest base distance for each cache query, computed in batches.

    Requires k >= 1: np.partition(d_batch, k-1, ...) at k=0 becomes
    partition(..., -1, ...), which extracts the *maximum* distance (last
    position after partitioning), not zero or anything meaningful. The K=0
    case (ball reduces to point) is handled by the caller
    (propose_candidates_ball) without calling this function at all -- see
    its docstring.
    """
    assert k >= 1, f"_precompute_kth_base_dist requires k >= 1, got k={k}"
    n_cache = len(cache_hat)
    d_kth = np.empty(n_cache, dtype=np.float32)
    for start in range(0, n_cache, batch_size):
        end = min(start + batch_size, n_cache)
        batch = cache_hat[start:end]
        if metric == "angular":
            d_batch = np.arccos(np.clip(batch @ base_hat.T, -1.0, 1.0)) / np.pi
        else:
            d_batch = np.array([
                np.linalg.norm(base_hat - batch[j], axis=1)
                for j in range(len(batch))
            ], dtype=np.float32)
        # partition is cheaper than full sort for large base
        d_kth[start:end] = np.partition(d_batch, k - 1, axis=1)[:, k - 1]
    return d_kth

#-------------------------------------------------------------------------------

def propose_candidates(
    observed_queries: np.ndarray,
    observed_r_Q: np.ndarray,
    observed_top_k_vectors: List[List[np.ndarray]],
    observed_top_k_indices: List[List[int]],
    observed_half_gaps: np.ndarray,
    train_queries: np.ndarray,
    train_r_t: np.ndarray,
    test_queries: np.ndarray,
    test_r_t: np.ndarray,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
    alpha_values: List[float] = (),
    store_rc: bool = True,
) -> List[CandidateEntry]:
    """
    Generate candidate cache entries by expanding each observed query ball.

    For each (observed_query, alpha) pair:
      - r_c = alpha * r_Q
      - R_c = base vectors within r_c
      - covered_train_indices = observed queries satisfying D(t, q) + r_t <= r_c (oracle CIG)
      - covered_test_indices  = held-out test queries satisfying the same condition

    A candidate is kept only if covered_train_indices is non-empty (source query excluded).
    """
    train_hat = _normalize(train_queries.copy())
    test_hat = _normalize(test_queries.copy())

    base_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_norms[base_norms == 0.0] = 1.0
    base_hat = base_vectors / base_norms

    candidates: List[CandidateEntry] = []

    for i in range(len(observed_queries)):
        r_Q = float(observed_r_Q[i])

        q_norm = float(np.linalg.norm(observed_queries[i]))
        q_hat = observed_queries[i] / q_norm if q_norm != 0.0 else observed_queries[i].copy()

        all_dists = _base_dists_for_query(q_hat, base_hat, metric)

        for alpha in alpha_values:
            r_c = alpha * r_Q

            R_c_mask = all_dists <= r_c
            rc_size = int(R_c_mask.sum())
            R_c = R_c_mask if store_rc else None

            train_covered = _cig_mask(q_hat, train_hat, train_r_t, r_c, metric)
            train_covered[i] = False
            if not train_covered.any():
                continue

            test_covered = _cig_mask(q_hat, test_hat, test_r_t, r_c, metric)

            candidates.append(CandidateEntry(
                query=observed_queries[i],
                r_Q=r_Q,
                alpha=alpha,
                r_c=r_c,
                R_c=R_c,
                rc_size=rc_size,
                covered_train_indices=train_covered,
                covered_test_indices=test_covered,
                top_k_vectors=observed_top_k_vectors[i],
                top_k_indices=observed_top_k_indices[i],
                half_gap=float(observed_half_gaps[i]),
            ))

    return candidates

#-------------------------------------------------------------------------------

def precompute_ball_invariants(
    observed_queries: np.ndarray,
    train_queries: np.ndarray,
    test_queries: np.ndarray,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
) -> dict:
    """K-independent invariants shared across a K-sweep for the same
    observed/train/test/base arrays -- see _propose_nested_spheres, which
    accepts these via its optional `invariants` parameter.

    Every observed query's distance to every base vector / train query /
    test query, and the cache queries' pairwise nearest-neighbor ordering,
    do not depend on K at all (only margin_lookup and the K-specific
    train_r_t/test_r_t do). Computing them once and reusing across
    multiple K values (e.g. debug_trajectory.py's K in [0,1,5,10] sweep
    per benchmark) avoids redoing this work once per K for no reason --
    confirmed via profiling (SIFT c2048/K=5: _base_dists_for_query alone
    was 4.1s of propose_candidates_ball's 12.8s, entirely K-independent,
    and was being recomputed on every one of the 4 K values swept for
    that benchmark).

    Memory: all_dists is (n_cache, n_base) -- the big one. At the largest
    real scale in this codebase (SIFT/ESCI c2048, n_base=50000): ~410MB
    in float32. Held only for the duration of whatever the caller's
    K-sweep is, by callers that opt in -- every existing call site is
    unaffected (this function is never called unless a caller explicitly
    passes its result to propose_candidates_ball's invariants parameter).
    """
    obs_hat = _normalize(observed_queries.copy())
    base_hat = _normalize(base_vectors.copy())

    # Raw (n_cache, n_cache) matrix computed once; sorted_cache_dists/
    # sorted_cache_indices derive from a copy of it, and train_dist_only
    # reuses it directly when train_queries is observed_queries (true for
    # every real caller) instead of a second, independently-rounded matmul.
    raw_cache_dists = _raw_pairwise_dists(obs_hat, metric)
    dists_for_sort = raw_cache_dists.copy()
    np.fill_diagonal(dists_for_sort, np.inf)
    order = np.argsort(dists_for_sort, axis=1)
    sorted_cache_dists = np.take_along_axis(dists_for_sort, order, axis=1)[:, :-1]
    sorted_cache_indices = order[:, :-1]

    def _dist_matrix(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
        dots = np.clip(a_hat @ b_hat.T, -1.0, 1.0)
        if metric == "angular":
            d = np.arccos(dots) / np.pi
        else:
            d = np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))
        return d.astype(np.float32)

    if train_queries is observed_queries or np.array_equal(train_queries, observed_queries):
        train_dist_only = raw_cache_dists.astype(np.float32)
    else:
        train_hat = _normalize(train_queries.copy())
        train_dist_only = _dist_matrix(obs_hat, train_hat)

    test_hat = _normalize(test_queries.copy())

    return dict(
        sorted_cache_dists=sorted_cache_dists,
        sorted_cache_indices=sorted_cache_indices,
        all_dists=_dist_matrix(obs_hat, base_hat),
        train_dist_only=train_dist_only,
        test_dist_only=_dist_matrix(obs_hat, test_hat),
    )


def _propose_nested_spheres(
    observed_queries: np.ndarray,
    observed_r_Q: np.ndarray,
    observed_top_k_vectors: List[List[np.ndarray]],
    observed_top_k_indices: List[List[int]],
    observed_half_gaps: np.ndarray,
    train_queries: np.ndarray,
    train_r_t: np.ndarray,
    test_queries: np.ndarray,
    test_r_t: np.ndarray,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
    n_max: int,
    margin_lookup: np.ndarray,
    store_rc: bool,
    invariants: Optional[dict] = None,
) -> List[CandidateEntry]:
    """
    Shared nested-sphere candidate construction for propose_candidates_ball
    (including its K=0 case, previously a standalone propose_candidates_point
    function -- removed in Phase 3 Round 12, see docs/POPULATION_DESIGN.md).

    For each observed (cache) query i and each n in [1, n_max] (its n-th
    nearest cache neighbor), proposes a candidate with radius
        r_c(i, n) = D(query_i, neighbor_n) + margin_lookup[neighbor_n]
    margin_lookup[neighbor_n] is uniformly "distance from neighbor_n to its
    own K-th nearest base vector" for every K, including K=0 (trivially 0 --
    a 0-nearest-base-vector ball has no radius). No epsilon term: the
    named-neighbor coverage check below uses margin_lookup[nn_index]
    directly (the exact value r_c was built from), so it holds by
    construction, not by floating-point luck -- see the check itself.

    All n_max spheres for the same query are proposed independently; a sphere
    is dropped only if its own covered_train_indices is empty (today's
    existing per-candidate pruning rule, now evaluated per n). No dominance/
    subsumption pruning is applied between nested spheres of the same query --
    a smaller sphere is left in the candidate pool even once a larger sphere
    for the same query is selected, since it can never win select_candidates'
    cost-efficiency scoring over the larger one it's subsumed by.

    Per-query invariants that do not depend on n (all_dists, train/test
    threshold distances) are computed once per query and reused across all
    n_max spheres, rather than recomputed per n.

    invariants: optional precomputed K-independent quantities from
    precompute_ball_invariants (same observed/train/test/base arrays,
    reused across a K-sweep). Default None recomputes everything exactly
    as before -- every existing call site is unaffected.
    """
    n_cache = len(observed_queries)
    if n_max >= n_cache:
        raise ValueError(f"n_max={n_max} must be < n_cache={n_cache}")

    if invariants is not None:
        sorted_cache_dists = invariants['sorted_cache_dists']
        sorted_cache_indices = invariants['sorted_cache_indices']
        all_dists_matrix = invariants['all_dists']
        train_dist_matrix = invariants['train_dist_only']
        test_dist_matrix = invariants['test_dist_only']
    else:
        train_hat = _normalize(train_queries.copy())
        test_hat = _normalize(test_queries.copy())
        base_hat = _normalize(base_vectors.copy())
        obs_hat = _normalize(observed_queries.copy())
        sorted_cache_dists, sorted_cache_indices = _pairwise_cache_dists(observed_queries, metric)

    candidates: List[CandidateEntry] = []

    for i in range(n_cache):
        # invariants across all n_max spheres for this query -- computed
        # once per query, either here or (if a K-sweep is reusing them)
        # already computed once for the whole sweep in precompute_ball_invariants.
        if invariants is not None:
            all_dists = all_dists_matrix[i]
            train_thresh = train_dist_matrix[i] + train_r_t
            test_thresh = test_dist_matrix[i] + test_r_t
        else:
            q_hat = obs_hat[i]
            all_dists = _base_dists_for_query(q_hat, base_hat, metric)
            train_thresh = _base_dists_for_query(q_hat, train_hat, metric) + train_r_t
            test_thresh = _base_dists_for_query(q_hat, test_hat, metric) + test_r_t

        for n in range(1, n_max + 1):
            nn_index = int(sorted_cache_indices[i, n - 1])
            nn_dist = float(sorted_cache_dists[i, n - 1])
            r_c = nn_dist + float(margin_lookup[nn_index])

            R_c_mask = all_dists <= r_c
            rc_size = int(R_c_mask.sum())
            R_c = R_c_mask if store_rc else None

            train_covered = train_thresh <= r_c
            # neighbor_n's own coverage uses margin_lookup[nn_index] directly
            # (the exact value r_c was built from) rather than the separately
            # recomputed train_thresh[nn_index] -- holds by construction
            # (nn_dist + margin_lookup[nn_index] <= r_c is an equality) for
            # every K, not a floating-point coincidence.
            train_covered[nn_index] = nn_dist + float(margin_lookup[nn_index]) <= r_c
            train_covered[i] = False
            if not train_covered.any():
                continue

            test_covered = test_thresh <= r_c

            candidates.append(CandidateEntry(
                query=observed_queries[i],
                r_Q=float(observed_r_Q[i]),
                alpha=None,
                r_c=r_c,
                R_c=R_c,
                rc_size=rc_size,
                covered_train_indices=train_covered,
                covered_test_indices=test_covered,
                top_k_vectors=observed_top_k_vectors[i],
                top_k_indices=observed_top_k_indices[i],
                half_gap=float(observed_half_gaps[i]),
                source_index=i,
                n=n,
            ))

    return candidates


def propose_candidates_ball(
    observed_queries: np.ndarray,
    observed_r_Q: np.ndarray,
    observed_top_k_vectors: List[List[np.ndarray]],
    observed_top_k_indices: List[List[int]],
    observed_half_gaps: np.ndarray,
    train_queries: np.ndarray,
    train_r_t: np.ndarray,
    test_queries: np.ndarray,
    test_r_t: np.ndarray,
    base_vectors: np.ndarray,
    metric: DistanceMetric,
    n_max: int = 20,
    K: int = 5,
    store_rc: bool = True,
    invariants: Optional[dict] = None,
) -> List[CandidateEntry]:
    """
    Ball-like proposal: for each observed query, propose up to n_max nested-
    sphere candidates. Sphere n's radius is r_c = D(Q, q_n) + D(q_n, E_K),
    the n-th nearest cache neighbor's distance plus the K-th nearest base
    distance FROM that neighbor -- sized to fully contain q_n's own K-ball.

    K is shared between the neighbor's-own-ball radius and the simulation K
    (no separate k_cache/k_test distinction -- that split had no inherent
    meaning).

    K=0: a "0-neighbor ball" around q_n has no radius, so margin_lookup is
    uniformly 0 -- the same formula as K>=1, not a special substitution.
    r_c = D(Q, q_n) exactly. covered_train_indices then reduces to pure
    distance containment (D(Q, t) <= r_c) for every t, since train_r_t/
    test_r_t are also 0 in containment mode (see population_runner.py's
    k_test_override==0 branch) -- consistent with simulate-time hit
    certification, which (once K flows through as 0, not clamped to 1) also
    reduces to "hit iff contained" via the CIG algorithm's existing O1 gate
    (see lemmas_sigma.py).

    K>0: margin_lookup reuses observed_r_Q directly instead of
    recomputing "each observed query's K-th nearest base-vector distance"
    via _precompute_kth_base_dist -- these are mathematically the exact
    same quantity by construction (observed_r_Q IS the K-th-nearest-base
    distance for the observed queries, computed against this same
    base_vectors array, for every real caller in this codebase). Reusing
    it instead of recomputing avoids a real, measured floating-point
    divergence: two independently-computed values for "the same" quantity
    (on-the-fly batched matmul here vs. a separately-prepared GT file
    upstream) can differ by a few ULPs, which is enough to flip a
    boundary comparison for a candidate's own generating neighbor (see
    _propose_nested_spheres' train_covered[nn_index] construction below).
    CONTRACT: this requires observed_r_Q[i] to actually be the K-th
    nearest-base-vector distance for observed_queries[i] against this
    exact base_vectors array, for the K passed to this call -- true for
    every current caller (run_population.py, debug_common.py,
    debug_ball_geometry.py all load/pass it that way), but not re-verified
    here, since doing so would defeat the point (it would mean
    recomputing the exact value this is meant to avoid recomputing).

    n is no longer swept externally: all n_max spheres per query are proposed
    in one pass and left for select_candidates to choose between. alpha is
    None (not applicable).

    invariants: optional precomputed K-independent quantities from
    precompute_ball_invariants, passed straight through to
    _propose_nested_spheres. Default None (every existing call site) is
    unaffected -- see that function's docstring.
    """
    if K == 0:
        margin_lookup = np.zeros(len(observed_queries), dtype=np.float32)
    else:
        margin_lookup = observed_r_Q.astype(np.float32)
    return _propose_nested_spheres(
        observed_queries, observed_r_Q, observed_top_k_vectors, observed_top_k_indices,
        observed_half_gaps, train_queries, train_r_t, test_queries, test_r_t,
        base_vectors, metric, n_max, margin_lookup=margin_lookup, store_rc=store_rc,
        invariants=invariants,
    )
