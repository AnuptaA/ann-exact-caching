#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Cache entries: a state is a list of indices j = candidate * len(k_values) + k_index."""

import numpy as np

from utils.base.knn import exact_topk, pairwise_dist, sq_norms

#-------------------------------------------------------------------------------


def entry_kmax(entries, k_values):
    """Cached candidate indices and the largest K cached for each."""
    n_kv = len(k_values)
    kmax = {}
    for j in entries:
        i, K = j // n_kv, k_values[j % n_kv]
        kmax[i] = max(kmax.get(i, 0), K)
    cand = np.array(sorted(kmax), dtype=np.int64)
    return cand, np.array([kmax[i] for i in cand], dtype=np.int64)


def perturbed(entries, frac, n_cand, n_kv, rng):
    """Replace a fraction of entries by random unselected candidates, same K."""
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    entries = list(entries)
    taken = set(entries)  # never re-pick an original or an earlier replacement
    for pos in rng.choice(len(entries), int(round(frac * len(entries))), replace=False):
        t = entries[pos] % n_kv
        if sum(1 for j in taken if j % n_kv == t) >= n_cand:
            raise ValueError(f"no unselected candidate left for k index {t}")
        while True:
            j = int(rng.integers(n_cand)) * n_kv + t
            if j not in taken:
                break
        taken.add(j)
        entries[pos] = j
    return entries


def build_entry_tables(cand_q, db, k_values, m_max, cached_top=None):
    """Per candidate: cached ids (as greedy used them), radius, and profile.

    Cached ids reproduce greedy's candidates exactly (same exact_topk call, or
    the saved neighbors passed as cached_top); the profile is the next nearest
    neighbors excluding the cached ids. With m_max <= 1 the profile is not
    computed (left empty), which is all the calibrated / predictor
    evaluations need.
    """
    k_values = list(k_values)
    if not k_values or k_values[0] < 1 or any(a >= b for a, b in zip(k_values, k_values[1:])):
        raise ValueError(f"k_values must be positive and strictly increasing, got {k_values}")
    k_max = k_values[-1]
    if cached_top is not None and cached_top.shape[1] < k_max:
        raise ValueError(f"saved neighbors have {cached_top.shape[1]} columns, "
                         f"but max(k_values) = {k_max}")
    if cached_top is not None and len(cached_top) != len(cand_q):
        raise ValueError(f"{len(cached_top)} saved neighbor rows for {len(cand_q)} candidates")
    if cached_top is None:
        cached_top = exact_topk(cand_q, db, k_max)
    big = exact_topk(cand_q, db, k_max + m_max) if m_max > 1 else None
    n_kv = len(k_values)
    n_cand = len(cand_q) * n_kv
    radius = np.empty(n_cand)
    prof_d = np.full((n_cand, m_max), np.inf)
    prof_id = np.zeros((n_cand, m_max), dtype=np.int64)
    cached_ids = []
    for i in range(len(cand_q)):
        top_ids = cached_top[i]
        d_top = pairwise_dist(cand_q[i:i + 1], db[top_ids])[0]
        if big is not None:
            ids_big = big[i]
            d_big = pairwise_dist(cand_q[i:i + 1], db[ids_big])[0]
        for t, K in enumerate(k_values):
            j = i * n_kv + t
            ids = top_ids[:K]
            cached_ids.append(ids)
            radius[j] = d_top[:K].max()
            if big is not None:
                keep = ~np.isin(ids_big, ids)
                pid = ids_big[keep][:m_max]
                prof_id[j, :len(pid)] = pid
                prof_d[j, :len(pid)] = d_big[keep][:m_max]
    return cached_ids, radius, prof_d, prof_id


def state_tables(run, entries):
    db = run.db
    n_db = len(db)
    in_cache = np.zeros(n_db, dtype=bool)
    keys = []
    for e, j in enumerate(entries):
        ids = np.asarray(run.cached_ids[j], dtype=np.int64)
        in_cache[ids] = True
        keys.append(e * n_db + ids)
    cached = np.flatnonzero(in_cache)
    ent = np.asarray(entries, dtype=np.int64)
    ent_q = run.cand_q[ent // len(run.k_values)]
    db_c = np.asarray(db[cached], dtype=np.float64)
    return {
        "in_cache": in_cache, "cached": cached, "n_db": n_db,
        # sorted (entry, vector) membership keys: entry e holds vector v iff
        # e * n_db + v is present (avoids an entries x database table)
        "member_keys": np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64),
        "db_c": db_c, "db_c_sq": sq_norms(db_c),
        "ent_q": ent_q, "ent_q_sq": sq_norms(np.asarray(ent_q, dtype=np.float64)),
        "ent_r": run.radius[ent],
    }


def is_member(st, entry_idx, vector_ids):
    """Elementwise: does entry entry_idx (index into the state's entries) hold
    vector vector_ids? Arguments broadcast against each other."""
    q = (np.asarray(entry_idx, dtype=np.int64) * st["n_db"]
         + np.asarray(vector_ids, dtype=np.int64))
    keys = st["member_keys"]
    if len(keys) == 0:
        return np.zeros(q.shape, dtype=bool)
    pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return keys[pos] == q
