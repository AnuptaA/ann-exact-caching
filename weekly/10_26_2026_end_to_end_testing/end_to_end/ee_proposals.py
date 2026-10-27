#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Candidate proposals built from a workload: Own, Group Union and Ball.

A candidate is (source train query Q, group of train queries, vector id set R). Candidates with |R| above the
maximum set size (5,000) are dropped; identical R sets are NOT merged (their groups differ).
- Own: group {Q}, R = Q's top-K ids.
- Group Union: for i = 0..20, group = {Q} + Q's i nearest other train queries, R = union of their top-K sets
  (i = 0 equals Own, kept as its own candidate).
- Ball: for i = 1..20 with q_i the i-th nearest other train query, radius = dist(Q, q_i) + theta(q_i), R = all database
  vectors within that radius of Q (from the sorted nearest-vector prefix), group = {Q, q_1, ..., q_i}. The prefix has one
  vector more than the maximum set size, so a ball of more than 5,000 vectors is told apart from one of exactly 5,000."""

import numpy as np

import ee_config as cfg
import ee_incidence as ei

KINDS = ("own", "group_union", "ball")

def _finish(w, R_lists, G_lists, source, level, dropped: int = 0) -> dict:
    return dict(R=R_lists, G=G_lists, source=np.array(source, dtype=np.int32), level=np.array(level, dtype=np.int16),
                dropped=dropped)

def own_candidates(w) -> dict:
    R = [np.sort(w.train_top_ids[q]) for q in range(w.n)]
    return _finish(w, R, [[q] for q in range(w.n)], range(w.n), [0] * w.n)

def group_union_candidates(w, max_i: int = cfg.GROUP_UNION_MAX_I, max_size: int = cfg.CANDIDATE_MAX_SET) -> dict:
    R, G, source, level, dropped = [], [], [], [], 0
    for q in range(w.n):
        nbrs = w.tt_order[q, :max_i]
        for i in range(max_i + 1):
            group = [q] + nbrs[:i].tolist()
            r = np.unique(w.train_top_ids[group].ravel())
            if len(r) > max_size:
                dropped += 1
                continue
            R.append(r.astype(np.int32))
            G.append(group)
            source.append(q)
            level.append(i)
    return _finish(w, R, G, source, level, dropped)

def ball_candidates(w, max_i: int = cfg.BALL_MAX_I, max_size: int = cfg.CANDIDATE_MAX_SET) -> dict:
    assert w.prefix_ids.shape[1] == max_size + 1, "the nearest-vector prefix must hold max_size + 1 vectors"
    R, G, source, level, dropped = [], [], [], [], 0
    theta = w.theta_train
    for q in range(w.n):
        nbrs = w.tt_order[q, :max_i]
        radii = w.tt_dist[q, nbrs] + theta[nbrs]
        counts = np.searchsorted(w.prefix_dist[q], radii, side="right")
        for i in range(1, max_i + 1):
            if counts[i - 1] > max_size:
                dropped += 1
                continue
            R.append(w.prefix_ids[q, :counts[i - 1]].copy())
            G.append([q] + nbrs[:i].tolist())
            source.append(q)
            level.append(i)
    return _finish(w, R, G, source, level, dropped)

BUILDERS = dict(own=own_candidates, group_union=group_union_candidates, ball=ball_candidates)

def make_incidence(w, kind: str, n_vec: int = cfg.N_DB, **kw) -> tuple:
    """(Incidence over the workload's train queries and top-K slots, candidate info) for one proposal kind."""
    cands = BUILDERS[kind](w, **kw)
    cptr, cvec = ei.csr_from_lists(cands["R"])
    gptr, gq = ei.csr_from_lists(cands["G"])
    inc = ei.Incidence(n=w.n, K=w.K, n_vec=n_vec, slot_vec=w.train_top_ids.astype(np.int32), cand_ptr=cptr,
                       cand_vec=cvec, grp_ptr=gptr, grp_q=gq, cand_source=cands["source"])
    return inc, cands
