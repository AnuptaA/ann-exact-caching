#!/usr/bin/env python3
"""Tests for the hits-only policy and the incremental greedy populator."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from simulations.approximate_hits.greedy_population import (
    GreedyPopulator,
    hits_from_histograms,
    online_hits,
)

#-------------------------------------------------------------------------------


def _brute_hits(counts, target, k):
    """Largest top-retention prefix with mean retention >= target."""
    r = np.sort(np.asarray(counts))[::-1] / k
    best = 0
    for m in range(1, len(r) + 1):
        if r[:m].mean() >= target - 1e-12:
            best = m
    return best


@pytest.mark.parametrize("target", [0.0, 0.5, 0.7, 0.9, 0.95, 1.0])
def test_hits_from_histograms_matches_brute_force(target):
    rng = np.random.default_rng(0)
    k = 10
    for _ in range(200):
        counts = rng.integers(0, k + 1, size=rng.integers(1, 40))
        hist = np.bincount(counts, minlength=k + 1)
        got, kept = hits_from_histograms(hist, target, k)
        assert int(got) == _brute_hits(counts, target, k)
        top = np.sort(counts)[::-1][:int(got)]
        assert int(kept) == int(top.sum())


def test_hits_from_histograms_batched():
    rng = np.random.default_rng(1)
    k = 10
    hists = rng.integers(0, 5, size=(50, k + 1))
    got, _ = hits_from_histograms(hists, 0.8, k)
    for row, g in zip(hists, got):
        counts = np.repeat(np.arange(k + 1), row)
        assert int(g) == _brute_hits(counts, 0.8, k)


def test_empty_cache_has_no_hits():
    k = 10
    hist = np.zeros(k + 1, dtype=np.int64)
    hist[0] = 100  # every query has retention 0
    for target in [0.5, 0.9, 0.99]:
        assert int(hits_from_histograms(hist, target, k)[0]) == 0


def test_online_hits_never_exceeds_best_case():
    rng = np.random.default_rng(2)
    k = 10
    for _ in range(100):
        counts = rng.integers(0, k + 1, size=50)
        best = int(hits_from_histograms(np.bincount(counts, minlength=k + 1), 0.8, k)[0])
        n_online, kept = online_hits(counts, 0.8, k)
        assert n_online <= best
        if n_online:
            assert kept / (n_online * k) >= 0.8 - 1e-12


#-------------------------------------------------------------------------------


def _random_instance(rng, n_db=60, n_cand=40, n_score=30, k=5):
    cands = [rng.choice(n_db, size=rng.integers(3, 15), replace=False)
             for _ in range(n_cand)]
    score_topk = np.stack([rng.choice(n_db, size=k, replace=False)
                           for _ in range(n_score)])
    eval_topk = np.stack([rng.choice(n_db, size=k, replace=False)
                          for _ in range(20)])
    return cands, score_topk, eval_topk


def _counts(topk, cached):
    return cached[topk].sum(axis=1)


def test_incremental_state_matches_recomputation():
    rng = np.random.default_rng(3)
    k = 5
    cands, score_topk, eval_topk = _random_instance(rng, k=k)
    pop = GreedyPopulator(cands, score_topk, n_db=60, k=k, target=0.6,
                          eval_topk=eval_topk)
    for _ in range(8):
        if pop.step(max_size=60) is None:
            break
        cached = pop.cached
        # uncached vectors per candidate
        assert np.array_equal(
            pop.new_vectors, np.array([(~cached[c]).sum() for c in cands]))
        # pair overlaps: |uncached(candidate) intersect topk(query)|
        for p in range(pop.n_pairs):
            j, s = pop.pair_j[p], pop.pair_s[p]
            unc = set(np.asarray(cands[j])[~cached[cands[j]]].tolist())
            assert pop.pair_overlap[p] == len(unc & set(score_topk[s].tolist()))
        assert np.array_equal(pop.score.counts, _counts(score_topk, cached))
        assert np.array_equal(pop.eval.counts, _counts(eval_topk, cached))
        assert pop.cache_size == int(cached.sum())


def test_predicted_hits_match_actual_and_choice_is_greedy_best():
    rng = np.random.default_rng(4)
    k = 5
    target = 0.6
    cands, score_topk, _ = _random_instance(rng, k=k)
    pop = GreedyPopulator(cands, score_topk, n_db=60, k=k, target=target)

    def hits_with(cached):
        c = _counts(score_topk, cached)
        return int(hits_from_histograms(np.bincount(c, minlength=k + 1), target, k)[0])

    for _ in range(8):
        before = pop.cached.copy()
        h0 = hits_with(before)
        # brute-force utilities
        best = None
        for j, c in enumerate(cands):
            new = c[~before[c]]
            if len(new) == 0:
                continue
            after = before.copy()
            after[new] = True
            d_hits = hits_with(after) - h0
            d_kept = int(_counts(score_topk, after).sum() - _counts(score_topk, before).sum())
            key = (d_hits / len(new), d_kept / len(new))
            if best is None or key > best[0]:
                best = (key, j)
        rec = pop.step(max_size=60)
        if rec is None:
            break
        chosen = rec["candidate"]
        c = cands[chosen]
        new = c[~before[c]]
        after = before.copy()
        after[new] = True
        chosen_key = ((hits_with(after) - h0) / len(new),
                      int(_counts(score_topk, after).sum()
                          - _counts(score_topk, before).sum()) / len(new))
        assert chosen_key == best[0]
        assert pop.last_predicted_hits == hits_with(after) == pop.score_hits


def test_weights_equal_duplicated_queries():
    rng = np.random.default_rng(5)
    k = 5
    cands, score_topk, eval_topk = _random_instance(rng, k=k)
    weights = rng.integers(1, 4, size=len(score_topk))
    eval_w = rng.integers(1, 4, size=len(eval_topk))
    weighted = GreedyPopulator(cands, score_topk, 60, k, 0.6, eval_topk,
                               score_weights=weights, eval_weights=eval_w)
    duplicated = GreedyPopulator(cands, np.repeat(score_topk, weights, axis=0), 60, k,
                                 0.6, np.repeat(eval_topk, eval_w, axis=0))
    for _ in range(10):
        a = weighted.step(max_size=60)
        b = duplicated.step(max_size=60)
        assert (a is None) == (b is None)
        if a is None:
            break
        assert a["candidate"] == b["candidate"]
        assert a["score_hits"] == b["score_hits"]
        assert np.array_equal(weighted.eval.histogram(), duplicated.eval.histogram())


def test_clone_for_target_matches_fresh_instance():
    rng = np.random.default_rng(6)
    k = 5
    cands, score_topk, eval_topk = _random_instance(rng, k=k)
    template = GreedyPopulator(cands, score_topk, 60, k, 1.0, eval_topk)
    for target in [0.6, 0.8]:
        clone = template.clone_for_target(target)
        fresh = GreedyPopulator(cands, score_topk, 60, k, target, eval_topk)
        for _ in range(10):
            a = clone.step(max_size=60)
            b = fresh.step(max_size=60)
            assert (a is None) == (b is None)
            if a is None:
                break
            assert a == b
    # template itself is untouched by the clones
    assert template.cache_size == 0
    assert not template.cached.any()
    assert np.array_equal(template.new_vectors, np.array([len(c) for c in cands]))
