#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""The earlier 12 base + 8 extended retention features (feed calibrated.json / predictors.json)."""

import numpy as np

from utils.base.knn import pairwise_dist
from utils.cache.entries import is_member, state_tables

CHUNK = 2048
EPS = 1e-12
SPLITS = ("fit", "dev", "cal", "test")
FEATURE_NAMES = [
    "log_D1", "log_Dk", "log_Dk_over_D1", "log_gap_k", "log_min_d_entry",
    "log_min_d_over_r", "log_overshoot", "a1_frac", "frac_topk_from_entry",
    "log1p_inside", "log_Dk_over_r_entry", "log_r_entry",
]
EXT_NAMES = ["log_Dq1_over_Dk", "log_Dq2_over_Dk", "log_Dq3_over_Dk",
             "covered_frac", "log_dentry_over_Dk", "entry_hist_mean",
             "entry_hist_log_count", "log_cache_frac"]

#-------------------------------------------------------------------------------


def features(run, qvecs, st):
    k = run.k
    n = len(qvecs)
    F = np.empty((n, len(FEATURE_NAMES)))
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        q = qvecs[s:e]
        d_qc = pairwise_dist(q, st["db_c"], st["db_c_sq"])
        part = np.argpartition(d_qc, k, axis=1)[:, :k + 1]
        dpart = np.take_along_axis(d_qc, part, axis=1)
        order = np.argsort(dpart, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        dpart = np.take_along_axis(dpart, order, axis=1)
        d1, dk, dk1 = dpart[:, 0], dpart[:, k - 1], dpart[:, k]
        topk_ids = st["cached"][part[:, :k]]

        d_qe = pairwise_dist(q, st["ent_q"], st["ent_q_sq"])
        r = st["ent_r"][None, :]
        norm = d_qe / np.maximum(r, EPS)
        jstar = norm.argmin(axis=1)
        overshoot = ((dk[:, None] + d_qe) / np.maximum(r, EPS)).min(axis=1)
        rho = (r - d_qe).max(axis=1)
        a1 = np.minimum((d_qc <= rho[:, None]).sum(axis=1), k) / k
        inside = (d_qe < r).sum(axis=1)
        frac_from = is_member(st, jstar[:, None], topk_ids).mean(axis=1)
        r_js = st["ent_r"][jstar]

        F[s:e] = np.column_stack([
            np.log(d1 + EPS), np.log(dk + EPS), np.log((dk + EPS) / (d1 + EPS)),
            np.log((dk1 - dk) / (dk + EPS) + 1e-6), np.log(d_qe.min(axis=1) + EPS),
            np.log(norm.min(axis=1) + EPS), np.log(overshoot + EPS), a1, frac_from,
            np.log1p(inside), np.log((dk + EPS) / (r_js + EPS)), np.log(r_js + EPS),
        ])
    return F


def full_features(run, qvecs, st, cache_frac):
    """Base features (as eval_calibrated.features), extended features, j*."""
    k = run.k
    n = len(qvecs)
    base = features(run, qvecs, st)
    ext = np.zeros((n, len(EXT_NAMES)))
    jstar_all = np.empty(n, dtype=np.int64)
    ranks = [max(1, int(np.ceil(k * f))) - 1 for f in (0.25, 0.5, 0.75)]
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        q = qvecs[s:e]
        d_qc = pairwise_dist(q, st["db_c"], st["db_c_sq"])
        part = np.argpartition(d_qc, k - 1, axis=1)[:, :k]
        dpart = np.take_along_axis(d_qc, part, axis=1)
        order = np.argsort(dpart, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        dpart = np.take_along_axis(dpart, order, axis=1)
        dk = dpart[:, k - 1]
        topk_ids = st["cached"][part]

        d_qe = pairwise_dist(q, st["ent_q"], st["ent_q_sq"])
        r = st["ent_r"][None, :]
        jstar = (d_qe / np.maximum(r, EPS)).argmin(axis=1)
        inside = d_qe < r
        covered = np.zeros(topk_ids.shape, dtype=bool)
        for j in np.flatnonzero(inside.any(axis=0)):
            rows = inside[:, j]
            covered[rows] |= is_member(st, j, topk_ids[rows])
        ext[s:e, 0:3] = np.log((dpart[:, ranks] + EPS) / (dk[:, None] + EPS))
        ext[s:e, 3] = covered.mean(axis=1)
        ext[s:e, 4] = np.log((d_qe[np.arange(e - s), jstar] + EPS) / (dk + EPS))
        jstar_all[s:e] = jstar
    ext[:, 7] = np.log(cache_frac + EPS)
    return base, ext, jstar_all


def entry_history(jstar_fit, y_fit, w_fit, n_entries, smooth=20.0, folds=5, seed=0):
    """Smoothed per-entry mean retention from FIT queries.

    Returns (fit_values out-of-fold, full_table_mean, full_table_count).
    """
    prior = float(np.average(y_fit, weights=w_fit))

    def table(mask):
        cnt = np.bincount(jstar_fit[mask], weights=w_fit[mask], minlength=n_entries)
        tot = np.bincount(jstar_fit[mask], weights=(w_fit * y_fit)[mask], minlength=n_entries)
        return (tot + smooth * prior) / (cnt + smooth), cnt

    fold = np.random.default_rng(seed).integers(0, folds, size=len(y_fit))
    oof_mean = np.empty(len(y_fit))
    oof_cnt = np.empty(len(y_fit))
    for f in range(folds):
        m, c = table(fold != f)
        rows = fold == f
        oof_mean[rows] = m[jstar_fit[rows]]
        oof_cnt[rows] = c[jstar_fit[rows]]
    full_mean, full_cnt = table(np.ones(len(y_fit), dtype=bool))
    return (oof_mean, oof_cnt), full_mean, full_cnt


def build_state(task, run):
    """Features and labels for all splits of one cache state."""
    target, point, entries = task
    if len(entries) == 0:
        return None
    st = state_tables(run, entries)
    if len(st["cached"]) <= run.k:
        return None
    out = {"target": target, "cache_frac": point["cache_frac"],
           "oracle_hit_rate_online": point["test_hit_rate_online"],
           "oracle_hit_rate": point["test_hit_rate"]}
    raw = {}
    for name in SPLITS:
        q, w = run.sets[name]
        y = st["in_cache"][run.topk[name]].mean(axis=1)
        base, ext, jstar = full_features(run, q, st, point["cache_frac"])
        raw[name] = (base, ext, jstar, y, w.astype(np.float64))
    n_ent = len(entries)
    (oof_m, oof_c), full_m, full_c = entry_history(raw["fit"][2], raw["fit"][3],
                                                   raw["fit"][4], n_ent)
    for name in SPLITS:
        base, ext, jstar, y, w = raw[name]
        if name == "fit":
            ext[:, 5], ext[:, 6] = oof_m, np.log1p(oof_c)
        else:
            ext[:, 5], ext[:, 6] = full_m[jstar], np.log1p(full_c[jstar])
        out[name] = {"base": base, "ext": ext, "y": y, "w": w}
    return out
