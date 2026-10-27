#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_lsh_bench. Run with pytest or `python3 -B test_ee_lsh_bench.py`."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import ee_estimators as est
import ee_lsh_bench as lb
import ee_workload as ew
from rt_engine import make_hyperplanes

D, N_BASE, K = 12, 300, 10

def fixture(seed=0):
    rng = np.random.default_rng(seed)
    unit = lambda n: (lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(rng.standard_normal((n, D)))
    base, pool = unit(N_BASE), unit(300)
    w = ew.build_workload("sift", base, pool, seed, 0.5, n=60, K=K, per_anchor=10, prefix_len=40)
    theta_db = np.sort(np.arccos(np.clip(base @ base.T, -1, 1)) / np.pi, axis=1)[:, K]  # K-th other vector (self is 0)
    return base, w, theta_db

def test_recall_and_radius_metrics():
    m = lb.recall_metrics(np.array([10, 7, 12, 5]), np.array([10, 8, 10, 5]), 10)
    assert np.isclose(m["mean_signed_error"], (0 - 0.1 + 0.2 + 0) / 4) and np.isclose(m["mae"], (0 + 0.1 + 0.2 + 0) / 4)
    assert np.isclose(m["over_rate"], 0.25)   # only the +0.2 query exceeds 0.05; 0.1 under is not an overestimate
    theta = np.array([0.1, 0.2, 0.3, 0.4])
    r = lb.radius_metrics(0.5 * theta + 0.05, theta)
    assert np.isclose(r["radius_slope"], 0.5) and np.isclose(r["radius_mean_rel_err"], np.mean((0.5 * theta + 0.05 - theta) / theta))
    assert lb.config_bytes(4, 128, 128, 1000) == 1000 * (64 + 8) + 4 * 128 * 128 // 8

def test_bench_unit_matches_direct_computation_and_sample_nesting():
    base, w, theta_db = fixture(1)
    mask = np.zeros(N_BASE, dtype=bool)
    mask[np.random.default_rng(0).choice(N_BASE, 40, replace=False)] = True
    configs, nus, s_opts = [(1, 64), (4, 32)], [1, 4], (0, 50, 120)
    df = lb.bench_unit(w, base, theta_db, 5, {"popularity": mask}, configs, nus, s_opts, extra=dict(dataset="sift", regime="lam0.5"))
    assert len(df) == len(configs) * len(nus) * len(s_opts)
    W = make_hyperplanes(5, D)
    perm = np.random.default_rng([5, 21]).permutation(N_BASE)
    for S, (L, m), nu in ((50, (4, 32), 4), (0, (1, 64), 1), (120, (1, 64), 4)):
        ids = perm[:S]
        labeled = np.vstack([w.train, base[ids]]) if S else w.train
        labels = np.concatenate([w.theta_train, theta_db[ids]]) if S else w.theta_train
        r_hat = est.LSHStore(W, L, m, labeled, labels).estimate(w.test, nu)
        counts = est.vectors_within_counts(w.test, r_hat, base[mask])
        true = est.true_recall_counts(mask, w.test_top_ids)
        row = df[(df.L == L) & (df.m == m) & (df.nu == nu) & (df.S_num == S)].iloc[0]
        exp = lb.recall_metrics(counts, true, K)
        assert np.isclose(row.mae, exp["mae"]) and np.isclose(row.mean_signed_error, exp["mean_signed_error"])
        assert np.isclose(row.radius_slope, lb.radius_metrics(r_hat, w.theta_test)["radius_slope"])
        assert row.n_labeled == len(labeled)

def test_choose_config_rule_excludes_full_and_breaks_ties_by_overestimation():
    recs = []
    def add(L, m, nu, S, mae, over):
        for regime in ("lam0", "lam1"):
            for seed in (100, 101):
                recs.append(dict(dataset="sift", pool="popularity", regime=regime, seed=seed, L=L, m=m, nu=nu, S=S, mae=mae,
                                 over_rate=over, mean_signed_error=-0.1, bytes=1))
    add(1, 512, 8, "full", 0.01, 0.0)          # best MAE but reference only
    add(4, 128, 8, "2500", 0.100, 0.20)        # the lowest MAE among the allowed ones
    add(8, 64, 8, "none", 0.104, 0.05)         # within 0.005 of the best, much lower overestimation: wins
    add(1, 2048, 4, "10000", 0.107, 0.01)      # not within the tolerance
    pick = lb.choose_config(pd.DataFrame(recs), "sift", "popularity")
    assert (pick["L"], pick["m"], pick["S"]) == (8, 64, "none") and pick["n_within_tie"] == 2 and pick["n_averaged"] == 4
    assert np.isclose(pick["best_mae"], 0.100)

#-------------------------------------------------------------------------------

def run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed

if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
