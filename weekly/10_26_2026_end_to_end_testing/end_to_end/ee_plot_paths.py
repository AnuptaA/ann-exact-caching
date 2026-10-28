#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Running Y_t paths of HP3 (oracle and LSH arm, seed 0) for two cells of the stage D and F grids. The unit files store
only stream summaries, so these two cells are replayed: the pool is rebuilt, the same estimates and the same stream
order are used, and the replay is checked against the stored stage F row (hit rate and achieved recall).

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_plot_paths.py"""

import sys

import numpy as np
import pandas as pd

import ee_baselines as eb
import ee_config as cfg
import ee_estimators as est
import ee_hitpolicy as hp
import ee_incidence as ei
import ee_proposals as ep
import ee_run_f as rf
import ee_workload as ew
from data_prep import load_base_pool

OUT_PATH = cfg.RESULTS_DIR / "stage_f" / "paths_seed0.npz"
SEED = 0
BUDGET = cfg.BUDGETS[1]
T = 0.9
CELLS = (("sift", "lam0", "popularity"), ("sift", "lam1", "group_union+SEL2"))

def running_y(test_est: np.ndarray, true_test: np.ndarray, train_est: np.ndarray, seed: int) -> tuple:
    """(Y_t path, hit rate) of HP3 with the cap learned from the train estimates."""
    order = hp.stream_order(seed, len(test_est))
    est_u = hp.counts_to_units(test_est)[order]
    cap = hp.learn_cap(hp.counts_to_units(train_est), T)
    hits = hp.run_policy("HP3", est_u, T, cap)
    deliv = hp.delivered_units(hits, np.asarray(true_test)[order])
    return np.cumsum(deliv) / (hp.U * np.arange(1, len(deliv) + 1)), float(hits.mean())

def build_pool(w, regime_inc: dict, name: str) -> eb.Pool:
    if name == "popularity":
        return eb.popularity(w, BUDGET)
    kind, score = name.split("+")
    inc, idx = regime_inc[kind]
    return eb.pool_from_result(name, ei.greedy(inc, idx, score, T, BUDGET), inc)

def main() -> int:
    chosen = rf.load_chosen()
    out, bases = {}, {}
    for ds, regime, name in CELLS:
        base = bases.setdefault(ds, load_base_pool(ds))
        lam, per_anchor = cfg.REGIME_SPECS[regime]
        w = ew.get_workload(ds, SEED, lam, base, per_anchor=per_anchor)
        c = chosen[ds]
        arm = est.LSHArm(w, base, SEED, c["L"], c["m"], c["nu"], c["S"], ew.db_theta(ds, base))
        pe = rf.PoolEstimates(w, base, arm)
        kinds = {k: (lambda inc: (inc, ei.build_index(inc)))(ep.make_incidence(w, k)[0]) for k in ("group_union",)}
        pool = build_pool(w, kinds, name)
        true_test = est.true_recall_counts(pool.pool_mask, w.test_top_ids)
        test_est, train_est = pe.counts(pool, BUDGET)
        y_lsh, hit_lsh = running_y(test_est, true_test, train_est, SEED)
        y_or, hit_or = running_y(true_test, true_test, pool.train_true_counts, SEED)
        stored = pd.read_csv(rf.UNIT_DIR / f"{ds}_{regime}_seed{SEED}.csv.gz", comment="#", low_memory=False)
        row = stored[(stored.policy == "HP3") & (stored.budget == BUDGET) & (stored["T"] == T) & (stored.pool == name)].iloc[0]
        assert np.isclose(row.hit_rate, hit_lsh, atol=1e-12) and np.isclose(row.achieved_recall, y_lsh[-1], atol=1e-12)
        assert np.isclose(row.or_hit_rate, hit_or, atol=1e-12)
        key = f"{ds}|{regime}|{name}"
        out[f"{key}|lsh"], out[f"{key}|oracle"] = y_lsh, y_or
        print(f"{key}: lsh hit {hit_lsh:.3f} final Y {y_lsh[-1]:.3f} min Y {y_lsh.min():.3f} at t={y_lsh.argmin() + 1}; "
              f"oracle hit {hit_or:.3f} final Y {y_or[-1]:.3f} min Y {y_or.min():.3f} at t={y_or.argmin() + 1}")
    np.savez(OUT_PATH, **out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
