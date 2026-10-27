#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Stage C (oracle arm): HP1, HP2 and HP3 on fixed pools with baselines, seed 0, both datasets, all regimes, budgets
and targets. Pools per cell: the five named selection triplets (Ball SEL1, Group Union SEL2, Own SEL5, Ball SEL3, Group
Union SEL4 with m = 0.5, 1.0, 1.5) and the baselines (no cache, Popularity, Random Own, threshold_baseline).

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_run_c.py"""

import json
import sys
import time

import numpy as np
import pandas as pd

import ee_baselines as eb
import ee_config as cfg
import ee_estimators as est
import ee_hitpolicy as hp
import ee_incidence as ei
import ee_proposals as ep
import ee_workload as ew
from data_prep import load_base_pool
from rt_chunks import current_rss_mb, write_table_atomic

OUT_DIR = cfg.RESULTS_DIR / "stage_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
M_GRID = (0.5, 1.0, 1.5)
SEED = 0

def eval_rows(w, pool: eb.Pool, T: float, meta: dict) -> list:
    true_test = est.true_recall_counts(pool.pool_mask, w.test_top_ids)
    rows = hp.evaluate_pool(SEED, T, true_test, true_test, pool.train_true_counts, pool.train_true_counts, w.K,
                            oracle_arm=True)
    return [dict(meta, pool=pool.name, pool_size=int(pool.pool_mask.sum()), n_selected=pool.n_selected, **r)
            for r in rows]

def main() -> int:
    rows = []
    t_start = time.time()
    for ds in cfg.DATASETS:
        base = load_base_pool(ds)
        for lam in cfg.LAMBDAS:
            w = ew.get_workload(ds, SEED, lam, base)
            incs = {}
            for kind in ep.KINDS:
                inc, _ = ep.make_incidence(w, kind)
                incs[kind] = (inc, ei.build_index(inc))
            base_pools = {}
            for B in cfg.BUDGETS:
                base_pools[B] = [eb.popularity(w, B), eb.random_own(w, B, SEED, *incs["own"]),
                                 eb.threshold_baseline(w, B), eb.no_cache(w)]
                ball_sel3 = ei.greedy(*incs["ball"], "SEL3", 0.9, B)  # T-independent
                base_pools[B].append(eb.pool_from_result("ball+SEL3", ball_sel3))
                base_pools[B][-1].selected_info = ball_sel3
            for T in cfg.TARGETS:
                for B in cfg.BUDGETS:
                    meta = dict(dataset=ds, lam=lam, seed=SEED, budget=B, T=T)
                    ref = ei.greedy(*incs["group_union"], "SEL2", T, B)
                    sel = {"ball+SEL1": ei.greedy(*incs["ball"], "SEL1", T, B), "group_union+SEL2": ref,
                           "own+SEL5": ei.greedy(*incs["own"], "SEL5", T, B)}
                    for m in M_GRID:
                        h0 = min(w.n, max(1, int(round(m * ref.h_star))))
                        sel[f"group_union+SEL4(m={m})"] = ei.greedy(*incs["group_union"], "SEL4", T, B, h0)
                    pools = [eb.pool_from_result(name, r) for name, r in sel.items()]
                    pools += [p for p in base_pools[B] if p.name != "ball+SEL3"] + \
                             [p for p in base_pools[B] if p.name == "ball+SEL3"]
                    for pool in pools:
                        extra = {}
                        if pool.name in sel:
                            r = sel[pool.name]
                            extra = dict(stalled=r.stalled, unused_budget=r.unused_budget, h_star_train=r.h_star)
                        elif pool.name == "ball+SEL3":
                            r = base_pools[B][-1].selected_info
                            extra = dict(stalled=r.stalled, unused_budget=r.unused_budget, h_star_train=r.h_star)
                        for row in eval_rows(w, pool, T, meta):
                            row.update(extra)
                            rows.append(row)
            print(f"{ds} lambda {lam}: {len(rows)} rows, {time.time() - t_start:.0f}s, rss {current_rss_mb():.0f} MB",
                  flush=True)
    table = pd.DataFrame(rows)
    write_table_atomic(table, OUT_DIR / "stage_c_runs.csv.gz")
    (OUT_DIR / "stage_c_summary.json").write_text(json.dumps(dict(rows=len(table), seconds=time.time() - t_start,
                                                                  peak_rss_mb=current_rss_mb()), indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
