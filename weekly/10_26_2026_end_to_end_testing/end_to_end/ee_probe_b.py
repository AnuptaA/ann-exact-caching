#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Stage B timing probe for the full-grid estimate: for every (dataset, lambda) workload of seed 0 and every proposal,
time the index build and the greedy runs of SEL1, SEL2, SEL3, SEL4 (m = 1.0) and SEL5 at the three budgets with
T = 0.9, plus a T sensitivity at one workload. Writes results/stage_b/probe_times.csv.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_probe_b.py"""

import sys
import time

import numpy as np
import pandas as pd

import ee_config as cfg
import ee_estimators as est
import ee_hitpolicy as hp
import ee_incidence as ei
import ee_proposals as ep
import ee_workload as ew
from data_prep import load_base_pool
from rt_chunks import current_rss_mb

OUT = cfg.RESULTS_DIR / "stage_b"
OUT.mkdir(parents=True, exist_ok=True)

def main() -> int:
    rows = []
    for ds in cfg.DATASETS:
        base = load_base_pool(ds)
        for lam in cfg.LAMBDAS:
            w = ew.get_workload(ds, 0, lam, base)
            incs = {}
            for kind in ep.KINDS:
                t0 = time.time()
                inc, cands = ep.make_incidence(w, kind)
                t1 = time.time()
                idx = ei.build_index(inc)
                incs[kind] = (inc, idx)
                rows.append(dict(dataset=ds, lam=lam, proposal=kind, what="build", n_cand=inc.n_cand,
                                 dropped=cands["dropped"], sum_R=int(np.diff(inc.cand_ptr).sum()), n_aff=len(idx.aff_q),
                                 seconds=round(t1 - t0 + (time.time() - t1), 2), rss_mb=round(current_rss_mb())))
            t_list = (0.9, 0.8, 0.95) if (ds, lam) == ("sift", 0.5) else (0.9,)
            for T in t_list:
                for B in cfg.BUDGETS:
                    ref_inc, ref_idx = incs["group_union"]
                    ref = ei.greedy(ref_inc, ref_idx, "SEL2", T, B)
                    h0 = min(w.n, max(1, ref.h_star))
                    for kind in ep.KINDS:
                        inc, idx = incs[kind]
                        for score in ("SEL1", "SEL2", "SEL3", "SEL4", "SEL5"):
                            r = ref if (kind, score) == ("group_union", "SEL2") else \
                                ei.greedy(inc, idx, score, T, B, h0 if score == "SEL4" else None)
                            true_test = est.true_recall_counts(r.pool_mask, w.test_top_ids)
                            t0 = time.time()
                            hp.evaluate_pool(0, T, true_test, true_test, r.covered.astype(np.int64),
                                             r.covered.astype(np.int64), w.K, oracle_arm=True)
                            rows.append(dict(dataset=ds, lam=lam, proposal=kind, what=score, T=T, budget=B,
                                             seconds=round(r.seconds, 3), eval_seconds=round(time.time() - t0, 3),
                                             n_steps=len(r.selected), pool_size=r.pool_size, h_star=r.h_star,
                                             stalled=r.stalled, rss_mb=round(current_rss_mb())))
            print(f"{ds} lambda {lam} done, rss {current_rss_mb():.0f} MB", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "probe_times.csv", index=False)
    return 0

if __name__ == "__main__":
    sys.exit(main())
