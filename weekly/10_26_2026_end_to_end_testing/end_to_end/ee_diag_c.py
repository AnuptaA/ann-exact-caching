#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Pre stage D diagnostics: where each budget binds. Per dataset and regime (seeds 0 to 4): the number of distinct vectors
in the union of all train top-K sets, the share of test top-K vectors that union contains, and the mean top-K overlap
between copies of one anchor. Also the same numbers and pool sizes for a candidate diffuse regime (250 anchors x 10 copies).

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_diag_c.py"""

import sys

import numpy as np
import pandas as pd

import ee_config as cfg
import ee_incidence as ei
import ee_proposals as ep
import ee_workload as ew
from data_prep import load_base_pool, load_official_query_pool

OUT = cfg.RESULTS_DIR / "stage_c"
OUT.mkdir(parents=True, exist_ok=True)

def union_stats(w) -> dict:
    train_union = np.unique(w.train_top_ids)
    in_union = np.isin(w.test_top_ids, train_union)
    out = dict(union_distinct=len(train_union), slots=w.train_top_ids.size,
               slots_per_distinct=w.train_top_ids.size / len(train_union), test_top_in_train_union=float(in_union.mean()),
               test_queries_fully_covered=float(in_union.all(axis=1).mean()))
    overlaps = []
    for a in np.unique(w.train_anchor[w.train_anchor >= 0]):
        sel = np.flatnonzero(w.train_anchor == a)
        m = np.zeros((len(sel), int(w.train_top_ids.max()) + 1), np.float32)
        for r, q in enumerate(sel):
            m[r, w.train_top_ids[q]] = 1
        o = m @ m.T / w.K
        overlaps.append((o.sum() - np.trace(o)) / (len(sel) * (len(sel) - 1)))
    out["anchor_copy_overlap"] = float(np.mean(overlaps)) if overlaps else np.nan
    return out

def pool_sizes(w, label: str) -> list:
    rows = []
    for kind in ("own", "group_union", "ball"):
        inc, _ = ep.make_incidence(w, kind)
        idx = ei.build_index(inc)
        for score in ("SEL2", "SEL3"):
            for B in cfg.BUDGETS:
                r = ei.greedy(inc, idx, score, 0.9, B)
                rows.append(dict(regime=label, proposal=kind, score=score, budget=B, pool_size=r.pool_size,
                                 stalled=r.stalled, h_star=r.h_star))
    return rows

def main() -> int:
    rows, diffuse_rows, sizes = [], [], []
    for ds in cfg.DATASETS:
        base = load_base_pool(ds)
        for lam in cfg.LAMBDAS:
            for seed in cfg.SEEDS:
                w = ew.get_workload(ds, seed, lam, base)
                rows.append(dict(dataset=ds, lam=lam, seed=seed, **union_stats(w)))
            print(f"{ds} lambda {lam} done", flush=True)
        pool, _ = ew.dedupe_pool(load_official_query_pool(ds))
        wd = ew.build_workload(ds, base, pool, 0, 1.0, n=cfg.N_QUERIES, K=cfg.K, per_anchor=10)
        diffuse_rows.append(dict(dataset=ds, lam="diffuse_250x10", seed=0, **union_stats(wd)))
        sizes += [dict(dataset=ds, **r) for r in pool_sizes(wd, "diffuse_250x10")]
        print(f"{ds} diffuse done", flush=True)
    t = pd.DataFrame(rows + diffuse_rows)
    t.to_csv(OUT / "stage_c_diag_union_overlap.csv", index=False)
    pd.DataFrame(sizes).to_csv(OUT / "stage_c_diag_diffuse_pool_sizes.csv", index=False)
    g = t.groupby(["dataset", "lam"], sort=False).agg(["mean", "std"]).round(3)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(g.to_string())
    print(pd.DataFrame(sizes).to_string(index=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
