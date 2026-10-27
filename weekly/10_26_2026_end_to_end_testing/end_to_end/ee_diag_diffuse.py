#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Diffuse regime check (500 anchors, 5 train and 5 test copies each, seed 0): union size, slots per distinct vector,
copy overlap, and whether each budget binds for Own and Group Union. A budget binds when the pool uses all of it.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_diag_diffuse.py"""

import sys

import pandas as pd

import ee_config as cfg
import ee_diag_c as dc
import ee_incidence as ei
import ee_proposals as ep
import ee_workload as ew
from data_prep import load_base_pool

def main() -> int:
    lam, per_anchor = cfg.REGIME_SPECS["diffuse"]
    stats, sizes = [], []
    for ds in cfg.DATASETS:
        base = load_base_pool(ds)
        w = ew.get_workload(ds, 0, lam, base, per_anchor=per_anchor)
        stats.append(dict(dataset=ds, anchors=w.n_anchors, **dc.union_stats(w)))
        for kind in ("own", "group_union"):
            inc, _ = ep.make_incidence(w, kind)
            idx = ei.build_index(inc)
            for B in cfg.BUDGETS:
                ref = ei.greedy(inc, idx, "SEL2", 0.9, B) if kind == "group_union" else None
                for score in ("SEL1", "SEL2", "SEL3", "SEL4", "SEL5"):
                    h0 = max(1, ref.h_star) if score == "SEL4" and ref is not None else (w.n if score == "SEL4" else None)
                    r = ei.greedy(inc, idx, score, 0.9, B, h0)
                    sizes.append(dict(dataset=ds, proposal=kind, score=score, budget=B, pool_size=r.pool_size,
                                      used=round(r.pool_size / B, 3), stalled=r.stalled, h_star=r.h_star))
    pd.set_option("display.width", 220)
    t = pd.DataFrame(stats)
    t.to_csv(cfg.RESULTS_DIR / "stage_c" / "diffuse_500x5_union.csv", index=False)
    print(t.round(3).to_string(index=False))
    s = pd.DataFrame(sizes)
    s.to_csv(cfg.RESULTS_DIR / "stage_c" / "diffuse_500x5_pool_sizes.csv", index=False)
    print(s.to_string(index=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
