#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Diagnostic: "filled" variant of the Group Union SEL2 and Own SEL5 pools. After the main greedy stalls, the remaining
budget is spent by SEL3 gain per cost (the main grid is unchanged). HP3 hit rate and the quality hit rate (hits with true
recall >= 0.5, as a share of all test queries) for base versus filled pools at the 5% and 10% budgets, SIFT and ESCI at
lambda 0, lambda 0.5 and the diffuse regime, seeds 0 to 4, T in {0.8, 0.9, 0.95}. Oracle arm.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_run_fill.py"""

import sys
import time

import numpy as np
import pandas as pd

import ee_baselines as eb
import ee_config as cfg
import ee_incidence as ei
import ee_proposals as ep
import ee_run_d as rd
import ee_workload as ew
from data_prep import load_base_pool
from rt_chunks import current_rss_mb, write_table_atomic

OUT_DIR = cfg.RESULTS_DIR / "fill"
UNIT_DIR = OUT_DIR / "units"
UNIT_DIR.mkdir(parents=True, exist_ok=True)
REGIMES = ("lam0", "lam0.5", "diffuse")
BUDGETS = (cfg.BUDGETS[1], cfg.BUDGETS[2])
SPECS = (("group_union", "SEL2"), ("own", "SEL5"))
ROWS_PER_UNIT = len(BUDGETS) * len(cfg.TARGETS) * len(SPECS) * 2 * len(rd.hp.POLICIES)

def run_unit(ds: str, regime: str, seed: int, base: np.ndarray) -> pd.DataFrame:
    lam, per_anchor = cfg.REGIME_SPECS[regime]
    w = ew.get_workload(ds, seed, lam, base, per_anchor=per_anchor)
    incs = {kind: ep.make_incidence(w, kind)[0] for kind in ("group_union", "own")}
    idxs = {kind: ei.build_index(inc) for kind, inc in incs.items()}
    rows = []
    for B in BUDGETS:
        for T in cfg.TARGETS:
            for kind, score in SPECS:
                plain = ei.greedy(incs[kind], idxs[kind], score, T, B)
                filled = ei.greedy(incs[kind], idxs[kind], score, T, B, fill_score="SEL3")
                for variant, r in (("base", plain), ("filled", filled)):
                    pool = eb.pool_from_result(f"{kind}+{score}", r)
                    meta = dict(dataset=ds, regime=regime, lam=lam, per_anchor=per_anchor, seed=seed, budget=B, T=T,
                                variant=variant, base_stalled=bool(plain.stalled), base_pool_size=plain.pool_size,
                                n_fill_picks=len(r.selected) - len(plain.selected), fill_vectors=r.pool_size - plain.pool_size)
                    for row in rd.evaluate(w, pool, T, seed, meta):
                        row.update(stalled=r.stalled, unused_budget=r.unused_budget, h_star_train=r.h_star)
                        rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df) == ROWS_PER_UNIT and not df[["hit_rate", "hit_rate_recall_ge_half"]].isna().any().any()
    return df

def main() -> int:
    bases = {}
    t_start = time.time()
    for seed in cfg.SEEDS:
        for ds in cfg.DATASETS:
            for rg in REGIMES:
                path = UNIT_DIR / f"{ds}_{rg}_seed{seed}.csv.gz"
                if path.exists() and len(pd.read_csv(path, comment="#")) == ROWS_PER_UNIT:
                    continue
                bases.setdefault(ds, load_base_pool(ds))
                t0 = time.time()
                write_table_atomic(run_unit(ds, rg, seed, bases[ds]), path)
                print(f"{ds} {rg} seed {seed}: {time.time() - t0:.0f}s, total {time.time() - t_start:.0f}s, "
                      f"rss {current_rss_mb():.0f} MB", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
