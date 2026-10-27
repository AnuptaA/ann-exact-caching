#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Stage B: one small real run (SIFT, lambda = 1.0, seed 0, 5% budget, T = 0.9) of three proposals x SEL1, SEL2, SEL3,
SEL4 (m = 0.5, 1.0, 1.5 against the Group Union SEL2 reference) and SEL5, with the oracle hit policies on every pool and
an LSH-arm check (group-aware train deficits) on the reference pool.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_run_b.py"""

import argparse
import json
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
from rt_chunks import current_rss_mb, free_disk_gb, write_table_atomic

OUT_DIR = cfg.RESULTS_DIR / "stage_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)
M_GRID = (0.5, 1.0, 1.5)

def h0_from_reference(m: float, h_ref: int, n: int) -> int:
    return min(n, max(1, int(round(m * h_ref))))

def step_summary(steps: dict) -> dict:
    g = steps["gain"]
    if len(g) == 0:
        return dict(n_steps=0)
    return dict(n_steps=len(g), gain_first=int(g[0]), gain_median=float(np.median(g)), gain_last=int(g[-1]),
                steps_gain_1=int((g == 1).sum()), steps_cost_0=int((steps["cost"] == 0).sum()),
                n_positive_first=int(steps["n_positive"][0]), n_positive_last=int(steps["n_positive"][-1]))

def evaluate_oracle(w, T: float, result: ei.Result) -> list:
    true_test = est.true_recall_counts(result.pool_mask, w.test_top_ids)
    true_train = result.covered.astype(np.int64)
    return hp.evaluate_pool(0, T, true_test, true_test, true_train, true_train, w.K, oracle_arm=True)

def lsh_check(w, base, inc, result: ei.Result, T: float, seed: int = 100, L: int = 4, m: int = 128, nu: int = 8) -> dict:
    """Group-aware LSH train deficits on one final pool, plus the consistency check against the slot counters."""
    pool_ids = np.flatnonzero(result.pool_mask)
    t0 = time.time()
    exact = ei.group_aware_counts(inc, result.selected, pool_ids, w.theta_train, w.train, base)
    t_group = time.time() - t0
    matches = float((exact == result.covered).mean())
    arm = est.LSHArm(w, base, seed, L, m, nu)
    train_counts = ei.group_aware_counts(inc, result.selected, pool_ids, arm.radius("train"), w.train, base)
    test_counts = est.vectors_within_counts(w.test, arm.radius("test"), base[pool_ids])
    true_test = est.true_recall_counts(result.pool_mask, w.test_top_ids)
    rows = hp.evaluate_pool(0, T, test_counts, true_test, train_counts, result.covered.astype(np.int64), w.K,
                            oracle_arm=False)
    return dict(seconds_group_aware=t_group, theta_vs_slot_counters_match=matches, rows=rows,
                train_count_mean=float(train_counts.mean()), slot_covered_mean=float(result.covered.mean()))

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=cfg.BUDGETS[1])
    ap.add_argument("--T", type=float, default=0.9)
    a = ap.parse_args()
    base = load_base_pool(a.dataset)
    w = ew.get_workload(a.dataset, a.seed, a.lam, base)
    T, B = a.T, a.budget
    incs = {}
    for kind in ep.KINDS:
        t0 = time.time()
        inc, cands = ep.make_incidence(w, kind)
        idx = ei.build_index(inc)
        incs[kind] = (inc, idx, cands)
        print(f"{kind}: {inc.n_cand} candidates, dropped {cands['dropped']}, sum|R| {int(np.diff(inc.cand_ptr).sum()):,}, "
              f"build {time.time() - t0:.1f}s, rss {current_rss_mb():.0f} MB", flush=True)

    ref_inc, ref_idx, _ = incs["group_union"]
    ref = ei.greedy(ref_inc, ref_idx, "SEL2", T, B)
    h_ref = ref.h_star
    print(f"reference (group_union, SEL2): h* = {h_ref}, pool {ref.pool_size}, {len(ref.selected)} picks, {ref.seconds:.1f}s",
          flush=True)

    runs = [(kind, "SEL1", None) for kind in ep.KINDS] + [(kind, "SEL2", None) for kind in ep.KINDS] + \
           [(kind, "SEL3", None) for kind in ep.KINDS] + [(kind, "SEL4", m) for kind in ep.KINDS for m in M_GRID] + \
           [(kind, "SEL5", None) for kind in ep.KINDS]
    rows, results = [], {}
    for kind, score, m in runs:
        inc, idx, cands = incs[kind]
        h0 = h0_from_reference(m, h_ref, w.n) if m is not None else None
        r = ref if (kind, score) == ("group_union", "SEL2") else ei.greedy(inc, idx, score, T, B, h0)
        results[(kind, score, m)] = r
        label = score if m is None else f"{score}(m={m})"
        row = dict(dataset=a.dataset, lam=a.lam, seed=a.seed, budget=B, T=T, proposal=kind, score=label, h0=h0,
                   pool_size=r.pool_size, n_selected=len(r.selected), n_sources=r.n_distinct_sources, stalled=r.stalled,
                   unused_budget=r.unused_budget, h_star_train=r.h_star, h_star_ref=h_ref, seconds=round(r.seconds, 2),
                   **step_summary(r.steps))
        for pol in evaluate_oracle(w, T, r):
            row[f"{pol['policy']}_hit_rate"] = pol["hit_rate"]
            row[f"{pol['policy']}_achieved"] = pol["achieved_recall"]
            if pol["policy"] == "HP3":
                row.update(hp3_cap_units=pol["learned_cap_units"], hp3_train_hit_fraction=pol["train_hit_fraction"],
                           hp3_test_hit_fraction=pol["test_hit_fraction"])
            row["hindsight_hit_rate_oracle"] = pol["hindsight_hit_rate_oracle"]
        rows.append(row)
        pd.DataFrame(dict(r.steps)).to_csv(OUT_DIR / f"steps_{kind}_{label}.csv.gz", index=False)
        print(f"  {kind:11s} {label:11s} pool {r.pool_size:5d} picks {len(r.selected):4d} h* {r.h_star:4d} "
              f"stalled {r.stalled!s:5s} unused {r.unused_budget:5d} {r.seconds:6.1f}s", flush=True)
        assert current_rss_mb() < cfg.RSS_LIMIT_MB and free_disk_gb(OUT_DIR) > cfg.MIN_FREE_DISK_GB
    table = pd.DataFrame(rows)
    write_table_atomic(table, OUT_DIR / "stage_b_runs.csv.gz")

    check = lsh_check(w, base, ref_inc, ref, T)
    pd.DataFrame(check["rows"]).to_csv(OUT_DIR / "stage_b_lsh_check_group_union_SEL2.csv", index=False)
    summary = {k: v for k, v in check.items() if k != "rows"}
    summary.update(peak_rss_mb=current_rss_mb(), total_seconds=float(table["seconds"].sum()))
    (OUT_DIR / "stage_b_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
