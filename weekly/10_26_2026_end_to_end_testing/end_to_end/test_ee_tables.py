#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_tables. Run with pytest or `python3 -B test_ee_tables.py`."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import ee_tables as et

def fake_rows(seed_effects):
    rng = np.random.default_rng(0)
    recs = []
    for seed, bump in enumerate(seed_effects):
        for pool, base in (("popularity", 0.30), ("random_own", 0.25), ("own+SEL5", 0.32 + bump)):
            for policy in ("HP2", "HP3"):
                recs.append(dict(dataset="sift", regime="lam0", budget=2500, T=0.9, seed=seed, pool=pool, policy=policy,
                                 hit_rate=base + (0.01 if policy == "HP3" else 0) + 0.001 * seed,
                                 hit_rate_recall_ge_half=base - 0.05, achieved_recall=0.9, mean_true_recall_of_hits=0.5))
    return pd.DataFrame(recs)

def test_cell_summary_mean_sd_and_seed_count():
    rows = fake_rows([0.0, 0.01, 0.02, 0.03, 0.04])
    c = et.cell_summary(rows)
    r = c[(c.pool == "own+SEL5") & (c.policy == "HP2")].iloc[0]
    vals = 0.32 + np.array([0.0, 0.01, 0.02, 0.03, 0.04]) + 0.001 * np.arange(5)
    assert np.isclose(r.hit_rate, vals.mean()) and np.isclose(r.hit_rate_sd, vals.std(ddof=1)) and r.n_seeds == 5

def test_pool_pairs_are_paired_by_seed():
    bumps = [0.0, 0.01, 0.02, -0.5, 0.04]
    rows = fake_rows(bumps)
    p = et.pool_pairs(rows, [("own+SEL5", "popularity")])
    r = p[p.policy == "HP2"].iloc[0]
    d = np.array([0.32 + b - 0.30 for b in bumps])  # the seed trend cancels in the pairing
    assert np.isclose(r.hit_rate_diff, d.mean()) and np.isclose(r.hit_rate_diff_sd, d.std(ddof=1))
    assert r.hit_rate_n_pos == 4 and r.hit_rate_n_neg == 1 and r.n_seeds == 5
    assert np.isclose(r.hit_rate_t, d.mean() / (d.std(ddof=1) / np.sqrt(5)))

def test_policy_pairs_hp3_minus_hp2_is_the_constant_bump():
    rows = fake_rows([0.0, 0.01, 0.02, 0.03, 0.04])
    p = et.policy_pairs(rows)
    r = p[(p.a == "HP3") & (p.b == "HP2") & (p.pool == "popularity")].iloc[0]
    assert np.isclose(r.hit_rate_diff, 0.01) and np.isclose(r.hit_rate_diff_sd, 0.0) and r.hit_rate_n_pos == 5

def test_default_pairs_cover_baselines_and_score_proposal_contrasts():
    pools = ["popularity", "random_own", "threshold_baseline", "no_cache"] + \
            [f"{k}+SEL{i}" for k in ("own", "group_union", "ball") for i in (1, 2, 3, 5)]
    pairs = et.default_pool_pairs(pools)
    assert ("own+SEL5", "popularity") in pairs and ("ball+SEL3", "own+SEL3") in pairs and ("own+SEL5", "own+SEL3") in pairs
    assert len(pairs) == len(set(pairs))

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
