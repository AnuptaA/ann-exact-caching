#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for the follow-up analyses. Run with pytest or `python3 -B test_rt_followup.py` (temp dirs only)."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import rt_config
import rt_data
import rt_engine
import rt_followup_a as fa
import rt_followup_b as fb

LOG = logging.getLogger("test_rt_followup")

#-------------------------------------------------------------------------------
# Item A

def make_long(seed_count=2, n=300, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for seed in range(seed_count):
        theta = rng.uniform(0.1, 0.4, n)
        r = 0.05 + 0.6 * theta + rng.normal(0, 0.02, n)
        rows.append(pd.DataFrame(dict(dataset="sift", stage=3, config="L1_m512", nu=8, estimate="sample_train",
                                      seed=seed, query_id=np.arange(n), theta=theta, r=r)))
    return pd.concat(rows, ignore_index=True)

def test_fit_stats_match_numpy():
    df = make_long()
    fs = fa.fit_stats(df)
    for _, row in fs.iterrows():
        g = df[df["seed"] == row["seed"]]
        t, r = g["theta"].to_numpy(), g["r"].to_numpy()
        f = np.polyfit(t, r, 1)
        rv = np.polyfit(r, t, 1)
        assert np.isclose(row["forward_slope"], f[0]) and np.isclose(row["forward_intercept"], f[1])
        assert np.isclose(row["reverse_slope"], rv[0]) and np.isclose(row["reverse_intercept"], rv[1])
        assert np.isclose(row["r_squared"], np.corrcoef(t, r)[0, 1] ** 2)
        rel = (r - t) / t
        assert np.isclose(row["mean_error_pct"], 100 * rel.mean())
        assert np.isclose(row["sd_error_pct"], 100 * rel.std())
        assert np.isclose(row["forward_slope"] * row["reverse_slope"], row["r_squared"])

def test_calibration_bins_equal_count_with_ties():
    df = make_long(seed_count=2, n=300)
    df.loc[df["seed"] == 0, "r"] = np.round(df.loc[df["seed"] == 0, "r"], 2)  # many ties
    cal = fa.calibration_table(df)
    assert (cal["n_per_bin"] == 30).all()
    assert cal.groupby(["config", "bin"]).ngroups == 10
    d0 = df[df["seed"] == 1].sort_values(["r", "query_id"])
    bins = np.arange(300) * 10 // 300
    exp = d0.assign(bin=bins).groupby("bin")[["r", "theta"]].mean()
    got = cal.set_index("bin")
    assert np.all(np.diff(got["mean_r_hat"]) > 0)
    assert exp.shape[0] == 10

#-------------------------------------------------------------------------------
# Item B

def test_locality_builder_angles_anchors_and_x10_equivalence():
    rng = np.random.default_rng(4)
    pool = rt_data.sample_uniform_sphere(200, 16, rng)
    db = rt_data.sample_uniform_sphere(300, 16, rng)
    sets = {}
    for X in (2, 10, 40):
        sets[X] = fb.build_locality_x(pool, db, 100, 100, X, LOG, f"t{X}")
        for role in ("train", "test"):
            meta, Q = sets[X][f"meta_{role}"], sets[X][role]
            anchors = pool[meta["source_index"].to_numpy()]
            ang = np.degrees(np.arccos(np.clip((Q * anchors).sum(axis=1), -1, 1)))
            assert ang.max() <= X + 1e-6 and ang.min() >= 0
            assert (meta.groupby("anchor_id").size() == 2).all()
        both = np.vstack([sets[X]["train"], sets[X]["test"]])
        assert len(np.unique(both, axis=0)) == len(both)
    for X in (2, 40):
        assert np.array_equal(sets[X]["meta_train"]["source_index"].to_numpy(),
                              sets[10]["meta_train"]["source_index"].to_numpy())
    ref = rt_data._build_locality(pool, db, 100, 100, LOG, "ref")
    assert np.array_equal(ref["train"], sets[10]["train"]) and np.array_equal(ref["test"], sets[10]["test"])

def test_reduced_state_matches_full_state_at_ref_s():
    rng = np.random.default_rng(1)
    N, d, S = 400, 16, 150
    db = rt_data.sample_uniform_sphere(N, d, rng)
    train = rt_data.sample_uniform_sphere(40, d, rng)
    test = rt_data.sample_uniform_sphere(25, d, rng)
    theta_db, theta_train, theta_test = (rng.uniform(0.3, 0.7, k) for k in (N, 40, 25))
    full = rt_engine.build_seed_state(2, db, theta_db, train, theta_train, [100, S, N], 2048)
    red = fb.build_reduced_state(2, db, theta_db, train, theta_train, S)
    cfgs = [(1, 512), (4, 128), (8, 64), (16, 128)]
    a = rt_engine.process_chunk(full, test, theta_test, np.arange(25),
                                rt_engine.ChunkOptions(configs=cfgs, nu_grid=[1, 8, 32], ref_S=S))
    b = rt_engine.process_chunk(red, test, theta_test, np.arange(25),
                                rt_engine.ChunkOptions(configs=cfgs, nu_grid=[1, 8, 32], ref_S=S, union_all_seeds=[S]))
    aa = a["agg"][a["agg"]["S"] == S].reset_index(drop=True)
    bb = b["agg"].reset_index(drop=True)
    keys = ["config", "pop", "nu"]
    mg = aa.merge(bb, on=keys, suffixes=("_f", "_r"))
    assert len(mg) == len(bb) == len(cfgs) * 2 * 3
    for c in fb.SUM_COLS:
        assert np.allclose(mg[f"{c}_f"], mg[f"{c}_r"], rtol=1e-9, atol=1e-12, equal_nan=True), c
    ra = a["rows"].sort_values(["config", "nu", "query_id"]).reset_index(drop=True)
    rb = b["rows"].sort_values(["config", "nu", "query_id"]).reset_index(drop=True)
    assert np.allclose(ra["r_hat"], rb["r_hat"], rtol=1e-5) and (ra["n_distinct_total"] == rb["n_distinct_total"]).all()

def test_metrics_from_sums_match_per_query_recomputation():
    rng = np.random.default_rng(3)
    theta = rng.uniform(0.1, 0.4, 400)
    r = 0.04 + 0.7 * theta + rng.normal(0, 0.02, 400)
    rel = (r - theta) / theta
    row = dict(config="c", L=1, m=64, S=10, pop="sample_train", nu=8, seed=0, distinct_scope="all_seeds", n=400,
               sum_theta=theta.sum(), sum_rhat=r.sum(), sum_theta2=(theta ** 2).sum(), sum_rhat2=(r ** 2).sum(),
               sum_theta_rhat=(theta * r).sum(), sum_relerr=rel.sum(), sum_relerr2=(rel ** 2).sum(), n_distinct=400,
               sum_distinct_total=400 * 7.0, sum_distinct_train=400 * 3.0, sum_distinct_sample=400 * 4.0,
               sum_train_share=400 * 0.5)
    m = fb.metrics_from_sums(pd.DataFrame([row])).iloc[0]
    assert np.isclose(m["slope"], np.polyfit(theta, r, 1)[0]) and np.isclose(m["reverse_slope"], np.polyfit(r, theta, 1)[0])
    assert np.isclose(m["r_squared"], np.corrcoef(theta, r)[0, 1] ** 2)
    assert np.isclose(m["mean_error_pct"], 100 * rel.mean()) and np.isclose(m["sd_error_pct"], 100 * rel.std())
    assert np.isclose(m["mean_distinct"], 7.0) and np.isclose(m["train_share"], 0.5) and m["estimate"] == "sample+train"

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
