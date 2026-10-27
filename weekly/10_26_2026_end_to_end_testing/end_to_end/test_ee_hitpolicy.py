#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_hitpolicy (toy streams from the spec, integer arithmetic). Run with pytest or
`python3 -B test_ee_hitpolicy.py`."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ee_config as cfg
import ee_hitpolicy as hp

def units(*tenths_est):
    """Estimated recall given as multiples of 0.1 (counts out of K = 10) to integer units of 0.05."""
    return hp.counts_to_units(np.array(tenths_est), 10)

def est_from_deficits(*deficits_tenths):
    return units(*[10 - d for d in deficits_tenths])

#-------------------------------------------------------------------------------

def test_units_and_slack_are_integers():
    assert [cfg.target_units(T) for T in cfg.TARGETS] == [16, 18, 19]
    assert [cfg.per_step_slack_units(T) for T in cfg.TARGETS] == [4, 2, 1]
    assert [cfg.slack_units(T, 2500) for T in cfg.TARGETS] == [10000, 5000, 2500]
    assert hp.counts_to_units(np.array([0, 3, 10, 14]), 10).tolist() == [0, 6, 20, 28]
    try:
        hp.counts_to_units(np.array([1]), 3)
        raise RuntimeError("expected AssertionError")
    except AssertionError:
        pass

def test_hp1_threshold_exact():
    est = units(10, 9, 8, 7, 11)
    assert hp.run_policy("HP1", est, 0.8).tolist() == [True, True, True, False, True]   # 16 units = 0.8
    assert hp.run_policy("HP1", est, 0.95).tolist() == [True, False, False, False, True]  # 19 units: only est >= 1.0

def test_stream_hp1_serves_5_hp2_serves_10_hindsight_10():
    est = units(*([10] * 5 + [6] * 5))  # five at 1.0, five at 0.6
    assert hp.run_policy("HP1", est, 0.8).sum() == 5
    assert hp.run_policy("HP2", est, 0.8).sum() == 10
    assert hp.hindsight_count(est, 0.8) == 10

def test_stream_hp2_12_hindsight_14_hp3_14():
    est = est_from_deficits(*([0] * 4 + [9] + [3] * 10))  # q5 has deficit 0.9, q6 to q15 have 0.3
    assert hp.run_policy("HP2", est, 0.8).sum() == 12
    assert hp.hindsight_count(est, 0.8) == 14
    cap_units = 6  # 0.3 at T = 0.8
    hits = hp.run_policy("HP3", est, 0.8, cap_units)
    assert hits.sum() == 14 and not hits[4]

def test_negative_deficit_never_creates_slack():
    est = units(15, 15, 15, 2)  # three estimates above 1.0 (negative deficit), then a deficit of 0.8
    d, a = hp.deficits(est)
    assert d.tolist() == [-10, -10, -10, 16] and a.tolist() == [0, 0, 0, 16]
    # slack at t = 4 is 4 * 4 = 16 units: a = 16 fits exactly; negative deficits added nothing on top of it
    assert hp.run_policy("HP2", est, 0.8).tolist() == [True, True, True, True]
    est2 = units(15, 15, 15, 1)  # deficit 0.9 = 18 units > 16, must miss
    assert hp.run_policy("HP2", est2, 0.8).tolist() == [True, True, True, False]
    # hindsight: negative deficits are clipped, they do not extend the prefix
    # n = 10, slack 40 units: sorted a = 0, 0, 0, 18, 18, 18, ...; running sums 0, 0, 0, 18, 36, 54: five fit
    assert hp.hindsight_count(units(*([15] * 3 + [1] * 7)), 0.8) == 3 + 2

def test_learn_cap_and_hindsight_on_toy_train_set():
    train = est_from_deficits(0, 0, 0, 3, 3, 3, 9)  # d units 0,0,0,6,6,6,18
    train = np.concatenate([train, units(15)])      # one estimate above 1.0, d = -10
    # n = 8, T = 0.8: slack 32 units; sorted d: -10, 0, 0, 0, 6, 6, 6, 18 -> running a: 0,0,0,0,6,12,18,36
    assert hp.learn_cap(train, 0.8) == 6
    assert hp.hindsight_count(train, 0.8) == 7
    assert hp.learn_cap(units(1, 1, 1), 0.95) == 0  # nothing fits, cap is 0
    assert hp.learn_cap(units(10, 10, 10), 0.95) == 0

def test_cap_is_floored_at_zero_and_sums_use_clipped_deficits():
    # every train query overestimated: all accepted, the largest accepted d is negative, the cap is floored at 0
    over = units(12, 13, 11, 15)
    assert hp.deficits(over)[0].max() < 0 and hp.learn_cap(over, 0.9) == 0
    # overestimated queries rank first by unclipped d but add nothing to the running sum (clipped a = 0)
    est = units(*([15] * 6 + [9] * 3 + [5] * 3))  # d = -10 x 6, 2 x 3, 10 x 3; n = 12, T = 0.9: slack 24 units
    d, a = hp.deficits(est)
    assert hp.learn_cap(est, 0.9) == 10         # running a: 0 x 6, 2, 4, 6, 16 (<= 24) accepted, then 26 (> 24) stops
    assert hp.learn_cap(est, 0.95) == 2         # slack 12 units, 6 + 10 > 12 for the 10s
    # negative deficits must not buy room for larger ones
    est2 = units(*([20] * 5 + [4] * 5))         # d = -20 x 5, 12 x 5; n = 10, T = 0.9: slack 20 units
    assert hp.learn_cap(est2, 0.9) == 12        # one d = 12 fits (12 <= 20), two do not (24 > 20)
    assert (hp.deficits(est2)[1][:5] == 0).all()
    # the stream gives no credit for overestimation either
    assert hp.run_policy("HP3", est2, 0.9, 12).sum() == 5 + 1 + 0

def test_hp3_uses_unclipped_deficit_for_cap_and_learned_cap_limits_hits():
    est = est_from_deficits(0, 6, 2, 6, 0)
    all_hp2 = hp.run_policy("HP2", est, 0.8)
    capped = hp.run_policy("HP3", est, 0.8, 4)  # cap 0.2 refuses the two 0.6 deficits
    assert capped.tolist() == [True, False, True, False, True] and capped.sum() <= all_hp2.sum()

def test_delivered_recall_and_running_metrics_exact():
    est = units(10, 10, 4, 4)
    true = np.array([10, 6, 0, 5])
    hits = np.array([True, True, True, False])
    assert hp.delivered_units(hits, true, 10).tolist() == [20, 12, 0, 20]  # hit: true recall, miss: 1
    m = hp.stream_metrics(hits, true, 0.8, burn_in=1)
    assert np.isclose(m["hit_rate"], 0.75) and np.isclose(m["achieved_recall"], 52 / 80)
    # running Y_t: 1.0, 0.8, 0.5333..., 0.65 ; below T = 0.8 at t = 3 and 4 (t = 2 is exactly 0.8, not below)
    assert np.isclose(m["y_min"], 32 / 60) and np.isclose(m["frac_below_T"], 0.5)
    assert np.isclose(m["frac_below_T_after_burn_in"], 2 / 3)
    assert np.isclose(m["mean_hit_recall"], (1.0 + 0.6 + 0.0) / 3)
    assert np.isclose(m["frac_hits_below_half"], 1 / 3) and np.isclose(m["frac_hits_zero_recall"], 1 / 3)
    assert np.isclose(m["p10_hit_recall"], np.percentile([1.0, 0.6, 0.0], 10))
    none = hp.stream_metrics(np.zeros(4, dtype=bool), true, 0.8)
    assert np.isnan(none["mean_hit_recall"]) and np.isclose(none["achieved_recall"], 1.0)

def test_hp3f_adds_the_estimated_recall_floor():
    # T = 0.8: slack 4 units per step. Six perfect estimates, then est 0.3 (d = 14 units): HP3 serves it, HP3f refuses
    est = units(*([10] * 6 + [3, 5, 4]))
    assert hp.run_policy("HP3", est, 0.8, 20).tolist()[6:] == [True, True, True]
    f = hp.run_policy("HP3f", est, 0.8, 20)
    assert f.tolist()[6:] == [False, True, False]        # est 0.5 (10 units) passes the floor, 0.3 and 0.4 do not
    assert (f <= hp.run_policy("HP3", est, 0.8, 20)).all()
    # the floor does not change the cap rule: a query above the floor but with d > cap is still refused
    assert not hp.run_policy("HP3f", units(*([10] * 6 + [6])), 0.8, 6)[6]   # est 0.6 passes the floor, d = 8 > cap 6
    try:
        hp.run_policy("HP3f", est, 0.8)
        raise RuntimeError("expected AssertionError")
    except AssertionError:
        pass

def test_secondary_metrics_hit_rate_with_recall_at_least_half():
    true = np.array([10, 6, 0, 5, 4])
    hits = np.array([True, True, True, True, False])
    m = hp.stream_metrics(hits, true, 0.8, burn_in=1)
    assert np.isclose(m["hit_rate"], 0.8) and np.isclose(m["hit_rate_recall_ge_half"], 0.6)   # counts 10, 6, 5 qualify
    assert np.isclose(m["mean_true_recall_of_hits"], (1.0 + 0.6 + 0.0 + 0.5) / 4)
    assert m["mean_true_recall_of_hits"] == m["mean_hit_recall"]
    none = hp.stream_metrics(np.zeros(5, dtype=bool), true, 0.8)
    assert none["hit_rate_recall_ge_half"] == 0 and np.isnan(none["mean_true_recall_of_hits"])

def test_stream_order_is_seeded_permutation():
    a, b, c = hp.stream_order(3, 50), hp.stream_order(3, 50), hp.stream_order(4, 50)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and sorted(a.tolist()) == list(range(50))

def test_evaluate_pool_rows_and_proxy_gap_logging():
    rng = np.random.default_rng(0)
    n = 400
    true_t = rng.integers(0, 11, n)
    est_t = np.clip(true_t + rng.integers(-1, 3, n), 0, 14)
    true_r = rng.integers(0, 11, n)
    est_r = np.clip(true_r + rng.integers(-1, 3, n), 0, 14)
    rows = hp.evaluate_pool(1, 0.9, est_t, true_t, est_r, true_r, oracle_arm=False)
    assert [r["policy"] for r in rows] == ["HP1", "HP2", "HP3", "HP3f"]
    hp3 = rows[2]
    assert np.isfinite(hp3["learned_cap_units"]) and 0.0 <= hp3["train_hit_fraction"] <= 1.0
    assert hp3["test_hit_fraction"] == hp3["hit_rate"] and np.isnan(rows[0]["learned_cap_units"])
    assert np.isclose(rows[0]["est_signed_error"], ((est_t - true_t) / 10).mean())
    assert np.isclose(rows[0]["est_mae"], np.abs((est_t - true_t) / 10).mean())
    assert rows[0]["hindsight_hit_rate_lsh"] >= 0 and rows[0]["hindsight_hit_rate_oracle"] >= 0
    oracle = hp.evaluate_pool(1, 0.9, true_t, true_t, true_r, true_r, oracle_arm=True)
    assert np.isnan(oracle[0]["hindsight_hit_rate_lsh"]) and "est_mae" not in oracle[0]
    assert oracle[1]["hit_rate"] <= oracle[1]["hindsight_hit_rate_oracle"] + 1e-12
    hp3f = rows[3]
    assert hp3f["learned_cap_units"] == hp3["learned_cap_units"] and np.isfinite(hp3f["train_hit_fraction"])
    assert hp3f["hindsight_hit_rate_oracle"] == hp3["hindsight_hit_rate_oracle"]  # hindsight unchanged

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
