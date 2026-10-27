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
    assert [r["policy"] for r in rows] == ["HP1", "HP2", "HP3"]
    hp3 = rows[2]
    assert np.isfinite(hp3["learned_cap_units"]) and 0.0 <= hp3["train_hit_fraction"] <= 1.0
    assert hp3["test_hit_fraction"] == hp3["hit_rate"] and np.isnan(rows[0]["learned_cap_units"])
    assert np.isclose(rows[0]["est_signed_error"], ((est_t - true_t) / 10).mean())
    assert np.isclose(rows[0]["est_mae"], np.abs((est_t - true_t) / 10).mean())
    assert rows[0]["hindsight_hit_rate_lsh"] >= 0 and rows[0]["hindsight_hit_rate_oracle"] >= 0
    oracle = hp.evaluate_pool(1, 0.9, true_t, true_t, true_r, true_r, oracle_arm=True)
    assert np.isnan(oracle[0]["hindsight_hit_rate_lsh"]) and "est_mae" not in oracle[0]
    assert oracle[1]["hit_rate"] <= oracle[1]["hindsight_hit_rate_oracle"] + 1e-12

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
