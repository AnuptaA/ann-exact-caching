#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for ee_tables_f. Run with pytest or `python3 -B test_ee_tables_f.py`."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import ee_tables_f as tf

def fake_rows():
    recs = []
    for seed, (hit, orc, ymin) in enumerate([(0.5, 0.6, 0.01), (0.4, 0.6, -0.03), (0.6, 0.7, 0.02)]):
        recs.append(dict(dataset="sift", regime="lam0", budget=500, T=0.9, pool="popularity", policy="HP3", seed=seed,
                         hit_rate=hit, or_hit_rate=orc, hit_rate_recall_ge_half=hit - 0.1, or_hit_rate_recall_ge_half=orc - 0.1,
                         mean_true_recall_of_hits=0.7, or_mean_true_recall_of_hits=0.8, achieved_recall=0.9,
                         or_achieved_recall=0.9, y_min_minus_T=ymin))
    return pd.DataFrame(recs)

def test_gap_is_per_seed_lsh_minus_oracle_and_worst_ymin_is_the_minimum():
    c = tf.cell_summary_f(fake_rows()).iloc[0]
    assert np.isclose(c.gap_hit_rate, np.mean([-0.1, -0.2, -0.1])) and np.isclose(c.gap_mean_true_recall_of_hits, -0.1)
    assert np.isclose(c.y_min_minus_T_worst, -0.03) and np.isclose(c.y_min_minus_T, 0.0) and c.n_seeds == 3

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
