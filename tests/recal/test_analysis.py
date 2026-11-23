import numpy as np

from recal import analysis
from recal import records as rec
from recal.analysis.rates import hit_mask


def synthetic_cols():
    n = 12
    path = np.array([rec.MINI, rec.MAIN, rec.MAIN, rec.MISS, rec.MAIN, rec.ABSTAIN] * 2, dtype=np.int8)
    recall = np.array([1, 1.0, 0.8, 0.9, 0.6, 1.0] * 2, dtype=np.float32)
    cand = np.array([np.nan, 1.0, 0.8, 0.95, 0.6, 0.7] * 2, dtype=np.float32)
    contrib = np.array([-1, 1, 3, 2, 2, 1] * 2, dtype=np.int32)
    cols = {name: np.zeros(n, dtype=dtype) for name, dtype in rec.FIELDS.items()}
    cols.update(path=path, recall=recall, cand_recall=cand, n_contrib=contrib)
    return cols


def test_rates_and_composition():
    cols = synthetic_cols()
    r = analysis.rates(cols)
    assert r["hit_rate"] == 8 / 12 and r["mini_hit_rate"] == 2 / 12 and r["main_hit_rate"] == 6 / 12
    assert np.isclose(r["recall_of_main_hits"], np.mean([1.0, 0.8, 0.6]))
    assert r["miss_rate"] == r["abstain_rate"] == 2 / 12
    c = analysis.hit_composition(cols)
    assert c["hits"] == 8 and c["mini"] == 2 / 8 and c["one_entry"] == 2 / 8 and c["several_entries"] == 4 / 8


def test_guarantee_windows_use_main_path_requests_only():
    cols = synthetic_cols()
    g = analysis.guarantee_table(cols, 0.9, window=5)
    assert g["n_windows"] == 2
    assert np.isclose(g["recall_of_hits"][0], np.mean([1.0, 0.8, 0.6]))
    assert g["share_below_target"] == 1.0
    assert np.isclose(g["mean_margin"], np.mean(g["recall_of_hits"]) - 0.9)


def test_oracle_admits_best_requests_while_mean_holds():
    cols = synthetic_cols()
    o = analysis.oracle(cols, 0.9)
    # main-path candidate recalls sorted: 1,1,.95,.95,.8,.8,.7,.7,.6,.6
    assert o["admitted"] == 6 and np.isclose(o["mean_recall_admitted"], 5.5 / 6)
    assert o["hit_rate"] == (6 + 2) / 12
    assert analysis.oracle(cols, 1.01)["admitted"] == 0


def test_hit_rate_interpolation_inside_the_curve_only():
    pts = [(0.80, 0.2), (0.90, 0.4), (0.95, 0.5)]
    assert np.isclose(analysis.hit_rate_at_recall(pts, 0.85), 0.3)
    assert np.isnan(analysis.hit_rate_at_recall(pts, 0.7))
    assert np.isnan(analysis.hit_rate_at_recall(pts[:1], 0.8))


def test_hit_mask_matches_paths():
    cols = synthetic_cols()
    assert hit_mask(cols).sum() == 8
