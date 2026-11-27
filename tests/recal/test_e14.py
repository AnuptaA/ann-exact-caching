import json
from statistics import NormalDist

import numpy as np
import pytest

from recal import prepare
from recal.micro import e10, e14, intervals, world as world_mod

PARAMS = {"H": 900, "cold_start": 500, "min_ess": 30.0}
BUDGET = 0.6
SMALL = dict(seeds=1, ess_target=40, log=lambda *_: None)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("e14")
    cfg = prepare.tiny_config(("tiny-a",))
    cfg["workloads"]["tiny-a"]["budgets"] = [BUDGET]
    prepare.make_tiny(cfg, root)
    mp = pytest.MonkeyPatch()
    mp.chdir(root)
    yield world_mod.build_world("results/online/tiny-a", "trace", BUDGET, 4, name="tiny-a", params=PARAMS)
    mp.undo()


@pytest.fixture(scope="module")
def result(world):
    return e14.run(world, **SMALL)


def test_inverse_probability_weights_keep_the_mean_unbiased():
    rng = np.random.default_rng(3)
    y = rng.random(2000)
    p = 0.25
    estimates = []
    for _ in range(400):
        kept = rng.random(len(y)) < p
        estimates.append((y[kept] / p).sum() / len(y))
    assert abs(np.mean(estimates) - y.mean()) < 3 * np.std(estimates) / np.sqrt(len(estimates)) + 1e-9


def test_bound_matches_the_two_weight_version():
    rng = np.random.default_rng(5)
    z = NormalDist().inv_cdf(0.95)
    for _ in range(20):
        n = 60
        y = rng.random(n)
        w = np.where(rng.random(n) < 0.2, 10.0, 1.0)
        amb = rng.random(n) < 0.3
        certain = ~amb & (rng.random(n) < 0.8)
        ours = e14.worst_case_bound(y, w, certain, amb, z)
        ref = intervals.worst_case_bound(y, w, certain, amb, z)
        assert ours["lcb"] == pytest.approx(ref["lcb"]) and ours["ess"] == pytest.approx(ref["ess"])


def test_a_coarse_item_reaching_the_threshold_is_ambiguous():
    n = 200
    y = np.full(n, 0.95)
    low = np.full(n, 0.9)
    high = np.full(n, 0.9)
    w = np.ones(n)
    edges = np.array([0.5, 0.8])
    clean = e14.scan(edges, low, high, y, w, 0.9, 0.05, 10)
    assert clean["lam"] == 0.5 and clean["n_ambiguous"] == 0
    # one C item (weight 1/p) scoring below every threshold but whose upper end reaches them
    low2, high2 = np.r_[low, 0.1], np.r_[high, 0.85]
    y2, w2 = np.r_[y, 0.0], np.r_[w, 4.0]
    out = e14.scan(edges, low2, high2, y2, w2, 0.9, 0.05, 10)
    assert out["n_ambiguous"] == 1
    z = NormalDist().inv_cdf(0.95)
    certain = low2 >= 0.5
    with_it = e14.worst_case_bound(y2, w2, certain, ~certain, z)
    without = e14.worst_case_bound(y2, w2, certain, np.zeros(len(y2), bool), z)
    assert with_it["lcb"] < without["lcb"]


def test_three_weight_classes_are_supported():
    z = NormalDist().inv_cdf(0.95)
    y = np.array([0.9, 0.8, 0.2, 0.1, 0.5, 1.0])
    w = np.array([1.0, 10.0, 4.0, 1.0, 10.0, 4.0])
    amb = np.array([False, True, True, True, True, True])
    out = e14.worst_case_bound(y, w, ~amb, amb, z)
    assert out["n_ambiguous"] == 5 and np.isfinite(out["lcb"])
    with pytest.raises(ValueError):
        intervals.worst_case_bound(y, w, ~amb, amb, z)


def brute_force(y, w, certain, amb, z):
    """Minimum lower confidence bound and minimum ESS over every served set between certain and certain + amb."""
    items = np.flatnonzero(amb)
    best, best_ess = np.inf, np.inf
    for mask in range(1 << len(items)):
        pick = certain.copy()
        for j, i in enumerate(items):
            if mask >> j & 1:
                pick[i] = True
        if not pick.any():
            continue
        yy, ww = y[pick], w[pick]
        W, SW2 = ww.sum(), (ww * ww).sum()
        mean = (ww * yy).sum() / W
        ess = W * W / SW2
        denom = max(W - SW2 / W, 1e-12)
        g = (ww * (yy - mean) ** 2).sum()
        best = min(best, mean - z * np.sqrt(g / denom) / np.sqrt(ess))
        best_ess = min(best_ess, ess)
    return best, best_ess


def small_case(rng, n=14, weights=(1.0, 10.0, 4.0)):
    y = rng.random(n)
    w = rng.choice(weights, n)
    amb = rng.random(n) < 0.55
    certain = ~amb & (rng.random(n) < 0.7)
    return y, w, certain, amb


def test_chunked_and_unchunked_agree_exactly():
    rng = np.random.default_rng(11)
    z = NormalDist().inv_cdf(0.95)
    for _ in range(15):
        y, w, certain, amb = small_case(rng)
        whole = e14.worst_case_bound(y, w, certain, amb, z, chunk=10**9)
        for chunk in (1, 7, 64):
            got = e14.worst_case_bound(y, w, certain, amb, z, chunk=chunk)
            assert got["lcb"] == whole["lcb"] and got["ess"] == whole["ess"] and got["lcb_prefix"] == whole["lcb_prefix"]
        assert whole["bound_exact"] and whole["cells"] >= 1


def test_exact_bound_is_at_most_the_true_minimum():
    rng = np.random.default_rng(12)
    z = NormalDist().inv_cdf(0.95)
    for _ in range(40):
        y, w, certain, amb = small_case(rng)
        truth, truth_ess = brute_force(y, w, certain, amb, z)
        got = e14.worst_case_bound(y, w, certain, amb, z)
        if got is None:
            continue
        assert got["lcb"] <= truth + 1e-9 <= got["lcb_prefix"] + 2e-9
        assert got["ess"] == pytest.approx(truth_ess)


def test_forced_fallback_is_a_valid_lower_bound_and_is_marked():
    rng = np.random.default_rng(13)
    z = NormalDist().inv_cdf(0.95)
    seen = 0
    for trial in range(60):
        y, w, certain, amb = small_case(rng)
        truth, truth_ess = brute_force(y, w, certain, amb, z)
        for blocks in (1, 4, 30, 10**6):
            got = e14.worst_case_bound(y, w, certain, amb, z, cell_budget=0, blocks=blocks)
            if got is None:
                assert not (certain | amb).any()
                continue
            seen += 1
            assert got["bound_exact"] is False and got["cells"] >= 1
            assert got["lcb"] <= truth + 1e-9
            assert got["ess"] <= truth_ess + 1e-9
    assert seen > 100


def test_fallback_scan_never_certifies_looser_than_the_exact_scan():
    rng = np.random.default_rng(14)
    n = 300
    y = np.clip(rng.normal(0.93, 0.08, n), 0, 1)
    low = rng.random(n)
    high = np.minimum(low + 0.15, 1.5)
    w = rng.choice([1.0, 10.0], n)
    edges = np.linspace(0.2, 0.9, 30)
    exact = e14.scan(edges, low, high, y, w, 0.9, 0.05, 10)
    relaxed = e14.scan(edges, low, high, y, w, 0.9, 0.05, 10, cell_budget=0, blocks=2000)
    assert exact["bound_exact"] and not relaxed["bound_exact"] and relaxed["cells"] > 0
    assert relaxed["lam"] is None or exact["lam"] is not None and relaxed["lam"] >= exact["lam"]


def test_a_large_grid_does_not_raise():
    n = 3000
    rng = np.random.default_rng(15)
    y, w = rng.random(n), rng.choice([1.0, 10.0, 4.0], n)
    certain = np.arange(n) < 600
    out = e14.worst_case_bound(y, w, certain, ~certain, 1.645, cell_budget=10**6)
    assert out["bound_exact"] is False and out["cells"] > 10**6 and np.isfinite(out["lcb"])


def test_precision_is_decided_once_at_arrival():
    lab = dict(B=np.array([True, True, True, False]), audited=np.array([False, True, False, False]))
    keep = np.array([0.0, 0.9, 0.9, 0.0])
    cl = e14.classes(lab, np.array([0.2, 0.1, 0.9, 0.9]), np.full(4, 0.5), keep, 0.25)
    assert cl["P"].tolist() == [False, True, True, False]
    assert cl["C"].tolist() == [True, False, False, False] and cl["kept"].tolist() == [True, False, False, False]


def test_half_a_items_store_no_vector():
    assert e14.item_bytes(100, 10)["A"] == e14.item_bytes(1000, 10)["A"] == 4 * 10 + 31
    assert e14.item_bytes(100, 10)[16] == 100 * 2 + 4 * 10 + 33


def test_shifted_rescoring_excludes_items_inside_the_shifted_history():
    ok = e14.eligible_rows(10, 5, 3)
    assert ok.tolist() == [False] * 3 + [True] * 7


def test_run_rows_respect_classes_and_shifts(world, result):
    rows, summary = result
    assert rows and summary["W"] > 0
    offsets = summary["offsets"]
    assert set(offsets) == {"none"} | {f"{k}{s:g}" for k in ("shift", "drop") for s in e14.SHIFTS}
    assert all(0 < v <= 1 for v in summary["pool_overlap"].values())
    by = {}
    for r in rows:
        assert r["start"] + r["W"] >= offsets[r["shift"]]
        by.setdefault((r["window"], r["start"], r["config"]), {})[r["shift"]] = r
    for changes in by.values():
        if "none" not in changes:
            continue
        for label, r in changes.items():
            # rescored windows hold only items after the shifted history, with the arrival classes
            assert r["n_B"] <= changes["none"]["n_B"] and r["n_P"] <= changes["none"]["n_P"]
            assert r["n_C_kept"] <= changes["none"]["n_C_kept"]
            if label.startswith("drop"):
                assert r["n_P"] == changes["none"]["n_P"]
    for r in rows:
        assert r["n_P"] + r["n_C"] == r["n_B"] and r["n_C_kept"] <= r["n_C"]
    assert summary["verdict"]["primary"] in summary["verdict"]["configs"]


def test_dropping_entries_shrinks_the_cache(world):
    feats = e10.context(world)["feats"]
    fewer = e14.dropped_features(feats, 0.25)
    assert len(fewer.centers) < len(feats.centers) and len(fewer.slots) == len(fewer.centers)
    assert set(fewer.pool_ids.tolist()) <= set(feats.pool_ids.tolist())
    assert len(feats.centers) == feats.table.n


def test_e14_is_deterministic_for_a_seed(world, result):
    again = e14.run(world, **SMALL)
    assert json.dumps(again[0], default=str) == json.dumps(result[0], default=str)


def test_quick_grid_skips_4_bit_and_the_shift_arms(world):
    rows, summary = e14.run(world, grid=e14.QUICK_GRID, **SMALL)
    assert rows and {r["shift"] for r in rows} == {"none", "drop0.1", "drop0.25"}
    assert {r["coarse"] for r in rows} == {8} and {r["p"] for r in rows} == {0.1, 0.25}
    assert {r["margin"] for r in rows} == {0.05}
    assert summary["grid"] == e14.QUICK_GRID.key() and summary["verdict"]["primary"] == "P16_C8_p0.25_d0.05"
    b = summary["bound"]
    assert b["n_windows"] > 0 and b["exact_bound_share"] == 1.0 and b["effective_windows"] > 0
    assert all(r["bound_exact"] and r["bound_exact_ref"] for r in rows)


def test_full_grid_is_todays_grid():
    assert e14.FULL_GRID.arms == ("none", "drop0.1", "drop0.25", "shift0.1", "shift0.25")
    assert e14.FULL_GRID.coarse_bits == (4, 8) and e14.FULL_GRID.p_values == (0.1, 0.25, 0.5)
    assert e14.FULL_GRID.margins == (0.0, 0.025, 0.05, 0.1) and e14.FULL_GRID.primary == e14.PRIMARY
    with pytest.raises(ValueError):
        e14.Grid(arms=("drop0.1",))


def test_fallback_is_reported_in_the_summary(world, monkeypatch):
    monkeypatch.setattr(e14, "CELL_BUDGET", 1)
    monkeypatch.setattr(e14.worst_case_bound, "__defaults__", (1, e14.CHUNK_CELLS, e14.FALLBACK_BLOCKS))
    rows, summary = e14.run(world, grid=e14.QUICK_GRID, **SMALL)
    assert summary["bound"]["exact_bound_share"] < 1.0 and summary["bound"]["max_cells"] > 1
