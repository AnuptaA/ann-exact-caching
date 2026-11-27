import json
from statistics import NormalDist

import numpy as np
import pytest

from recal import prepare
from recal.micro import e14, intervals, world as world_mod

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
    offsets = dict(zip([0.0] + [float(s) for s in e14.SHIFTS], summary["offsets"]))
    by = {}
    for r in rows:
        if r["shift"] > 0:
            assert r["start"] + r["W"] >= offsets[r["shift"]]
        by.setdefault((r["window"], r["start"], r["config"]), {})[r["shift"]] = r
    for shifts in by.values():
        if 0.0 not in shifts:
            continue
        for share, r in shifts.items():
            # rescored windows hold only items after the shifted history, with the arrival classes
            assert r["n_B"] <= shifts[0.0]["n_B"] and r["n_P"] <= shifts[0.0]["n_P"]
            assert r["n_C_kept"] <= shifts[0.0]["n_C_kept"]
    for r in rows:
        assert r["n_P"] + r["n_C"] == r["n_B"] and r["n_C_kept"] <= r["n_C"]
    assert set(summary["verdict"]) >= {"memory_ok", "hit_ok", "ambiguous_ok"}


def test_e14_is_deterministic_for_a_seed(world, result):
    again = e14.run(world, **SMALL)
    assert json.dumps(again[0], default=str) == json.dumps(result[0], default=str)
