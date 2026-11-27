import json

import numpy as np
import pytest

from recal import prepare
from recal.micro import e10, e15, intervals as iv, world as world_mod

PARAMS = {"H": 900, "cold_start": 500, "min_ess": 30.0}
BUDGET = 0.6
SMALL = dict(degrees=(8,), efs=(32, 64), log=lambda *_: None)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("e15")
    cfg = prepare.tiny_config(("tiny-a",))
    cfg["workloads"]["tiny-a"]["budgets"] = [BUDGET]
    prepare.make_tiny(cfg, root)
    mp = pytest.MonkeyPatch()
    mp.chdir(root)
    yield world_mod.build_world("results/online/tiny-a", "trace", BUDGET, 4, name="tiny-a", params=PARAMS)
    mp.undo()


@pytest.fixture(scope="module")
def engine(world):
    feats = iv.Features(world, c=e15.C_FIT)
    return e15.Engine(world, feats), feats


@pytest.fixture(scope="module")
def sample(world, engine):
    eng, _ = engine
    vec, truth, _, keep = e10.held_queries(world)
    ids = world.heldout()["unique"][keep]
    n = min(120, len(vec))
    return vec[:n], truth[:n], np.asarray(world.top[ids[:n]], np.int64)


@pytest.fixture(scope="module")
def data(engine, sample):
    return engine[0].compute(*sample)


@pytest.fixture(scope="module")
def result(world):
    return e15.run(world, **SMALL)


def test_v2_all_equals_the_global_count_and_the_golden_features(engine, data, sample):
    eng, feats = engine
    assert (data[f"cnt{eng.n}"] == data["glob"]).all()
    e15.self_check(eng, feats, data, sample[0])


def test_counts_do_not_decrease_with_c(engine, data):
    eng, _ = engine
    cs = eng.count_cs
    for a, b in zip(cs, cs[1:]):
        assert (data[f"cnt{a}"] <= data[f"cnt{b}"]).all()


def test_earliest_rank_pass_matches_a_brute_force_loop(world, engine, sample):
    eng, feats = engine
    q = sample[0][:15]
    got = eng.compute(q, sample[1][:15], sample[2][:15])
    from recal.micro.e4 import dist64
    Dc = dist64(q, eng.centers)
    Ds = dist64(q, eng.shell_vec)
    for i in range(len(q)):
        nr = got["nr"][i]
        order = np.lexsort((np.arange(eng.n), Dc[i]))
        for c in eng.count_cs:
            seen = set()
            for e in order[:c]:
                seen.update(feats.slots[e].tolist())
            count = sum(1 for s in seen if Ds[i][s] < nr)
            assert got[f"cnt{c}"][i] == count


def test_index_with_large_ef_matches_the_global_count(engine, data, sample):
    eng, _ = engine
    index = e15.build_index(eng.shell_vec, 16)
    cap = len(eng.shell_vec)
    out = e15.index_counts(eng, index, sample[0], data["nr"], cap, cap, 1)
    assert (out["count"] == data["glob"]).all()


def test_rows_cover_every_variant_and_report_the_rules(result):
    rows, summary = result
    names = {r["variant"] for r in rows}
    assert {"V0", "V1", "V2_all", "V3", "V3_cfit16", "V3_cfit128", "V6"} <= names
    assert {f"V4_cap{m * 10}" for m in e15.CAP_MULTIPLES} <= names
    assert any(n.startswith("V5_d8_") for n in names)
    for r in rows:
        assert r["certified"] in (True, False) and "hit_rate" in r and "oracle_hit_rate" in r
    assert summary["shell_pool"]["shell_slots"] > 0 and summary["shell_pool"]["index"]
    assert {"v3_recovery", "v5_any_within_1pt", "v2_smallest_c_within_1pt_of_v0"} <= set(summary["rules"])
    v5 = [r for r in rows if r["family"] == "V5"]
    assert all(r["us_per_query"] > 0 and 0 <= r["flips_vs_v3"] <= 1 for r in v5)


def test_e15_is_deterministic(world, result):
    again = e15.run(world, **SMALL)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "us_per_query"} for r in rows]  # noqa: E731
    assert json.dumps(strip(again[0]), default=str) == json.dumps(strip(result[0]), default=str)
