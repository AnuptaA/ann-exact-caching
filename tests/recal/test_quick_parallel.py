import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from recal import history_mode, policy, prepare
from recal.micro import (common, e1, e3, e4, e7, e8, e10, e11, e12, e13, fit, golden, intervals, intruder_check,
                         preflight, provenance, topk, units as units_mod, world as world_mod)
from scripts.micro import inputs as micro_inputs, run as micro_run, summarize
from tests.recal.helpers import ID_BASE, build_core, exact_order
from tests.recal.reference import lru_old

PARAMS = {"H": 900, "cold_start": 500, "min_ess": 30.0}
BUDGET = 0.6


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("quick")
    cfg = prepare.tiny_config(("tiny-a",))
    cfg["workloads"]["tiny-a"]["budgets"] = [0.4, BUDGET]
    prepare.make_tiny(cfg, root)
    (root / "matrix.json").write_text(json.dumps(cfg))
    mp = pytest.MonkeyPatch()
    mp.chdir(root)
    yield root
    mp.undo()


def new_world(workers: int = 1):
    world = world_mod.build_world("results/online/tiny-a", "trace", BUDGET, 4, name="tiny-a", params=PARAMS)
    world.workers = workers
    return world


@pytest.fixture(scope="module")
def world(tiny):
    return new_world()


def run_cli(tiny, *extra, out="out"):
    return micro_run.main(["--dataset", "tiny", "--config", str(tiny / "matrix.json"), "--out", str(tiny / out),
                           "--params", json.dumps(PARAMS), *extra])


def same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def close(a, b, rtol: float = 1e-9) -> bool:
    """Structures equal, floats within rtol (BLAS summation order differs with the worker count)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], rtol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b) or abs(a - b) <= rtol * max(abs(a), abs(b))
    return a == b


TIMING = ("us_per_query", "rerank_us_per_query", "build_s")


def untimed(value):
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items() if k not in TIMING}
    if isinstance(value, (list, tuple)):
        return [untimed(v) for v in value]
    return value


# ---- the top-K rule and duplicates ------------------------------------------------------------------------------

def test_ties_go_to_the_lower_slot_like_the_core():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(50, 6)).astype(np.float32)
    pool[40] = pool[7]                  # a duplicate with a higher id
    q = pool[7] + 0.001
    approx = np.linalg.norm(pool.astype(np.float64) - q, axis=1)
    for _ in range(5):
        slots, dist = topk.exact_topk(pool, q, approx + rng.normal(0, 1e-9, size=50), 1)
        assert slots.tolist() == [7]
    both, _ = topk.exact_topk(pool, q, approx, 2)
    assert both.tolist() == [7, 40]


def test_candidate_recall_counts_the_lower_id_duplicate():
    rng = np.random.default_rng(1)
    feats = object.__new__(intervals.Features)
    feats.world, feats.k, feats.c = None, 1, 4
    feats.centers, feats.shell_vec = np.zeros((4, 5), np.float32), np.zeros((0, 5))
    base = rng.normal(size=(30, 5)).astype(np.float32)
    feats.pool = np.vstack([base, base[[0]]])         # id 30 duplicates id 0
    feats.pool_ids = np.arange(len(feats.pool))
    Q = (base[:1] + 0.0005).astype(np.float32)
    has, recall = feats.candidate_recall(np.repeat(Q, 8, axis=0), np.zeros((8, 1), np.int64))
    assert has.all() and (recall == 1.0).all()        # the truth keeps the lower id and so does the answer


def test_e7_index_builds_are_reproducible(world):
    pytest.importorskip("hnswlib")
    first = untimed(e7.run(world, max_queries=100)[0])
    second = untimed(e7.run(world, max_queries=100)[0])
    assert same(first, second)


# ---- audit fixes ---------------------------------------------------------------------------------------------------

def test_e3_certifies_quantized_scores_with_the_true_labels(world, monkeypatch):
    real = e1.lambda_for
    calls = []

    def spy(world_, model, F, y, has):
        calls.append((np.asarray(y).copy(), np.asarray(has).copy()))
        return real(world_, model, F, y, has)

    monkeypatch.setattr(e1, "lambda_for", spy)
    e3.run(world)
    cal, _ = world.role_rows("cal")
    labels = [c for c in calls if len(c[0]) == len(cal.recall)]
    assert len(labels) == 1 + len(e3.BITS)
    assert all(np.array_equal(y, cal.recall) for y, _ in labels)


def test_e11_uses_every_request_whatever_the_unique_query_cap(tiny, monkeypatch):
    first = e11.run(new_world())
    monkeypatch.setattr(e10, "MAX_QUERIES", 40)
    second = e11.run(new_world())
    assert same(first, second)
    n = len(new_world().full_heldout_stream())
    for r in first:
        assert r["n_stretches"] == max(n // r["W_requests"], 1) if r["W_requests"] else True


def test_e12_trains_only_on_queries_after_the_earlier_history(world):
    assert np.array_equal(world.snap["vec"], np.asarray(world.q[world.stream[:world.history_len]], np.float32))
    stale = e12.stale_model(world)
    is_train = np.flatnonzero(world.history()["role"] == world_mod.ROLE_CODES["train"])
    early_len = history_mode.history_length(len(world.stream), world.history_len / len(world.stream) * e12.EARLIER_SHARE)
    assert stale["late"].sum() == (is_train >= early_len).sum() and (is_train[stale["late"]] >= early_len).all()
    rows = e12.run(world)
    assert "hit_rate_fresh_same_rows" in rows[0]


def test_uncertified_hit_rates_are_nan_in_every_experiment(world, monkeypatch):
    monkeypatch.setattr(e1, "lambda_for", lambda *a, **k: policy.Certified(None))
    for row in e1.run(world) + e3.run(world):
        assert np.isnan(row["hit_rate"]) if "hit_rate" in row else np.isnan(row["hit_rate_full"])


# ---- LRU --------------------------------------------------------------------------------------------------------

def lru_case(seed=0, n=500, d=8, n_entries=30, K=5):
    from recal.lru import LruCache
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, d)).astype(np.float32)
    centers = rng.normal(size=(n_entries, d)).astype(np.float32)
    core, ids, lists = build_core(data, centers, K, np.full(n_entries, 2))
    new = LruCache(d, 1e9, 4.0, data.min(axis=0), data.max(axis=0))
    old = lru_old.LruCache(d, 1e9, 4.0, data.min(axis=0), data.max(axis=0))
    for c in centers:
        order = exact_order(data, c)[:4 * K]
        dists = np.linalg.norm(data[order].astype(np.float64) - c.astype(np.float64), axis=1).astype(np.float32)
        for cache in (new, old):
            cache.admit(c, K, order + ID_BASE, dists, data[order])
    return rng, core, new, old, K


def test_lru_lookup_equals_the_core_and_the_old_lookup_except_pooled_shell_vectors():
    rng, core, new, old, K = lru_case()
    queries = rng.normal(size=(60, 8)).astype(np.float32)
    feat, cand_ids, _ = core.candidates_batch(queries, K, 0)
    differs = 0
    for i, q in enumerate(queries):
        res, ref = new.lookup(q, K, None, 0), old.lookup(q, K, None, 0)
        assert set(res.ids.tolist()) == set(cand_ids[i].tolist()) == set(ref.ids.tolist())
        np.testing.assert_allclose(res.features[:2], feat[i, 2:4], rtol=1e-5)
        assert res.features[2] == pytest.approx(feat[i, 4], abs=1e-9)          # the core's intruder rate
        np.testing.assert_allclose(res.features[:2], ref.features[:2], rtol=1e-4)
        differs += int(abs(ref.features[2] - feat[i, 4]) > 1e-9)               # the old lookup counted pooled ids
        assert sorted(res.contrib) == sorted(ref.contrib)
    assert differs > 0


def shell_code_pairs(data: np.ndarray, K: int = 5, n_entries: int = 12):
    """(LRU code, core code) of every shell vector both hold, from the same entries over the same quantizer range."""
    from recal.lru import LruCache
    centers = data[np.random.default_rng(1).choice(len(data), n_entries, replace=False)] + 0.01
    core, ids, lists = build_core(data, centers, K, np.full(n_entries, 2))
    cache = LruCache(data.shape[1], 1e12, 4.0, data.min(axis=0), data.max(axis=0))
    orders = [exact_order(data, c)[:4 * K] for c in centers]
    for c, order in zip(centers, orders):
        dists = np.linalg.norm(data[order].astype(np.float64) - c.astype(np.float64), axis=1).astype(np.float32)
        cache.admit(c, K, order + ID_BASE, dists, data[order])
    packed, d = core.shell_codes(), data.shape[1]
    core_codes = np.zeros((len(packed), d), np.uint8)
    core_codes[:, 0::2] = packed & 15
    core_codes[:, 1::2] = (packed >> 4)[:, :d // 2]
    final_pool = {int(i) + ID_BASE for o in orders for i in o[:2 * K]}
    pairs = []
    for e, order in enumerate(orders):
        shell_ids = [int(i) + ID_BASE for i in order[2 * K:] if int(i) + ID_BASE not in final_pool]
        slots = core.entry(e)["shell_slots"]
        assert len(slots) == len(shell_ids)
        pairs += [(cache.scode[cache.shell[i]], core_codes[slot]) for i, slot in zip(shell_ids, slots)
                  if i in cache.shell]
    return core, cache, pairs


def tie_heavy_int_data(rng, n=300, d=8):
    """Integers 0..30 with both ends in every coordinate: the step is exactly 2, so every odd value is a midpoint."""
    x = rng.integers(0, 31, size=(n, d)).astype(np.float32)
    x[0], x[1] = 0.0, 30.0
    return x


@pytest.mark.parametrize("kind", ["int8_ties", "float"])
def test_lru_shell_codes_equal_the_cores(kind):
    from recal.lru import LruCache
    rng = np.random.default_rng(4)
    data = tie_heavy_int_data(rng) if kind == "int8_ties" else rng.normal(size=(300, 8)).astype(np.float32)
    core, cache, pairs = shell_code_pairs(data)
    lo, step = core.quantizer()
    np.testing.assert_array_equal(cache.lo, lo)
    np.testing.assert_array_equal(cache.step, step)                  # the same float32 step as Core.set_quantizer
    assert len(pairs) > 50 and all(np.array_equal(a, b) for a, b in pairs)
    old = lru_old.LruCache(data.shape[1], 1e12, 4.0, data.min(axis=0), data.max(axis=0))
    differs = (old._quantize(data) != cache._quantize(data)).any()
    assert differs == (kind == "int8_ties")                          # the old rule only disagreed on ties
    assert (LruCache(8, 1e9).step == np.float32(1.0 / 15)).all()     # the core's default range [0, 1]


def test_lru_m_nearest_and_eviction_keep_the_arrays_consistent():
    rng, core, new, old, K = lru_case(seed=3)
    q = rng.normal(size=8).astype(np.float32)
    for m in (3, 7):
        res, ref = new.lookup(q, K, None, m), old.lookup(q, K, None, m)
        assert res.n_read == ref.n_read == m and res.n_candidates == ref.n_candidates
        assert set(res.ids.tolist()) == set(ref.ids.tolist())
    small = type(new)(8, 3000.0, 4.0)
    for i in range(80):
        ids = np.arange(i * 20, i * 20 + 20)
        small.admit(rng.normal(size=8).astype(np.float32), 5, ids, np.sort(rng.random(20)).astype(np.float32),
                    rng.normal(size=(20, 8)).astype(np.float32))
    assert small.total_bytes() <= 3000.0 and small.n_entries() < 80
    live_slots = {s for e in small.entries.values() for s in e["resp"].tolist()}
    assert live_slots == set(small.pool.values()) == set(small.holders)


def test_lru_quantizer_range_comes_from_database_vectors(monkeypatch):
    from recal import lru_build
    seen = {}

    class Spy(lru_build.LruCache):
        def __init__(self, dim, budget, coord_bytes=4.0, lo=None, hi=None):
            seen["lo"], seen["hi"] = np.asarray(lo), np.asarray(hi)
            super().__init__(dim, budget, coord_bytes, lo, hi)

    monkeypatch.setattr(lru_build, "LruCache", Spy)
    import tests.recal.test_ablations as ablations
    system, _ = ablations.run(Path(__import__("tempfile").mkdtemp()), n=3200, lru=True)
    db = system.backend.vectors
    queries = np.asarray(system.log.snapshot()["vec"])
    assert (seen["lo"] >= db.min(axis=0) - 1e-6).all() and (seen["hi"] <= db.max(axis=0) + 1e-6).all()
    # cached database vectors spread wider than the queries, which the old range (the queries') clipped
    assert (seen["lo"] < queries.min(axis=0)).any() and (seen["hi"] > queries.max(axis=0)).any()


# ---- golden comparison ------------------------------------------------------------------------------------------

def test_golden_comparison_passes_on_a_built_cache(world):
    summary = golden.check(world)
    assert summary["ok"] and summary["failures"] == {} and summary["n_queries"] > 0
    assert summary["truth_scanned"] and summary["read_set_recall_vs_exact"] == 1.0


def test_golden_fails_loudly_when_a_python_definition_drifts(world, monkeypatch):
    real = intruder_check.decode_codes
    monkeypatch.setattr(golden.intruder_check, "decode_codes", lambda codes, lo, step: real(codes, lo + 0.2 * step,
                                                                                            step))
    with pytest.raises(golden.GoldenFailure, match="intruder_count"):
        golden.check(world)
    monkeypatch.setattr(golden.intruder_check, "decode_codes", real)
    summary = golden.check(world)
    assert summary["ok"]


def test_golden_fails_when_the_stored_answers_do_not_match_the_corpus(world, monkeypatch):
    top = np.array(world.top)
    top[:, :12] = top[:, :12][:, ::-1]
    monkeypatch.setattr(world, "top", top)
    world.cache.pop("heldout", None)
    try:
        with pytest.raises(golden.GoldenFailure, match="truth_order"):
            golden.check(world)
    finally:
        world.cache.pop("heldout", None)


def test_preflight_stops_when_the_probe_golden_fails(tiny, monkeypatch):
    monkeypatch.setattr(preflight, "PROBE_REQUESTS", 2500)

    def fail(*a, **k):
        raise golden.GoldenFailure("golden comparison failed: injected")

    monkeypatch.setattr(golden, "check", fail)
    errors = preflight.probe_build(Path("results/online/tiny-a"), "trace", 0.6, 4, PARAMS)
    assert errors and "injected" in errors[0]


# ---- provenance and hashes ---------------------------------------------------------------------------------------

FAKE_FILES = {
    "recal/__init__.py": "",
    "recal/micro/__init__.py": "",
    "recal/build.py": "# build\n",
    "recal/micro/world.py": "from recal import build\n",
    "recal/micro/e1.py": "from recal.micro import world\n",
    "recal/micro/e13.py": "from recal.micro import e1\n",
    "recal/micro/e14.py": "from recal.micro import e1\nimport recal.micro.world\n",
    "recal/core/recal_core.cpp": "// core\n",
    "recal/unrelated.py": "x = 1\n",
    "recal/micro/provenance.py": "# registry\n",
    "scripts/micro/run.py": "# runner\n",
}


def fake_root(tmp_path):
    root = tmp_path / "repo"
    for rel, text in FAKE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "recal/core/_recal_core.fake.so").write_bytes(b"binary")
    return root


def hashes(root, names=("E1", "E13", "E14")):
    return {e: provenance.code_hash(e, root) for e in names}


def test_code_hash_follows_the_import_closure_of_an_experiment(tmp_path):
    root = fake_root(tmp_path)
    before = hashes(root)
    assert provenance.source_files("E1", root) == [
        "recal/__init__.py", "recal/build.py", "recal/core/recal_core.cpp", "recal/micro/__init__.py",
        "recal/micro/e1.py", "recal/micro/world.py"]
    (root / "recal/unrelated.py").write_text("x = 2\n")
    assert hashes(root) == before
    (root / "recal/micro/e13.py").write_text("from recal.micro import e1\n# changed\n")
    after = hashes(root)
    assert after["E1"] == before["E1"] and after["E14"] == before["E14"] and after["E13"] != before["E13"]
    (root / "recal/build.py").write_text("# changed\n")
    assert all(hashes(root)[e] != after[e] for e in after)
    now = hashes(root)
    (root / "recal/core/_recal_core.fake.so").write_bytes(b"rebuilt")
    assert all(hashes(root)[e] != now[e] for e in now)


def test_adding_an_experiment_or_editing_the_runner_changes_no_other_hash(tmp_path):
    root = fake_root(tmp_path)
    before = hashes(root)
    (root / "recal/micro/e15.py").write_text("from recal.micro import e1\n")
    (root / "recal/micro/provenance.py").write_text("# registry with E15\n")
    (root / "scripts/micro/run.py").write_text("# runner with E15\n")
    assert hashes(root) == before
    assert provenance.code_hash("E15", root) not in before.values()
    for rel in ("recal/micro/provenance.py", "scripts/micro/run.py"):
        assert all(rel not in provenance.source_files(e, root) for e in before)


def test_provenance_version_is_in_every_hash(tmp_path, monkeypatch):
    root = fake_root(tmp_path)
    before = hashes(root)
    monkeypatch.setattr(provenance, "PROVENANCE_VERSION", provenance.PROVENANCE_VERSION + 1)
    assert all(hashes(root)[e] != before[e] for e in before)


def test_no_real_experiment_imports_the_runner_or_the_registry():
    for e in ("E1", "E3", "E4", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "intruder_check", "golden"):
        files = provenance.source_files(e)
        assert "recal/micro/provenance.py" not in files and "scripts/micro/run.py" not in files
        assert f"recal/micro/{e.lower()}.py" in files and "recal/core/recal_core.cpp" in files


def test_unit_keys_depend_on_the_code_hash(tmp_path):
    settings = {"experiment": "E1", "code_hash": "aaaa"}
    units = units_mod.Units(tmp_path / "E1", "E1", settings)
    units.cached("b0.6", lambda: {"rows": [1]})
    assert units_mod.Units(tmp_path / "E1", "E1", dict(settings)).done("b0.6")
    assert not units_mod.Units(tmp_path / "E1", "E1", dict(settings, code_hash="bbbb")).done("b0.6")
    assert not units_mod.Units(tmp_path / "E1", "E1", dict(settings, prep_hash="x")).done("b0.6")


def test_settings_carry_hashes_and_not_the_commit_or_workers(tiny):
    cfg = json.loads((tiny / "matrix.json").read_text())
    w = cfg["workloads"]["tiny-a"]
    s = micro_run.settings_of("E1", "tiny-a", w, [0.4, 0.6], PARAMS, 0, 5000)
    assert {"code_hash", "prep_hash", "libraries"} <= set(s) and "code_commit" not in s and "workers" not in s
    assert s["code_hash"] == provenance.code_hash("E1") and len(s["prep_hash"]) == 16
    assert micro_run.settings_of("E1", "tiny-a", w, [0.4, 0.6], PARAMS, 0, 5000, "quick")["tier"] == "quick"


# ---- one worker equals many -------------------------------------------------------------------------------------

def test_workers_do_not_change_any_experiment(tiny, monkeypatch):
    pytest.importorskip("hnswlib")
    monkeypatch.setattr(world_mod, "MIN_CHUNK", 16)
    out = {}
    for workers in (1, 3):
        w = new_world(workers)
        out[workers] = untimed({
            "intruder_check": intruder_check.run(w), "E1": e1.run(w), "E3": e3.run(w), "E4": e4.run(w),
            "E8": e8.run(w), "E10": e10.run(w), "E11": e11.run(w), "E12": e12.run(w), "E13": e13.run(w)})
    assert same(out[1], out[3])


def test_splitting_workers_over_budgets_and_queries():
    assert micro_run.split_workers(1, 3, None, None) == (1, 1)
    assert micro_run.split_workers(12, 3, None, None) == (3, 4)
    assert micro_run.split_workers(32, 1, None, None) == (1, 32)
    assert micro_run.split_workers(8, 3, 2, 5) == (2, 5)


def test_runner_results_match_for_one_and_many_workers(tiny):
    for workers in (1, 3):
        run_cli(tiny, "--experiments", "E1,E4,E10", "--no-upload", "--workers", str(workers), "--force",
                out=f"w{workers}")
    for e in ("E1", "E4", "E10"):
        rows = [json.loads((tiny / f"w{n}" / e / "tiny-a" / f"{e.lower()}.json").read_text()) for n in (1, 3)]
        assert close(untimed(rows[0]["results"]), untimed(rows[1]["results"]))
        assert rows[1]["runtime"]["workers"] == 3 and rows[0]["runtime"]["workers"] == 1
        assert rows[0]["runtime"]["libraries"]["numpy"] and rows[0]["config_hash"] == rows[1]["config_hash"]


# ---- quick tier and identical caches ----------------------------------------------------------------------------

def test_quick_tier_runs_the_middle_budget_on_a_sample_and_writes_its_own_folder(tiny):
    store = tiny / "store_quick"
    assert run_cli(tiny, "--tier", "quick", "--experiments", "intruder_check,E1,E3,E4,E10,E11,E13,E15", "--store",
                   str(store), out="quick") == 0
    for e, name in (("E1", "e1"), ("E3", "e3"), ("E4", "e4"), ("E10", "e10"), ("E11", "e11"), ("E13", "e13"),
                    ("E15", "e15"), ("intruder_check", "intruder_check")):
        body = json.loads((tiny / "quick" / e / "tiny-a-quick" / f"{name}.json").read_text())
        assert {r["budget"] for r in body["results"]["rows"]} == {0.6} and body["settings"]["tier"] == "quick"
        assert all(r["tier"] == "quick" for r in body["results"]["rows"])
        assert (store / "micro" / e / "tiny-a-quick" / f"{name}.json").exists()
    e1_rows = json.loads((tiny / "quick" / "E1" / "tiny-a-quick" / "e1.json").read_text())["results"]["rows"]
    assert e1_rows[0]["n_requests"] == micro_run.QUICK_SAMPLE_REQUESTS
    e13_rows = json.loads((tiny / "quick" / "E13" / "tiny-a-quick" / "e13.json").read_text())["results"]["rows"]
    assert {(r["c"], r["degree"], r["ef"]) for r in e13_rows if r["c"] != "m"} == {(32, 16, 64), (32, 16, 128)}
    e11_rows = json.loads((tiny / "quick" / "E11" / "tiny-a-quick" / "e11.json").read_text())["results"]["rows"]
    assert {(r["window_share_of_M"], r["bits"], r["delta"]) for r in e11_rows} == {
        (s, b, 0.05) for s in (0.01, 0.03) for b in (8, 16)}
    assert (tiny / "quick" / "golden" / "tiny-a-quick" / "units").exists()
    assert not (tiny / "quick" / "E1" / "tiny-a").exists()


def test_quick_tier_refuses_experiments_it_does_not_cover(tiny):
    with pytest.raises(SystemExit, match="E8 is not part of the quick tier"):
        run_cli(tiny, "--tier", "quick", "--experiments", "E1,E8", "--no-upload", out="refused")
    assert micro_run.experiment_budgets("E1", [0.01, 0.05, 0.1], "quick") == [0.05]
    assert micro_run.experiment_budgets("E11", [0.01, 0.05, 0.1]) == [0.01, 0.05]


def test_identical_caches_reuse_the_smaller_budgets_result(tiny, monkeypatch, capsys):
    monkeypatch.setattr(micro_run, "cache_signature", lambda world: "same")
    run_cli(tiny, "--experiments", "E1,E8", "--no-upload", "--force", out="same")
    out = capsys.readouterr().out
    assert "E1 at budget 0.6: same cache as budget 0.4, reusing its result" in out
    assert "E8 at budget 0.6: same cache" not in out
    rows = json.loads((tiny / "same" / "E1" / "tiny-a" / "e1.json").read_text())["results"]["rows"]
    by_budget = {b: [r for r in rows if r["budget"] == b] for b in (0.4, 0.6)}
    assert len(by_budget[0.4]) == len(by_budget[0.6]) == 2
    for a, b in zip(by_budget[0.4], by_budget[0.6]):
        assert a["hit_rate"] == b["hit_rate"] and a["lam"] == b["lam"]
    assert micro_run.rebudget({"budget": 0.4, "x": [{"budget": 0.4}, {"budget": 0.1}]}, 0.4, 0.6) == {
        "budget": 0.6, "x": [{"budget": 0.6}, {"budget": 0.1}]}


# ---- inputs for a fresh instance ---------------------------------------------------------------------------------

def test_inputs_are_pushed_and_pulled_with_hashes(tiny, tmp_path):
    store = tmp_path / "store_inputs"
    cfg = tiny / "matrix.json"
    assert micro_inputs.main(["push", "--dataset", "tiny", "--config", str(cfg), "--store", str(store)]) == 0
    key = store / "inputs" / "micro" / "tiny" / "tiny-a"
    assert (key / "streams.npz").exists() and (key / "manifest.json").exists()
    prep = tiny / "results" / "online" / "tiny-a"
    moved = tmp_path / "moved"
    shutil.move(str(prep), str(moved))
    try:
        assert micro_inputs.main(["pull", "--dataset", "tiny", "--config", str(cfg), "--store", str(store)]) == 0
        assert provenance.prep_hash(prep) == provenance.prep_hash(moved)
        (prep / "top500.npy").write_bytes(b"corrupt")
        with pytest.raises(SystemExit, match="top500.npy"):
            micro_inputs.main(["verify", "--dataset", "tiny", "--config", str(cfg), "--store", str(store)])
    finally:
        shutil.rmtree(prep, ignore_errors=True)
        shutil.move(str(moved), str(prep))


def test_time_features_prints_what_it_timed_and_stops(tiny, capsys):
    assert run_cli(tiny, "--time-features", "--experiments", "E1,E4", "--no-upload") == 0
    out = capsys.readouterr().out
    assert "[time]" in out and "ms per query" in out and "Only the build and the core's features were timed" in out
    with pytest.raises(SystemExit):
        run_cli(tiny, "--plan", "--experiments", "E1", "--no-upload")


def test_summarize_reads_both_the_full_and_the_quick_folder(tiny, tmp_path):
    store = tmp_path / "store_both"
    run_cli(tiny, "--experiments", "E1", "--store", str(store), "--force", out="both_full")
    run_cli(tiny, "--tier", "quick", "--experiments", "E1", "--store", str(store), "--force", out="both_quick")
    rows = summarize.read_rows(store / "micro", "E1")
    assert {(r["folder"], r["tier"]) for r in rows} == {("tiny-a", "full"), ("tiny-a-quick", "quick")}
    assert len([r for r in rows if r["tier"] == "full"]) == 4 and len([r for r in rows if r["tier"] == "quick"]) == 2
    assert summarize.main(["--store", str(store), "--pulled", str(tmp_path / "pb"), "--out", str(tmp_path / "tb")]) == 0
    header = (tmp_path / "tb" / "E1.csv").read_text().splitlines()[0].split(",")
    assert "tier" in header and "folder" in header


def test_summarize_reads_quick_folders_and_golden_units(tiny, tmp_path):
    store = tmp_path / "store_s"
    run_cli(tiny, "--tier", "quick", "--experiments", "E1", "--store", str(store), out="qs")
    assert summarize.main(["--store", str(store), "--pulled", str(tmp_path / "p"), "--out", str(tmp_path / "t")]) == 0
    assert "tiny-a" in (tmp_path / "t" / "E1.csv").read_text()
