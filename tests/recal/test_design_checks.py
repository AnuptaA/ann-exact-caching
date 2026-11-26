import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from recal import policy, prepare
from recal.micro import (common, e1, e3, e4, e7, e10, e11, e12, e13, fit, intervals, intruder_check, preflight,
                         units as units_mod, world as world_mod)
from scripts.micro import run as micro_run, summarize

PARAMS = {"H": 900, "cold_start": 500, "min_ess": 30.0}
BUDGET = 0.6


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("design")
    cfg = prepare.tiny_config(("tiny-a",))
    cfg["workloads"]["tiny-a"]["budgets"] = [BUDGET]
    prepare.make_tiny(cfg, root)
    (root / "matrix.json").write_text(json.dumps(cfg))
    mp = pytest.MonkeyPatch()
    mp.chdir(root)
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def world(tiny):
    return world_mod.build_world("results/online/tiny-a", "trace", BUDGET, 4, name="tiny-a", params=PARAMS)


def run_cli(tiny, *extra, out="out"):
    return micro_run.main(["--dataset", "tiny", "--config", str(tiny / "matrix.json"), "--out", str(tiny / out),
                           "--params", json.dumps(PARAMS), *extra])


# ---- part 1: the intruder check ---------------------------------------------------------------------------------

def test_intruder_check_reproduces_the_core_and_explains_the_definitions(world):
    rows, extra = intruder_check.run(world)
    pairs = {r["pair"]: r for r in rows}
    assert pairs["a_vs_c"]["count_agree_share"] == 1.0          # the Python rebuild equals the core
    assert pairs["b_vs_d_fixed"]["count_agree_share"] == 1.0    # E4 without cached vectors equals full precision
    assert pairs["b_vs_c"]["count_agree_share"] < 1.0           # 4-bit codes change counts
    assert extra["mean_count"]["d_asis"] > extra["mean_count"]["d_fixed"]
    assert extra["vectors_within_5pct_of_nr_mean"] >= extra["vectors_within_1pct_of_nr_mean"]


def test_e4_shells_never_hold_cached_vectors(world):
    table = e4.EntryTable(world)
    pool = set(table.pool_ids.tolist())
    assert not any(pool & set(shell.tolist()) for shell in table.shell_ids)
    assert any(set(shell.tolist()) & pool for shell in e4.EntryTable(world, exclude_pool=False).shell_ids)
    assert len(intruder_check.shell_slot_ids(world, table)) == world.core.n_shell_slots()


# ---- part 2: intervals ---------------------------------------------------------------------------------------------

def synthetic_features(rng, c):
    d, n_entries, n_pool, n_slots = 8, 40, 200, 150
    feats = object.__new__(intervals.Features)
    feats.world = None
    feats.k, feats.c = 5, c
    feats.centers = rng.normal(size=(n_entries, d)).astype(np.float32)
    feats.radius = rng.uniform(1.0, 3.0, size=n_entries)
    feats.pool = rng.normal(size=(n_pool, d)).astype(np.float32)
    feats.shell_vec = rng.normal(size=(n_slots, d))
    feats.slots = [rng.choice(n_slots, size=12, replace=False) for _ in range(n_entries)]
    feats.pool_ids = np.arange(n_pool)
    return feats


@pytest.mark.parametrize("c", [4, 12, 100])
@pytest.mark.parametrize("bits", [4, 8, 16])
def test_full_precision_features_and_scores_lie_in_their_intervals(c, bits):
    rng = np.random.default_rng(c * 100 + bits)
    feats = synthetic_features(rng, c)
    Q = rng.normal(size=(120, 8)).astype(np.float32)
    exact = feats.exact(Q)
    qhat, err = intervals.quantize_with_error(Q, bits)
    assert (err >= np.linalg.norm(Q.astype(np.float64) - qhat, axis=1) - 1e-12).all()
    assert (err.astype(np.float16).astype(np.float64) == err).all()      # stored in two bytes
    lower, upper = feats.bounds(qhat, err)
    assert ((exact >= lower - 1e-9) & (exact <= upper + 1e-9)).all()
    model = fit.fit_linear(exact, rng.uniform(size=len(exact)))
    low, high = intervals.score_interval(model, lower, upper)
    score = fit.score(model, exact)
    assert ((score >= low - 1e-9) & (score <= high + 1e-9)).all()


def test_worst_case_set_matches_brute_force():
    rng = np.random.default_rng(3)
    for trial in range(60):
        n = int(rng.integers(3, 10))
        y = rng.uniform(0.3, 1.0, size=n)
        w = rng.uniform(0.5, 3.0, size=n)
        kind = rng.integers(0, 3, size=n)         # 0 certain, 1 ambiguous, 2 outside
        certain, ambiguous = kind == 0, kind == 1
        got = intervals.worst_case_set(y, w, certain, ambiguous)
        assert (got[certain]).all() and not got[kind == 2].any()
        free = np.flatnonzero(ambiguous)
        best = np.inf
        for r in range(len(free) + 1):
            for chosen in itertools.combinations(free, r):
                mask = certain.copy()
                mask[list(chosen)] = True
                if mask.any():
                    best = min(best, (w[mask] * y[mask]).sum() / w[mask].sum())
        if got.any():
            assert (w[got] * y[got]).sum() / w[got].sum() == pytest.approx(best, abs=1e-12) or not certain.any()


def test_degenerate_intervals_certify_like_the_plain_procedure():
    rng = np.random.default_rng(5)
    score = rng.uniform(size=600)
    y = np.clip(0.55 + 0.5 * score + rng.normal(0, 0.1, size=600), 0, 1)
    edges = policy.threshold_edges(score)
    plain = fit.certify(score, score, y, 0.9, 0.05, 30.0)
    exact = intervals.certify_worst_case(edges, score, score, y, 0.9, 0.05, 30.0)
    assert exact["lam"] == plain.lam and plain.lam is not None
    wide = intervals.certify_worst_case(edges, score - 0.2, score + 0.2, y, 0.9, 0.05, 30.0)
    assert wide["lam"] is None or wide["lam"] >= plain.lam


def test_e10_intervals_hold_and_precision_costs_hit_rate(world):
    rows = e10.run(world)
    assert [r["bits"] for r in rows] == [4, 8, 16]
    assert all(r["violations"] == 0 for r in rows)
    by = {r["bits"]: r for r in rows}
    assert by[16]["error_mean"] < by[8]["error_mean"] < by[4]["error_mean"]
    for r in rows:
        if r["certified"]:
            assert r["hit_rate_wc"] <= r["hit_rate_full"] + 0.05 and 0 <= r["ambiguous_share"] <= 1
            assert len(r["ambiguous_share_curve"]) == len(e10.CURVE_LEVELS)
        else:
            assert np.isnan(r["hit_rate_wc"])
    if by[4]["certified"] and by[16]["certified"]:
        assert by[16]["ambiguous_share"] <= by[4]["ambiguous_share"] + 1e-12


def test_e11_memory_arithmetic_rows_and_summary(world):
    assert e11.item_bytes(512, 10, 8) == (71, 512 + 40 + 33)
    assert e11.items_per_half(0.01, 1e6, 512, 10, 8) == int(0.01 * 1e6 // (71 + 585))
    assert e11.pick_budgets([0.1, 0.01, 0.05]) == [0.01, 0.05] and e11.pick_budgets([0.5]) == [0.5]
    rows = e11.run(world)
    assert len(rows) == len(e11.PRECISIONS) * len(e11.SHARES) * len(e11.DELTAS)
    first = rows[0]
    for key in ("certified_share", "hit_rate", "recall_of_hits", "sampling_slack", "quantization_slack", "in_sample"):
        assert key in first
    wide = [r for r in rows if r["window_share_of_M"] == max(e11.SHARES) and r["certified_share"] > 0]
    assert all(r["quantization_slack"] >= -1e-9 for r in wide if r["quantization_slack"] == r["quantization_slack"])
    out = e11.summary(rows)
    assert out and set(out[0]) == {"workload", "budget", "smallest_share_certifying", "better_bits"}


def test_e12_stale_features_give_a_correlated_model(world):
    rows = e12.run(world)
    assert len(rows) == 1
    r = rows[0]
    assert -1 <= r["spearman"] <= 1 and r["n_entries_earlier"] <= r["n_entries"] and r["n_train"] > 10
    assert len(r["coef_fresh"]) == len(r["coef_stale"]) == 3


# ---- part 3: index accuracy ---------------------------------------------------------------------------------------

def test_e13_exact_search_over_few_centers_is_reproduced(world):
    pytest.importorskip("hnswlib")
    rows, summary = e13.run(world)
    index_rows = [r for r in rows if r["c"] != "m"]
    assert len(index_rows) == len(e13.CS) * len(e13.DEGREES) * len(e13.EFS)
    assert [r["c"] for r in rows].count("m") == 1
    top = [r for r in index_rows if r["ef"] == max(e13.EFS)]
    assert all(r["set_recall"] == 1.0 and r["best_fit_in_set_share"] == 1.0 and r["features_identical_share"] == 1.0
               for r in top)
    assert set(summary["smallest_for_99"]) == {"c8", "c16", "c32"} and summary["core_m_path"]["m"] == world.m


def test_e7_extended_columns(world):
    pytest.importorskip("hnswlib")
    rows, summary = e7.run(world, max_queries=100)
    assert e7.EFS[-1] == 1024 and len(rows) == len(e7.DEGREES) * len(e7.EFS)
    for r in rows:
        assert 0 <= r["rerank_exact_share"] <= 1 and r["rerank_us_per_query"] > 0
        assert (r["fast_unchanged_decision_share"] == r["fast_unchanged_decision_share"]) == (r["ef"] in e7.FAST_EFS)
    wide = [r for r in rows if r["ef"] == 1024]
    assert all(r["exact_share"] >= 0.9 and r["fast_unchanged_decision_share"] != 2 for r in wide)
    assert "smallest_rerank_exact_99" in summary and "smallest_fast_unchanged_decisions_99" in summary


def test_e3_writes_nan_for_an_uncertified_quantized_lambda(world, monkeypatch):
    real = e1.lambda_for
    calls = []

    def flaky(world_, model, F, y, has):
        calls.append(1)
        cert = real(world_, model, F, y, has)
        return cert if len(calls) == 1 else policy.Certified(None)

    monkeypatch.setattr(e1, "lambda_for", flaky)
    rows = e3.run(world)
    assert all(np.isnan(r["hit_rate_quantized_lambda"]) and r["lam_quantized"] is None for r in rows)
    assert all(r["hit_rate_full"] == r["hit_rate_full"] for r in rows)


# ---- part 0: checkpoints, progress, preflight ---------------------------------------------------------------------

def test_units_skip_finished_work_and_survive_an_interruption(tmp_path):
    units = units_mod.Units(tmp_path / "E", "E", {"experiment": "E"})
    calls = []

    def compute(key):
        calls.append(key)
        if key == "b" and calls.count("b") == 1:
            raise RuntimeError("stopped")
        return {"key": key}

    with pytest.raises(RuntimeError):
        for key in ("a", "b", "c"):
            units.cached(key, lambda key=key: compute(key))
    assert calls == ["a", "b"]
    out = [units.cached(key, lambda key=key: compute(key)) for key in ("a", "b", "c")]
    assert calls == ["a", "b", "b", "c"] and out == [{"key": "a"}, {"key": "b"}, {"key": "c"}]
    changed = units_mod.Units(tmp_path / "E", "E", {"experiment": "E", "code_commit": "new"})
    assert not changed.done("a")


def test_runner_resumes_after_an_interruption_and_writes_progress(tiny, monkeypatch, capsys):
    monkeypatch.setattr(micro_run, "budgets_for", lambda cfg, name, w: [BUDGET, 0.4])
    real = e1.run

    def stopping(world):
        if world.budget == 0.4:
            raise RuntimeError("stopped here")
        return real(world)

    monkeypatch.setattr(e1, "run", stopping)
    with pytest.raises(RuntimeError, match="stopped here"):
        run_cli(tiny, "--experiments", "E1", "--no-upload", out="resume")
    directory = tiny / "resume" / "E1" / "tiny-a"
    assert (directory / "units" / "b0.6.json").exists() and not (directory / "units" / "b0.4.json").exists()
    progress = json.loads((directory / "progress.json").read_text())
    assert "stopped here" in progress["last_error"] and progress["units_done"] == 1 and progress["units_total"] == 2
    for key in ("queries_per_second", "peak_memory_mb", "eta_s", "updated"):
        assert key in progress
    monkeypatch.setattr(e1, "run", real)
    capsys.readouterr()
    run_cli(tiny, "--experiments", "E1", "--no-upload", out="resume")
    out = capsys.readouterr().out
    assert "building the cache at budget 0.4" in out and "building the cache at budget 0.6" not in out
    rows = json.loads((directory / "e1.json").read_text())["results"]["rows"]
    assert sorted({r["budget"] for r in rows}) == [0.4, BUDGET]
    done = json.loads((directory / "progress.json").read_text())
    assert done["finished"] and done["last_error"] is None and done["units_done"] == 1


def test_units_are_pushed_as_they_finish_and_summaries_read_partial_results(tiny):
    store = tiny / "store_units"
    run_cli(tiny, "--experiments", "E1", "--store", str(store), "--force", out="pushed")
    pushed = store / "micro" / "E1" / "tiny-a"
    assert (pushed / "units" / f"b{BUDGET:g}.json").exists() and (pushed / "progress.json").exists()
    (pushed / "e1.json").unlink()
    pulled = tiny / "pulled_partial"
    (pulled / "E1").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(pushed, pulled / "E1" / "tiny-a", dirs_exist_ok=True)
    assert len(summarize.read_rows(pulled, "E1")) == 2


def test_preflight_names_the_file_the_expected_and_the_actual_value(tiny, capsys):
    with pytest.raises(SystemExit, match=r"streams.npz.*expected at least \d+ requests, found 500"):
        run_cli(tiny, "--experiments", "E1", "--no-upload", "--max-requests", "500", out="short")
    prep = tiny / "results" / "online" / "tiny-a"
    moved = prep / "top500.npy.moved"
    (prep / "top500.npy").rename(moved)
    try:
        with pytest.raises(SystemExit, match=r"top500.npy: file is missing"):
            run_cli(tiny, "--experiments", "E1", "--no-upload", out="missing")
    finally:
        moved.rename(prep / "top500.npy")
    errors, info = preflight.check_prep(prep, "trace", 0)
    assert errors == [] and info["n_requests"] == 3000 and info["history"] == 900
    assert preflight.check_prep(prep, "nope", 0)[0][0].endswith("only ['trace']") or "has no stream" in \
        preflight.check_prep(prep, "nope", 0)[0][0]


def test_experiment_names_accept_the_new_checks_in_any_case():
    assert micro_run.parse_experiments("e13,Intruder_Check,E10") == ["intruder_check", "E10", "E13"]
    assert micro_run.ORDER[0] == "intruder_check"
    assert micro_run.experiment_budgets("E11", [0.01, 0.05, 0.1]) == [0.01, 0.05]


def test_the_new_experiments_run_through_the_runner_and_are_summarized(tiny):
    pytest.importorskip("hnswlib")
    store = tiny / "store_new"
    assert run_cli(tiny, "--experiments", "intruder_check,E10,E11,E12,E13", "--store", str(store), "--force",
                   out="new") == 0
    for experiment, name in (("intruder_check", "intruder_check"), ("E10", "e10"), ("E11", "e11"), ("E12", "e12"),
                             ("E13", "e13")):
        body = json.loads((store / "micro" / experiment / "tiny-a" / f"{name}.json").read_text())
        assert body["results"]["rows"] and body["config_hash"]
    assert summarize.main(["--store", str(store), "--pulled", str(tiny / "pulled_new"), "--out",
                           str(tiny / "tables_new")]) == 0
    assert (tiny / "tables_new" / "E10.md").exists() and (tiny / "tables_new" / "intruder_check.csv").exists()
    assert common.read_unit(tiny / "new" / "E13" / "tiny-a", "b0.6/degree8",
                            json.loads((tiny / "new" / "E13" / "tiny-a" / "e13.json").read_text())["settings"])
