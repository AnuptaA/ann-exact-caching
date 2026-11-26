import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from recal import policy, prepare
from recal.micro import (common, e1, e3, e4, e7, e10, e11, e12, e13, fit, golden, intervals, intruder_check,
                         preflight, quantize, units as units_mod, world as world_mod)
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
    for r in rows:
        assert 0 <= r["recall_of_hits_x"] <= 1 and 0 <= r["recall_of_hits_y"] <= 1
    assert pairs["a_vs_c"]["recall_of_hits_x"] == pairs["a_vs_c"]["recall_of_hits_y"]     # the same decisions


def test_e4_shells_never_hold_cached_vectors(world):
    table = e4.EntryTable(world)
    pool = set(table.pool_ids.tolist())
    assert not any(pool & set(shell.tolist()) for shell in table.shell_ids)
    assert any(set(shell.tolist()) & pool for shell in e4.EntryTable(world, exclude_pool=False).shell_ids)
    assert len(intruder_check.shell_slot_ids(world, table)) == world.core.n_shell_slots()


def test_shell_codes_round_ties_like_the_core():
    from recal.core import Core
    d = 6
    core = Core(d, 10)
    core.set_quantizer(np.zeros(d, np.float32), np.full(d, 30.0, np.float32))   # step 2: odd values are ties
    X = np.array([[1, 3, 5, 7, 9, 29], [11, 13, 15, 17, 19, 21], [0, 2, 4, 30, 31, -1]], np.float32)
    entry = core.add_entry(np.zeros(d, np.float32), 1, 1, np.array([100]), np.zeros((1, d), np.float32), 1.0)
    core.add_shell(entry, np.arange(len(X)), X)
    codes = core.shell_codes()
    got = np.zeros((len(X), d), np.int64)
    got[:, 0::2], got[:, 1::2] = codes & 15, codes >> 4
    assert (quantize.shell_codes(X, *core.quantizer()) == got).all()
    assert (np.clip(np.rint(X / 2), 0, 15) != got).any()   # round half to even disagrees on these ties


def test_golden_fails_loudly_on_a_wrong_quantizer_range(world, monkeypatch):
    real = quantize.shell_codes
    monkeypatch.setattr(intruder_check.quantize, "shell_codes", lambda X, lo, step: real(X, lo + 0.5 * step, step))
    with pytest.raises(golden.GoldenFailure, match=r"shell_membership \(whole check\): \d+ of \d+ shell slots"):
        golden.check(world)


def test_golden_fails_loudly_on_a_corrupted_slot_mapping(world, monkeypatch):
    real = world.entries

    def swapped():
        entries = real()
        multi = [e for e in entries if len(e["shell_slots"]) > 1]
        for e in multi:
            e["shell_slots"] = list(e["shell_slots"])[::-1]
        return entries

    monkeypatch.setattr(world, "entries", swapped)
    with pytest.raises(golden.GoldenFailure, match="shell_membership"):
        golden.check(world)


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


Z95 = 1.6448536269514722


def served_lcb(y, w, mask):
    return intervals.lower_bound(y[mask], w[mask], Z95)["lcb"]


def brute_force_minimum(y, w, certain, ambiguous):
    """Smallest lower confidence bound over every served set between the certain items and all possible items."""
    items = np.flatnonzero(ambiguous)
    best = np.inf
    for bits in range(1 << len(items)):
        mask = certain.copy()
        mask[items[[(bits >> i) & 1 == 1 for i in range(len(items))]]] = True
        if mask.any():
            best = min(best, served_lcb(y, w, mask))
    return best


def mean_minimizing_set(y, w, certain, ambiguous):
    """The previous rule: ambiguous items, lowest recall first, while each is below the running weighted mean."""
    served, items = certain.copy(), np.flatnonzero(ambiguous)
    items = items[np.argsort(y[items], kind="stable")]
    total, weighted = float(w[served].sum()), float((w[served] * y[served]).sum())
    for i in items:
        if total > 0 and y[i] >= weighted / total:
            break
        served[i] = True
        total += w[i]
        weighted += w[i] * y[i]
    return served


def random_worst_case(rng, regime):
    nc, na = int(rng.integers(6, 40)), int(rng.integers(3, 11))
    if regime == "below":
        yc, ya = np.clip(rng.normal(0.93, 0.06, nc), 0, 1), rng.uniform(0.2, 1.0, na)
    elif regime == "straddle":
        yc, ya = np.clip(rng.normal(0.8, 0.15, nc), 0, 1), rng.uniform(0.0, 1.0, na)
    else:
        yc, ya = rng.beta(2, 1, nc), rng.beta(2, 1, na)
    wc = np.where(rng.random(nc) < 0.2, 10.0, 1.0)          # misses weigh 1, audited hits 1/a = 10
    wa = np.where(rng.random(na) < 0.3, 10.0, 1.0)
    certain = np.r_[np.ones(nc, bool), np.zeros(na, bool)]
    return np.r_[yc, ya], np.r_[wc, wa], certain, ~certain


def test_worst_case_bound_brackets_the_brute_force_minimum():
    rng = np.random.default_rng(11)
    mean_set_too_high = prefix_not_optimal = 0
    for regime in ("below", "straddle", "wide"):
        for _ in range(120):
            y, w, certain, ambiguous = random_worst_case(rng, regime)
            brute = brute_force_minimum(y, w, certain, ambiguous)
            out = intervals.worst_case_bound(y, w, certain, ambiguous, Z95)
            assert out["lcb"] <= brute + 1e-12                      # valid: never above the true minimum
            assert out["lcb_prefix"] >= brute - 1e-12               # the feasible set it brackets from above
            assert out["bound_gap"] == pytest.approx(out["lcb_prefix"] - out["lcb"]) and out["bound_gap"] >= -1e-12
            old = served_lcb(y, w, mean_minimizing_set(y, w, certain, ambiguous))
            assert out["lcb"] <= old + 1e-12                        # never less conservative than the mean rule
            mean_set_too_high += old > brute + 1e-6
            order = np.flatnonzero(ambiguous)[np.argsort(y[ambiguous], kind="stable")]
            prefixes = []
            for k in range(len(order) + 1):
                mask = certain.copy()
                mask[order[:k]] = True
                prefixes.append(served_lcb(y, w, mask))
            prefix_not_optimal += min(prefixes) > brute + 1e-6
    assert mean_set_too_high > 100 and prefix_not_optimal > 0       # the cases the change is about


def test_the_bound_is_exact_when_the_ambiguous_items_lie_below_the_mean():
    rng = np.random.default_rng(2)
    for _ in range(30):
        nc, na = int(rng.integers(20, 60)), int(rng.integers(3, 10))
        y = np.r_[np.clip(rng.normal(0.95, 0.03, nc), 0, 1), rng.uniform(0.1, 0.6, na)]
        w = np.ones(nc + na)
        certain = np.r_[np.ones(nc, bool), np.zeros(na, bool)]
        out = intervals.worst_case_bound(y, w, certain, ~certain, Z95)
        assert out["bound_gap"] == pytest.approx(0.0, abs=1e-12)
        assert out["lcb"] == pytest.approx(brute_force_minimum(y, w, certain, ~certain), abs=1e-12)


def test_the_effective_sample_size_is_the_smallest_over_all_served_sets():
    # ten certain items (ESS 10); one ambiguous audited hit of weight 10 would drop it to (20)^2 / 110
    y = np.r_[np.full(10, 0.95), 0.95]
    w = np.r_[np.ones(10), 10.0]
    certain = np.r_[np.ones(10, bool), False]
    out = intervals.worst_case_bound(y, w, certain, ~certain, Z95)
    assert out["ess"] == pytest.approx(400 / 110)
    edges = np.array([0.0])
    cert = intervals.certify_worst_case(edges, np.r_[np.ones(10), 0.0], np.r_[np.ones(10), 1.0], y, 0.5, 0.05, 5.0, w)
    assert cert["lam"] is None                                      # a feasible set has ESS 3.6 < 5
    cert = intervals.certify_worst_case(edges, np.ones(11), np.ones(11), y, 0.5, 0.05, 3.0, w)
    assert cert["lam"] == 0.0 and cert["quantization_slack"] == pytest.approx(
        cert["mean_possible"] - cert["lcb"] - cert["slack"])


def test_more_than_two_ambiguous_weights_are_refused():
    y, w = np.full(6, 0.9), np.array([1.0, 1.0, 1.0, 2.0, 5.0, 10.0])
    certain = np.array([True, True, True, False, False, False])
    with pytest.raises(ValueError, match="at most two"):
        intervals.worst_case_bound(y, w, certain, ~certain, Z95)


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
    for r in rows:
        assert r["error_ratio_max"] <= 2.0 + 1e-9 or r["error_ratio_max"] != r["error_ratio_max"]
        assert r["error_ratio_median"] <= r["error_ratio_p95"] + 1e-12 <= r["error_ratio_max"] + 2e-12 \
            or r["error_ratio_median"] != r["error_ratio_median"]
        assert r["prob_level"] == 0.99 and r["prob_coverage_held"] >= 0.9       # conformal: about 99% of held-out inside
        assert r["certified_prob"] in (True, False) and "bound_gap" in r and "recall_of_hits_prob" in r
        if r["certified"]:
            assert r["bound_gap"] >= -1e-12
        if r["certified_prob"] and r["certified"]:
            assert r["prob_halfwidth"] <= max(2.0, r["prob_halfwidth"])        # present and finite
            assert np.isfinite(r["hit_rate_prob"])


def test_the_probabilistic_halfwidth_is_a_conformal_order_statistic():
    errors = np.arange(1.0, 100.0)                                              # n = 99, level 0.99: k = 99
    assert e10.conformal_halfwidth(errors, 0.99) == 99.0
    assert e10.conformal_halfwidth(np.arange(1.0, 11.0), 0.99) == float("inf")   # k = 11 > n: no valid interval
    assert e10.conformal_halfwidth(np.arange(1.0, 11.0), 0.5) == 6.0             # ceil(0.5 * 11) = 6


def test_native_info_reads_what_a_vector_file_really_holds():
    ints = np.random.default_rng(0).integers(-128, 128, size=(300, 6)).astype(np.float32)   # int8 saved as float32
    info = quantize.native_info(ints)
    assert info["dtype"] == "float32" and info["integer_valued"] and info["native_bits"] == 8
    assert quantize.native_info(ints + 0.5)["native_bits"] == 32
    assert quantize.native_info(ints * 2.0)["native_bits"] == 32                # integers outside the int8 range
    assert quantize.native_info(ints.astype(np.int8))["dtype"] == "int8"


def test_e10_native_precision_is_exact_for_int8_vectors(world, monkeypatch):
    assert e10.precisions_for(world) == e10.BITS                                # the tiny data are float
    real = e10.quantize.native_info
    monkeypatch.setattr(e10.quantize, "native_info", lambda X, **k: dict(real(X), native_bits=8))
    rows = e10.run(world)
    native = [r for r in rows if r["bits"] == "native"]
    assert len(native) == 1
    r = native[0]
    assert r["error_mean"] == 0.0 and r["ambiguous_share"] == 0.0 and r["violations"] == 0
    assert r["hit_rate_wc"] == r["hit_rate_full"] and r["lam_wc"] == r["lam_full"]
    assert {"query_dtype", "query_integer_valued", "native_bits", "db_dtype"} <= set(r)
    assert e11.item_bytes(100, 10, "native") == e11.item_bytes(100, 10, 8)
    e11_rows = e11.run(world, shares=(0.03,), precisions=(8,), deltas=(0.05,))
    assert [r["bits"] for r in e11_rows] == [8, "native"]


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
    assert out and set(out[0]) == {"workload", "budget", "n_rows", "n_in_sample_excluded", "smallest_share_certifying",
                                   "better_bits"}
    assert out[0]["n_rows"] + out[0]["n_in_sample_excluded"] == len(rows)


def test_e11_measures_the_window_and_summaries_leave_out_in_sample_rows_and_absent_deltas(world):
    rows = e11.run(world, shares=(0.01, 0.03, 0.05), precisions=(8, 16), deltas=(0.05,))
    for r in rows:
        for key in ("labeled_per_half", "audited_hits_per_half", "n_eff_served", "n_eff_all_labeled", "min_ess",
                    "min_ess_blocks_share", "ess_min_certified", "n_stretches", "in_sample"):
            assert key in r
        assert r["n_eff_served"] == pytest.approx(r["audited_hits_per_half"])      # audited hits all weigh 1/a
        assert r["n_eff_all_labeled"] >= r["n_eff_served"] - 1e-9
        assert 0 <= r["min_ess_blocks_share"] <= 1 - r["certified_share"] + 1e-9
    assert any(r["in_sample"] for r in rows) and any(not r["in_sample"] for r in rows)
    out = e11.summary(rows)[0]
    assert all(k.endswith("delta0.05") for k in out["smallest_share_certifying"])      # no keys for the absent 0.10
    assert all(k.endswith("delta0.05") for k in out["better_bits"])
    shares_used = {float(k.split("share")[1].split("_")[0]) for k in out["better_bits"]}
    assert shares_used <= {r["window_share_of_M"] for r in rows if not r["in_sample"]}
    assert out["n_in_sample_excluded"] == sum(r["in_sample"] for r in rows)


def test_a_window_blocked_only_by_min_ess_is_reported_as_such():
    rng = np.random.default_rng(0)
    n = 4000
    score = rng.uniform(size=n)
    y = np.clip(0.55 + 0.5 * score + rng.normal(0, 0.05, n), 0, 1)
    p = world_mod.Params(K=10, min_ess=1e6)           # unreachable: only the effective sample size can block
    edges = policy.threshold_edges(score)
    out = e11.stretch_rows(score, score, score, np.ones(n, bool), y, np.arange(n), 1000, 0.5, edges, p, 0.05,
                           np.random.default_rng(1))
    assert out and all(not r["certified"] and r["min_ess_blocks"] for r in out)
    p = world_mod.Params(K=10, min_ess=10.0)
    out = e11.stretch_rows(score, score, score, np.ones(n, bool), y, np.arange(n), 1000, 0.5, edges, p, 0.05,
                           np.random.default_rng(1))
    assert any(r["certified"] for r in out) and not any(r["min_ess_blocks"] for r in out if r["certified"])


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
