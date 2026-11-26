import json
from pathlib import Path

import numpy as np
import pytest

from recal import prepare
from recal.micro import common, e1, e3, e4, e7, e8, e9, fit, quantize, world as world_mod
from scripts import prepare as prepare_cli
from scripts.micro import run as micro_run, summarize

ROOT = Path(__file__).resolve().parents[2]
PARAMS = {"H": 900, "cold_start": 500, "min_ess": 30.0}
BUDGET = 0.6


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("micro")
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


def test_world_uses_the_construction_roles_and_exact_labels(world):
    assert world.built.n_entries > 0 and world.history_len == 900
    held = world.heldout()
    assert held["features"].has.all() and 0 < held["features"].recall.mean() <= 1
    assert len(held["inverse"]) == len(world.stream) - world.history_len
    cal, _ = world.role_rows("cal")
    assert len(cal.recall) == world.built.info["roles"]["cal"]


def test_e1_models_hit_rates_and_agreement(world):
    rows = e1.run(world)
    assert [r["model"] for r in rows] == ["logit", "linear"]
    logit = rows[0]
    assert logit["lam"] == pytest.approx(world.built.lam)
    for r in rows:
        assert -1 <= r["spearman"] <= 1 and 0 <= r["hit_rate"] <= r["oracle_hit_rate"] + 1e-12
        assert r["gap"] == pytest.approx(r["oracle_hit_rate"] - r["hit_rate"])
    assert e1.summarize(rows)["max_hit_rate_difference"] == pytest.approx(abs(rows[0]["hit_rate"] - rows[1]["hit_rate"]))


def test_fit_helpers():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(300, 3))
    y = np.clip(0.5 + 0.2 * F[:, 0], 0, 1)
    linear = fit.fit_linear(F, y)
    assert np.corrcoef(fit.score(linear, F), y)[0, 1] > 0.95
    assert fit.oracle_count(np.array([1.0, 1.0, 0.0, 0.0]), 0.9) == 2
    assert fit.oracle_count(np.array([0.5, 0.4]), 0.9) == 0
    assert not fit.decide(np.array([1.0]), np.array([True]), None).any()


def test_quantizer_error_shrinks_with_bits():
    X = np.random.default_rng(1).normal(size=(50, 32)).astype(np.float32)
    err8, err4 = (np.abs(quantize.quantize(X, b) - X).max() for b in (8, 4))
    assert err8 < err4 / 8 and quantize.quantize(np.ones((2, 4), np.float32), 4).shape == (2, 4)


def test_e3_eight_bits_stay_closer_than_four(world):
    rows = e3.run(world)
    by_bits = {r["bits"]: r for r in rows}
    assert set(by_bits) == {8, 4}
    assert by_bits[8]["score_diff_median"] <= by_bits[4]["score_diff_median"]
    assert by_bits[8]["score_diff_p95"] <= by_bits[8]["score_diff_p99"] + 1e-12
    assert 0 <= by_bits[4]["flip_share"] <= 1 and by_bits[8]["lam_full"] == by_bits[4]["lam_full"]


def test_e4_methods_and_the_unbounded_intruder_check(world):
    rows, diag = e4.run(world)
    by = {r["method"]: r for r in rows}
    assert list(by) == list(e4.METHODS)
    shares = [by[f"c{c}"]["contains_reference_share"] for c in e4.NEAREST]
    assert shares == sorted(shares)
    assert all(0 <= r["contains_reference_share"] <= 1 for r in rows)
    ex = diag["intruder_examined"]
    assert ex["median"] <= ex["p95"] <= ex["p99"] <= ex["max"]
    agree = diag["core_agreement"]
    assert agree["offset_abs_diff_p95"] < 1e-3 and agree["size_abs_diff_p95"] < 1e-3


def test_e4_with_all_entries_the_set_contains_the_reference(world):
    table = e4.EntryTable(world)
    held = world.heldout()
    out = e4.compute(table, held["vec"][:50], held["truth"][:50], m=table.n)
    assert out["methods"]["m_nearest"]["contains"].all()


def test_e7_degree_and_ef_table(world):
    pytest.importorskip("hnswlib")
    rows, summary = e7.run(world, max_queries=100)
    assert len(rows) == len(e7.DEGREES) * len(e7.EFS)
    for degree in e7.DEGREES:
        by_ef = {r["ef"]: r for r in rows if r["degree"] == degree}
        assert by_ef[256]["exact_share"] >= by_ef[16]["exact_share"]
    r = rows[0]
    assert r["edge_bytes_per_vector_4"] > r["edge_bytes_per_vector_sized"] > 0 and r["id_bytes_pool"] == 2
    assert summary["pool_vectors"] == len(world.built.pool_ids)


def test_e8_window_grows_with_the_cap_and_small_caps_abstain(world):
    rows = e8.run(world)
    caps = {r["cap_share_of_M"]: r for r in rows}
    assert set(caps) == {"uncapped", 0.01, 0.03, 0.05}
    assert caps[0.01]["W_requests"] < caps[0.03]["W_requests"] < caps[0.05]["W_requests"]
    labeled = (1 - caps[0.03]["reference_hit_rate"]) + 0.1 * caps[0.03]["reference_hit_rate"]
    assert caps[0.03]["calibration_labeled"] == pytest.approx(caps[0.03]["W_requests"] * labeled / 2)
    assert caps["uncapped"]["certified_share"] == 1.0 and caps[0.01]["certified_share"] == 0.0
    assert caps[0.01]["hit_rate"] == 0.0
    assert e8.item_bytes(512, 10) == 512 + 40 + 31


def test_e9_reports_both_methods_with_and_without_the_mini_cache(tiny):
    rows = e9.run("tiny-a", "results/online/tiny-a", "trace", BUDGET, 4,
                  e9.sim_extra(PARAMS) + ["--max-requests", "1500"], window=200)
    assert [(r["method"], r["mini_cache"]) for r in rows] == [
        ("recal", True), ("recal", False), ("lru", True), ("lru", False)]
    for r in rows:
        assert 0 <= r["hit_rate"] <= 1 and r["n_scored"] == 1050
        if not r["mini_cache"]:
            assert r["mini_hit_rate"] == 0 and r["mini_share_of_hits"] == 0


def run_cli(tiny, *extra):
    return micro_run.main(["--dataset", "tiny", "--config", str(tiny / "matrix.json"), "--out", str(tiny / "out"),
                           "--params", json.dumps(PARAMS), *extra])


def test_runner_skips_a_finished_experiment_and_pushes_to_the_store(tiny, capsys):
    store = tiny / "store"
    assert run_cli(tiny, "--experiments", "E8,E1", "--store", str(store)) == 0
    out = capsys.readouterr().out
    assert "E1: wrote" in out and "E8: wrote" in out and out.index("E1: wrote") < out.index("E8: wrote")
    assert (store / "micro" / "E1" / "tiny-a" / "e1.json").exists()
    assert (store / "micro" / "E8" / "tiny-a" / "e8.csv").exists()
    first = json.loads((tiny / "out" / "E1" / "tiny-a" / "e1.json").read_text())
    assert first["config_hash"] and first["code_commit"] and "environment" in first and first["settings"]["budgets"] == [BUDGET]
    run_cli(tiny, "--experiments", "E1", "--store", str(store))
    out = capsys.readouterr().out
    assert "skipped" in out and "building" not in out
    run_cli(tiny, "--experiments", "E1", "--store", str(store), "--max-requests", "2000")
    assert "E1: wrote" in capsys.readouterr().out


def test_summarize_writes_a_table_per_experiment(tiny, capsys):
    store = tiny / "store2"
    run_cli(tiny, "--experiments", "E1", "--store", str(store), "--force")
    assert summarize.main(["--store", str(store), "--pulled", str(tiny / "pulled"), "--out", str(tiny / "tables")]) == 0
    md = (tiny / "tables" / "E1.md").read_text().splitlines()
    assert md[0].startswith("| workload | budget | model") and len(md) == 4
    assert (tiny / "tables" / "E1.csv").exists()


def test_unknown_experiment_is_refused():
    with pytest.raises(SystemExit):
        micro_run.parse_experiments("E1,E5")
    assert micro_run.parse_experiments("e9,E1") == ["E1", "E9"]


def test_config_hash_depends_on_settings():
    assert common.config_hash({"a": 1}) != common.config_hash({"a": 2})
    assert common.config_hash({"a": 1, "b": 2}) == common.config_hash({"b": 2, "a": 1})


# ---- prepare stage selection --------------------------------------------------------------------------------

def matrix_cfg():
    return json.loads((ROOT / "experiments" / "matrix.json").read_text())


def test_stage_names_and_the_working_set_alias():
    assert prepare.parse_stage_names("fetch, prep,working_set") == ["fetch", "prep", "medrag_budgets"]
    with pytest.raises(ValueError, match="unknown stage"):
        prepare.parse_stage_names("fetch,nope")


def test_stage_selection_filters_stages_in_order():
    cfg = matrix_cfg()
    assert [s.name for s in prepare.stages(cfg, "spacev", ["prep", "fetch"])] == ["fetch", "prep"]
    assert [s.name for s in prepare.stages(cfg, "laion", ["fetch", "base", "prep"])] == ["fetch", "base", "prep"]
    assert len(prepare.stages(cfg, "spacev")) > 2


def test_working_set_reads_the_exact_answers_without_the_diskann_stage():
    cfg = matrix_cfg()
    exact = prepare.stages(cfg, "medrag", ["fetch", "prep", "medrag_budgets"])[-1]
    assert "--backend stored" in exact.commands[0] and "--backend-file" not in exact.commands[0]
    full = [s for s in prepare.stages(cfg, "medrag") if s.name == "medrag_budgets"][0]
    assert "--backend-file" in full.commands[0]


def test_prepare_reports_skipped_stages(capsys):
    code = prepare_cli.main(["--config", str(ROOT / "experiments" / "matrix.json"), "--dataset", "spacev",
                             "--stages", "fetch,prep", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0 and "[spacev] diskann_index: skipped (not in --stages)" in out
    assert "[spacev] prep" in out and "[spacev] fetch" in out and "[spacev] fetch: skipped" not in out
