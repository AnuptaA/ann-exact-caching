import json
from pathlib import Path

from recal import matrix
from recal.measure import jobs_util as ju
from scripts import gen_matrix

CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "matrix.json"
MEDRAG = {"q1_budgets": [0.0001, 0.0003, 0.001], "standin_5pct": 0.0003}


def cfg():
    return json.loads(CONFIG.read_text())


def lines(exp, **kw):
    return [j.line for j in matrix.cell_jobs(cfg(), MEDRAG)[0][exp]]


def test_distinct_runs_follow_the_appendix():
    table = matrix.sharing_table(cfg(), medrag=MEDRAG)
    assert table["q1"]["cells"] == table["q1"]["runs_emitted"] == 384
    assert table["q2"]["cells"] == table["q2"]["runs_emitted"] == 56
    d = table["q3_detail"]
    assert d["split_points"] == 128 and d["node_only"] == 8 and d["thread_lines"] == 32
    assert d["hit_rate_runs_new"] + d["hit_rate_runs_shared_with_q1"] == 120
    assert table["q4"]["cells"] == 93 and table["q4"]["runs_emitted"] == 90


def test_without_medrag_budgets_no_medrag_cell_is_emitted():
    jobs, skipped = matrix.cell_jobs(cfg(), None)
    assert skipped == ["medrag"]
    assert not [j for v in jobs.values() for j in v if j.workload == "medrag"]


def test_q1_has_sixteen_systems_per_cell_and_no_qvcache_sweep():
    c = cfg()
    systems = matrix.q1_systems(c, c["workloads"]["spacev-0.3"])
    labels = [s[0] for s in systems]
    assert len(systems) == 16 and sum(x.startswith("recal_") for x in labels) == 8
    assert not [x for x in labels if "-d" in x and x.startswith("qvcache")]
    one = [x for x in lines("q1") if "/spacev-0.3/" in x and "_b0.01_s0" in x]
    assert len(one) == 16 and all("--coord-bytes 1" in x for x in one)
    assert all("--deviation-factor 0.25" in x for x in one if "qvcache" in x)
    assert all("--deviation-factor 0.075" in x for x in lines("q1") if "/laion/" in x and "qvcache" in x)
    assert sum("--idealized" in x for x in one) == 4
    assert all(ju.is_job_line(x) for x in lines("q1"))


def test_the_target_sweep_is_q1_only_and_modes_follow_the_backend():
    one = [x for x in lines("q1") if "/laion/" in x and "_b0.05_" in x and "recal_" in x]
    assert {t for t in (0.8, 0.85, 0.9, 0.95) if any(f"--target {t:g} " in x for x in one)} == {0.8, 0.85, 0.9, 0.95}
    assert all("--backend exact" in x for x in one if "--idealized" in x)
    assert all("--backend file" in x for x in one if "--idealized" not in x)
    for exp in ("q2", "q3", "q4"):
        assert all("--target 0.9 " in x for x in lines(exp) if "recal.cells --method recal" in x)


def test_q2_has_every_ablation_and_no_oracle():
    q2 = lines("q2")
    assert {x.split("/")[3].rsplit("_s", 1)[0] for x in q2} == {
        "no-union", "no-expansion", "no-shells", "lru", "no-calibration", "no-recalibration", "no-mini-cache"}
    assert all("--budget 0.0003 " in x for x in q2 if "/medrag/" in x)
    assert not [x for x in q2 if "oracle" in x]


def test_q3_split_points_pair_a_hit_rate_run_with_a_node_cache_share():
    hits = lines("q3")
    assert len(hits) == 120
    mine = [x for x in hits if "/q3/spacev-0.3/" in x]
    assert len(mine) == 15
    assert any("recal_n80" in x and "--budget 0.01 " in x for x in mine)
    assert any("aker_n0" in x and "--budget 0.05 " in x for x in mine)
    assert all("--backend file" in x for x in mine)


def test_q3_followups_run_in_stage_order_with_exclusive_throughput():
    r = matrix.runs(cfg(), medrag=MEDRAG)
    follow = [j for j in r["jobs"]["q3"] if j.kind in ("replay", "best_split", "threads")]
    assert len(follow) == 24 and all(j.exclusive == (j.kind != "best_split") for j in follow)
    mine = {j.kind: j for j in follow if j.workload == "spacev-0.3"}
    sweep = mine["replay"].line
    assert sweep.count("--arm ") == 16 and "--arm nodeonly=none,nodes=" in sweep and "--clients 64 " in sweep
    assert "--arm recal_n80=" in sweep and ",nodes=0" in sweep and "--repetitions 3" in sweep
    threads = mine["threads"].line
    assert "--clients 1 2 4 8 16 32 64" in threads and "best_split get" in threads.replace("-m recal.system_benefit.", "")
    assert threads.count("--arm ") == 4
    assert [j.stage for j in (mine["replay"], mine["best_split"], mine["threads"])] == [1, 2, 3]
    assert len(r["split_points"]) == 128


def test_q4_dimensions_and_the_update_job_contract():
    jobs = matrix.cell_jobs(cfg(), MEDRAG)[0]["q4"]
    q4 = [j.line for j in jobs]
    assert sum("--no-rebuild-signals" in x for x in q4) == 18 and sum("/shift_" in x for x in q4) == 36
    assert sum("--k-mixed" in x for x in q4) == 2 and sum(" --k " in x and "--k-mixed" not in x for x in q4) == 10
    assert not [x for x in q4 if "burst" in x]
    updates = [j for j in jobs if j.kind == "update"]
    assert len(updates) == 30
    ids = [j.cell.split("/updates/")[1] for j in updates]
    assert len(set(ids)) == 30
    assert all("--backend pgvector --pg-dsn @$PG_CLONES/" in j.line for j in updates)
    assert not [j for j in updates if j.system == "update_qvcache"]
    prox = [j for j in updates if j.system.startswith("update_proximity")]
    assert len(prox) == 6 and all("targeted10" in j.line for j in prox)
    assert all("--budget 0.0003 " in j.line for j in updates if j.workload == "medrag")


def test_cli_table_writes_files_manifest_and_counts(tmp_path, capsys):
    path = tmp_path / "cfg.json"
    budgets = tmp_path / "medrag_budgets.json"
    budgets.write_text(json.dumps(MEDRAG))
    path.write_text(json.dumps(dict(cfg(), medrag_budgets_file=str(budgets))))
    gen_matrix.main(["--config", str(path), "--table"])
    out = capsys.readouterr().out
    assert "q1" in out and "384" in out and "128 split points" in out and "32 thread lines" in out
    gen_matrix.main(["--config", str(path), "--out", str(tmp_path / "jobs"), "--only", "q2"])
    assert json.loads((tmp_path / "jobs" / "manifest.json").read_text())["cells"] == {"q2": 56}
    assert len((tmp_path / "jobs" / "q2.txt").read_text().splitlines()) == 56


def test_a_generated_line_has_a_config_hash_that_follows_its_arguments():
    q2 = lines("q2")
    a = ju.jobline_hash(q2[0])
    b = ju.jobline_hash(q2[0].replace("--seed 0", "--seed 1"))
    assert a and b and a != b
