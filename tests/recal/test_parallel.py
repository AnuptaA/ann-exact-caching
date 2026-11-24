import json
import os
import sys
from pathlib import Path

import pytest

from recal import artifacts, matrix, prepare, shards, store as store_mod
from scripts import gen_matrix, prepare as prepare_cli, reduce as reduce_cli, run_shard

ROOT = Path(__file__).resolve().parents[2]
MEDRAG = {"q1_budgets": [0.0001, 0.0003, 0.001], "standin_5pct": 0.0003}


def real_cfg():
    return json.loads((ROOT / "experiments" / "matrix.example.json").read_text())


def test_prepare_dry_run_lists_every_stage_of_a_dataset(capsys):
    cfg = real_cfg()
    prepare_cli.prepare_dataset(cfg, "spacev", dry_run=True)
    out = capsys.readouterr().out
    for stage in ("fetch", "prep", "diskann_index", "diskann_answers", "pgvector_db", "shift_preps", "update_preps",
                  "qvcache_memory"):
        assert f"[spacev] {stage}" in out
    assert "--fixed 0.95:700" in out and "Q1_PG_WORKERS=$(getconf _NPROCESSORS_ONLN)" in out
    assert "--seeds 0 1 2" in out and "--gt-out" in out
    names = [s.name for s in prepare.stages(cfg, "medrag")]
    assert "medrag_budgets" in names and names.index("diskann_answers") < names.index("medrag_budgets")
    assert "pgvector_db" not in [s.name for s in prepare.stages(cfg, "triviaqa")]


def test_tiny_dataset_prepare_pushes_artifacts_with_a_manifest(tmp_path):
    cfg = prepare.tiny_config()
    s = store_mod.Store(store_mod.LocalBackend(tmp_path / "store"))
    manifest = prepare_cli.prepare_dataset(cfg, "tiny", s, root=tmp_path / "prep")
    assert set(manifest["artifacts"]) == {"prep", "diskann_answers"} and manifest["uploaded"]
    stored = artifacts.read_manifest(s, "tiny")
    entry = stored["artifacts"]["prep"]
    assert entry["bytes"] > 0 and entry["code_commit"] and all(f["sha256"] for f in entry["files"])
    assert any("artifacts/tiny/prep/results/online/tiny-a/q.npy" == k for k in s.keys("artifacts/tiny"))


def test_tripclick_artifacts_are_not_uploaded_by_default():
    assert "tripclick" in prepare.RESTRICTED
    cfg = real_cfg()
    s = store_mod.Store(store_mod.LocalBackend("/nonexistent"))
    stage = prepare.Stage("x", None, [])
    assert prepare_cli.push_stage(s, cfg, "tripclick", stage, Path("."), upload=False) is None


def test_shards_are_in_dependency_order_and_throughput_jobs_are_exclusive():
    built = shards.build(real_cfg(), medrag=MEDRAG)
    assert len(built) == 8 and sum(len(v["jobs"]) for v in built.values()) == 629 - 0
    for shard in built.values():
        stages = [j["stage"] for j in shard["jobs"]]
        assert stages == sorted(stages) and shard["artifacts"]
        assert all(j["exclusive"] == (j["kind"] in ("replay", "threads")) for j in shard["jobs"])
        q3 = [j for j in shard["jobs"] if j["exp"] == "q3"]
        assert {j["workload"] for j in q3} == {shard["name"]}
        assert not shard["pull_results"]
    assert "spacev/update_preps" in built["spacev-0.99"]["artifacts"]
    assert "spacev/shift_preps" in built["spacev-0.6"]["artifacts"]


def test_q3_jobs_of_a_workload_stay_together_under_every_sharding():
    cfg = real_cfg()
    by_system = shards.build(cfg, "system", medrag=MEDRAG)
    for name, shard in by_system.items():
        if name.startswith("q3_"):
            assert {j["workload"] for j in shard["jobs"]} == {name[3:]}
            assert {j["exp"] for j in shard["jobs"]} == {"q3"}
    by_question = shards.build(cfg, "question", medrag=MEDRAG)
    assert set(by_question) == {"q1", "q2", "q3", "q4"}


def test_filters_and_cross_shard_inputs():
    cfg = real_cfg()
    only = shards.build(cfg, "workload", question=["q3"], workload=["tripclick"], medrag=MEDRAG)
    shard = only["tripclick"]
    assert {j["exp"] for j in shard["jobs"]} == {"q3"} and shard["pull_results"]
    assert all(p["question"] in ("q1", "q3") for p in shard["pull_results"])
    systems = shards.build(cfg, "workload", system=["aker"], medrag=MEDRAG)
    assert all(j["system"] == "aker" for v in systems.values() for j in v["jobs"])


def test_cli_writes_shard_files_and_artifact_lists(tmp_path):
    budgets = tmp_path / "b.json"
    budgets.write_text(json.dumps(MEDRAG))
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(dict(real_cfg(), medrag_budgets_file=str(budgets))))
    gen_matrix.main(["--config", str(path), "--shard-by", "workload", "--out", str(tmp_path / "jobs"),
                     "--workload", "tripclick"])
    shard = json.loads((tmp_path / "jobs" / "shards" / "tripclick.json").read_text())
    assert (tmp_path / "jobs" / "shards" / "tripclick.txt").read_text().count("\n") == len(shard["jobs"])
    assert json.loads((tmp_path / "jobs" / "shards" / "tripclick.artifacts.json").read_text()) == shard["artifacts"]


@pytest.fixture
def tiny_world(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "PY", sys.executable)
    monkeypatch.setattr(matrix, "PYPATH", f"PYTHONPATH={ROOT}")
    cfg = prepare.tiny_config()
    s = store_mod.Store(store_mod.LocalBackend(tmp_path / "store"))
    prep_dir = tmp_path / "prepare"
    prep_dir.mkdir()
    prepare_cli.prepare_dataset(cfg, "tiny", s, root=prep_dir)
    built = shards.build(cfg, "workload", question=["q1", "q2"], system=["recal"])
    return cfg, s, built, tmp_path


def run_instance(path: Path, shard, s):
    path.mkdir(exist_ok=True)
    here = os.getcwd()
    os.chdir(path)
    try:
        return run_shard.run_shard(shard, s, workers=2, cores_per_job=1, first_core=0)
    finally:
        os.chdir(here)


def test_two_instances_run_their_shards_and_reduce_builds_the_figure_inputs(tiny_world, capsys):
    cfg, s, built, tmp = tiny_world
    assert set(built) == {"tiny-a", "tiny-b"}
    first = run_instance(tmp / "inst1", built["tiny-a"], s)
    second = run_instance(tmp / "inst2", built["tiny-b"], s)
    expected = len(built["tiny-a"]["jobs"])
    assert first["ran"] == second["ran"] == expected and first["failed"] == second["failed"] == 0
    assert (tmp / "inst1" / "results" / "online" / "tiny-a" / "q.npy").exists()
    assert s.keys("results/q1") and s.keys("results/q2")

    again = run_instance(tmp / "inst1", built["tiny-a"], s)
    assert again["ran"] == 0 and again["skipped"] == expected

    fresh = run_instance(tmp / "inst3", built["tiny-a"], s)
    assert fresh["ran"] == 0 and fresh["skipped"] == expected
    assert (tmp / "inst3" / "results" / "q2").exists()

    gen = tmp / "jobs"
    jobs = matrix.runs(cfg)
    table = {h: e for h, e in jobs["runs"].items()
             if e["system"].startswith("recal") and e["cells"][0]["experiment"] in ("q1", "q2")}
    wanted = {k: v for k, v in table.items()}
    out = tmp / "reduce"
    out.mkdir()
    here = os.getcwd()
    os.chdir(out)
    try:
        assert reduce_cli.pull_results(s, wanted) > 0
        cells = {e: [c for c in items if c["hash"] in wanted] for e, items in jobs["cells"].items()}
        report = reduce_cli.reduce(cfg, wanted, cells, out / "figures_data")
    finally:
        os.chdir(here)
    assert not report["missing"] and not report["failed"]
    for name in ("runs", "guarantee", "composition", "q2_ablations"):
        assert (out / "figures_data" / f"{name}.csv").exists()
    guarantee = (out / "figures_data" / "guarantee.csv").read_text().splitlines()
    assert guarantee[-1].split(",")[1] == "pooled" and len(guarantee) > 3
    assert report["tables"]["runs"] == sum(len(c) for c in cells.values())
    assert gen.exists() is False


def test_environment_is_recorded_in_every_result(tiny_world):
    cfg, s, built, tmp = tiny_world
    run_instance(tmp / "inst1", built["tiny-a"], s)
    results = list((tmp / "inst1" / "results" / "q1").rglob("recal.json"))
    assert results
    res = json.loads(results[0].read_text())
    assert res["environment"]["cores"] and res["code_commit"]


def test_q3_tables_have_throughput_with_system_recall_best_splits_and_thread_lines(tmp_path):
    from recal.analysis import figures
    rows = [{"arm": a, "clients": 64, "qps": q, "hit_rate": 0.5, "system_recall": 0.93, "nodes": 10,
             "saturation_clients": 64}
            for a, q in (("nodeonly", 100.0), ("recal_n80", 150.0), ("recal_n20", 180.0))]
    threads = [{"arm": a, "clients": c, "qps": q + c, "mean_ms": 5.0, "p99_ms": 9.0, "hit_rate": 0.5,
                "system_recall": 0.93} for a, q in (("nodeonly", 10), ("recal_n20", 20)) for c in (1, 2)
               for _ in range(3)]
    (tmp_path / "sweep.json").write_text(json.dumps(rows))
    (tmp_path / "best.json").write_text(json.dumps({"recal": {"arm": "recal_n20", "node_share": 20, "qps": 180.0,
                                                             "spec": "recal_n20=x,nodes=1"}}))
    (tmp_path / "threads.json").write_text(json.dumps(threads))
    table = {"1": {"kind": "replay", "workload": "w", "outputs": ["sweep.json"]},
             "2": {"kind": "best_split", "workload": "w", "outputs": ["best.json"]},
             "3": {"kind": "threads", "workload": "w", "outputs": ["threads.json"]}}
    out = figures.q3_rows(table, tmp_path)
    split = {r["arm"]: r for r in out["q3_split"]}
    assert split["recal_n20"]["qps"] == 180.0 and split["recal_n20"]["system_recall"] == 0.93
    assert split["nodeonly"]["system"] == "nodeonly" and split["recal_n80"]["node_share"] == 80
    assert out["q3_best"][0]["node_share"] == 20
    assert len(out["q3_threads"]) == 4 and out["q3_threads"][0]["qps_std"] == 0.0


def test_job_lines_use_absolute_paths_and_pinning_keeps_the_interpreter():
    from simulations.approximate_hits import run_jobs
    built = shards.build(real_cfg(), medrag=MEDRAG)
    lines = [j["line"] for shard in built.values() for j in shard["jobs"]]
    assert all(f"PYTHONPATH={ROOT} " in line for line in lines)
    assert not any(" .venv-" in line or " simulations/" in line for line in lines)
    pinned = run_jobs.pin_line(f"PYTHONPATH={ROOT} {ROOT}/.venv-core/bin/python -u -m recal.cells", 2, 3)
    assert pinned == f"PYTHONPATH={ROOT} taskset -c 2-3 {ROOT}/.venv-core/bin/python -u -m recal.cells"
