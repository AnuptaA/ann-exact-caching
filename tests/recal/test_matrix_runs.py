import json
from pathlib import Path

from recal import matrix, matrix_hash as mh
from scripts import gen_matrix

CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "matrix.json"
MEDRAG = {"q1_budgets": [0.0001, 0.0003, 0.001], "standin_5pct": 0.0003}


def cfg():
    return json.loads(CONFIG.read_text())


def lines(exp):
    return [j.line for j in matrix.cell_jobs(cfg(), MEDRAG)[0][exp]]


def line(items, *needles):
    return [x for x in items if all(n in x for n in needles)][0]


def test_options_that_do_not_change_a_run_do_not_change_its_hash():
    base = line(lines("q2"), "/spacev-0.3/no-union_s0")
    assert mh.cell_hash(base) == mh.cell_hash(base.replace("results/q2/spacev-0.3/no-union_s0", "results/elsewhere/x"))
    assert mh.cell_hash(base) == mh.cell_hash(base.replace("--seed 0", "--seed 0 --tag _label"))


def test_options_that_change_a_run_change_its_hash():
    base = line(lines("q2"), "/spacev-0.3/no-union_s0")
    h = mh.cell_hash(base)
    for old, new in (("--seed 0", "--seed 1"), ("--target 0.9", "--target 0.85"), ("--budget 0.05", "--budget 0.1"),
                     ("--backend file", "--backend exact"), ("--strict-modes", "--strict-modes --idealized"),
                     ("--stream trace", "--stream random"), ("--target 0.9", "--target 0.9 --history 10000")):
        assert mh.cell_hash(base.replace(old, new)) != h, (old, new)


def test_cells_of_different_experiments_coincide_when_they_are_the_same_run():
    q1, q3, q4 = lines("q1"), lines("q3"), lines("q4")
    base = mh.cell_hash(line(q1, "/spacev-0.6/recal_deployable_t0.9_b0.05_s0"))
    assert mh.cell_hash(line(q3, "/q3/spacev-0.6/recal_n0")) == base
    assert mh.cell_hash(line(q3, "/q3/spacev-0.6/recal_n80")) == \
        mh.cell_hash(line(q1, "/spacev-0.6/recal_deployable_t0.9_b0.01_s0"))
    assert mh.cell_hash(line(q3, "/q3/spacev-0.6/recal_n60")) == mh.cell_hash(line(q4, "/q4/spacev-0.6/budget_b0.02_s0"))
    assert mh.cell_hash(line(q3, "/q3/spacev-0.6/aker_n0")) == mh.cell_hash(line(q1, "/spacev-0.6/aker_diskann_b0.05_s0"))
    assert mh.cell_hash(line(q3, "/q3/spacev-0.6/qvcache_n0")) == \
        mh.cell_hash(line(q1, "/spacev-0.6/qvcache_diskann_b0.05_s0"))


def test_runs_keep_one_job_per_distinct_run_and_both_experiments_find_it():
    result = matrix.runs(cfg(), medrag=MEDRAG)
    h = mh.cell_hash(line(lines("q1"), "/spacev-0.6/recal_deployable_t0.9_b0.05_s0"))
    entry = result["runs"][h]
    assert {c["experiment"] for c in entry["cells"]} == {"q1", "q3"}
    emitted = [j.line for v in result["jobs"].values() for j in v]
    assert len(emitted) == len(set(emitted)) == len(result["runs"])
    assert not any("/q3/spacev-0.6/recal_n0/" in x for x in emitted)


def test_every_cell_resolves_to_a_run_and_the_table_adds_up():
    result = matrix.runs(cfg(), medrag=MEDRAG)
    for cells in result["cells"].values():
        assert all(cell["hash"] in result["runs"] for cell in cells)
    table = matrix.sharing_table(cfg(), medrag=MEDRAG)
    assert table["total"]["runs_emitted"] == len(result["runs"]) == sum(table[e]["runs_emitted"] for e in matrix.EXPERIMENTS)
    d = table["q3_detail"]
    assert d["hit_rate_runs_shared_with_q1"] >= 24
    assert table["total"]["cells"] > table["total"]["runs_emitted"]


def test_cli_writes_deduplicated_jobs_and_the_mapping(tmp_path, capsys):
    budgets = tmp_path / "medrag_budgets.json"
    budgets.write_text(json.dumps(MEDRAG))
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(dict(cfg(), medrag_budgets_file=str(budgets))))
    out = tmp_path / "jobs"
    gen_matrix.main(["--config", str(path), "--out", str(out)])
    mapping = json.loads((out / "matrix_runs.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    emitted = sum(len((out / f"{e}.txt").read_text().splitlines()) for e in matrix.EXPERIMENTS)
    assert emitted == len(mapping) == manifest["distinct_runs"]
    assert sum(len(v["cells"]) for v in mapping.values()) == sum(manifest["cells"].values())
    assert all(v["kind"] and v["system"] for v in mapping.values())
