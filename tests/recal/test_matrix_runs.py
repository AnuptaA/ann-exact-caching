import json
from pathlib import Path

from recal import matrix, matrix_hash as mh
from scripts import gen_matrix

CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "matrix.example.json"


def cfg():
    return json.loads(CONFIG.read_text())


def line(lines, *needles):
    return [x for x in lines if all(n in x for n in needles)][0]


def test_options_that_do_not_change_a_run_do_not_change_its_hash():
    base = line(matrix.q2_lines(cfg()), "/spacev-0.3/default_s0")
    assert mh.cell_hash(base) == mh.cell_hash(base.replace("results/q2/spacev-0.3/default_s0", "results/elsewhere/x"))
    assert mh.cell_hash(base) == mh.cell_hash(base.replace("--seed 0", "--seed 0 --tag _label"))


def test_options_that_change_a_run_change_its_hash():
    base = line(matrix.q2_lines(cfg()), "/spacev-0.3/default_s0")
    h = mh.cell_hash(base)
    for old, new in (("--seed 0", "--seed 1"), ("--target 0.9", "--target 0.85"), ("--budget 0.05", "--budget 0.1"),
                     ("--backend file", "--backend exact"), ("--strict-modes", "--strict-modes --idealized"),
                     ("--stream trace", "--stream random"), ("--target 0.9", "--target 0.9 --history 10000")):
        assert mh.cell_hash(base.replace(old, new)) != h, (old, new)


def test_default_cells_hash_like_their_namesakes_in_other_experiments():
    c = cfg()
    q1, q2 = matrix.q1_lines(c), matrix.q2_lines(c)
    a = line(q1, "/spacev-0.3/recal_deployable_t0.9_b0.05_s0")
    b = line(q2, "/spacev-0.3/default_s0")
    assert mh.cell_hash(a) == mh.cell_hash(b)
    medrag_q1 = line(q1, "/medrag/recal_deployable_t0.9_b0.001_s1")
    assert mh.cell_hash(medrag_q1) == mh.cell_hash(line(q2, "/medrag/default_s1"))
    # the QVCache sweep value equal to the paper default is the default cell
    default = line(q1, "/spacev-0.3/qvcache_exact_b0.01_s0")
    sweep = line(q1, "/spacev-0.3/qvcache-d0.25_exact_b0.01_s0")
    assert mh.cell_hash(default) == mh.cell_hash(sweep)
    assert mh.cell_hash(line(q1, "/tripclick/qvcache_exact_b0.05_s0")) == \
        mh.cell_hash(line(q1, "/tripclick/qvcache-d0.075_exact_b0.05_s0"))
    assert mh.cell_hash(default) != mh.cell_hash(line(q1, "/spacev-0.3/qvcache-d0.5_exact_b0.01_s0"))


def test_q3_q4_cells_coincide_with_q1_and_q2_runs():
    c = cfg()
    q1, q2 = matrix.q1_lines(c), matrix.q2_lines(c)
    q3, q4 = matrix.q3_lines(c), matrix.q4_lines(c)
    base = mh.cell_hash(line(q1, "/spacev-0.6/recal_deployable_t0.9_b0.05_s0"))
    assert mh.cell_hash(line(q3, "/q3/spacev-0.6/b0.05/recal_s100_s0")) == base
    assert mh.cell_hash(line(q4, "/q4/spacev-0.6/recal_b0.05_h30000_s0")) == base
    assert mh.cell_hash(line(q4, "/q4/spacev-0.6/k10_s0")) == base
    assert mh.cell_hash(line(q4, "/q4/tripclick/burst_b0.05_mini_s0")) == \
        mh.cell_hash(line(q1, "/tripclick/recal_deployable_t0.9_b0.05_s0"))
    assert mh.cell_hash(line(q4, "/q4/tripclick/burst_b0.05_nomini_s0")) == mh.cell_hash(line(q2, "/tripclick/no-mini-cache_s0"))
    # the same absolute ReCal budget from different (budget, split) pairs is one run
    assert mh.cell_hash(line(q3, "/q3/spacev-0.3/b0.01/recal_s50_s0")) == mh.cell_hash(line(q3, "/q3/spacev-0.3/b0.05/recal_s10_s0"))


def test_two_experiments_with_an_identical_cell_give_one_job_line_and_both_find_it():
    result = matrix.runs(cfg())
    h = mh.cell_hash(line(matrix.q1_lines(cfg()), "/spacev-0.3/recal_deployable_t0.9_b0.05_s0"))
    entry = result["runs"][h]
    experiments = {c["experiment"] for c in entry["cells"]}
    assert {"q1", "q2"} <= experiments
    outs = {c["out"] for c in entry["cells"]}
    assert "results/q1/spacev-0.3/recal_deployable_t0.9_b0.05_s0" in outs and "results/q2/spacev-0.3/default_s0" in outs
    assert entry["out"] == "results/q1/spacev-0.3/recal_deployable_t0.9_b0.05_s0"
    emitted = [x for lines in result["jobs"].values() for x in lines]
    assert len(emitted) == len(set(emitted)) == len(result["runs"])
    assert not any("/q2/spacev-0.3/default_s0" in x for x in result["jobs"]["q2"])


def test_every_cell_resolves_to_a_run_and_the_sharing_table_adds_up():
    c = cfg()
    result = matrix.runs(c)
    for exp, cells in result["cells"].items():
        assert all(cell["hash"] in result["runs"] for cell in cells)
    table = matrix.sharing_table(c)
    assert table["q1"]["cells"] == 1050 and table["total"]["cells"] == 2014
    assert table["total"]["runs_emitted"] == len(result["runs"]) < 2014
    assert table["q1"]["repeated_within"] == 50 and table["q2"]["shared_with_other"] >= 16


def test_cli_writes_deduplicated_jobs_and_the_mapping(tmp_path, capsys):
    gen_matrix.main(["--config", str(CONFIG), "--out", str(tmp_path)])
    mapping = json.loads((tmp_path / "matrix_runs.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    emitted = sum(len((tmp_path / f"{e}.txt").read_text().splitlines()) for e in ("q1", "q2", "q3", "q4"))
    assert emitted == len(mapping) == manifest["distinct_runs"] and sum(manifest["cells"].values()) == 2014
    assert sum(len(v["cells"]) for v in mapping.values()) == 2014
    gen_matrix.main(["--config", str(CONFIG), "--table"])
    assert "cells shared with another experiment" in capsys.readouterr().out


def test_prunability_stats_cover_every_cell_and_reductions_only_remove_runs():
    from scripts import prunability_stats as ps
    c = cfg()
    data = ps.collect(c)
    rows = ps.summarize(data["cells"])
    assert sum(r["cells"] for r in rows.values()) == 2014 == len(data["cells"])
    runs, units = ps.total_units(data["cells"])
    assert runs == len(matrix.runs(c)["runs"]) and units > 0
    effects = {name: ps.effect(data["cells"], fn) for name, fn in ps.rules(c).items()}
    assert all(e["runs_after"] <= e["runs_before"] and e["units_saved"] >= 0 for e in effects.values())
    assert sum(r["owned"] for r in rows.values()) == runs
    updates = [e for n, e in effects.items() if n.startswith("Q4 updates")][0]
    assert updates["dropped_cells"] == 96
    through = [e for n, e in effects.items() if n.startswith("Q3 throughput")][0]
    assert through["dropped_cells"] == 84
