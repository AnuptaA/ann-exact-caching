import json
from pathlib import Path

from recal import matrix
from recal.measure import jobs_util as ju
from scripts import gen_matrix

CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "matrix.example.json"

# the design: cells per experiment (2 seeds unless noted)
# QVCache is left out of the update runs: the artifact has no update path (the paper routes updates to the backend)
DESIGN = {"q1": 1050, "q2": 128, "q3": 552, "q4": 284}


def cfg():
    return json.loads(CONFIG.read_text())


def test_cells_per_experiment_follow_the_design_and_the_table():
    lines = matrix.all_lines(cfg())
    table = matrix.design_table(cfg())
    for name, expected in DESIGN.items():
        assert len(lines[name]) == table[name]["cells"] == expected, name
    assert table["total"]["cells"] == sum(DESIGN.values())


def test_q1_design_dimensions():
    c = cfg()
    pairs = sum(len(w["budgets"]) for w in c["workloads"].values())
    assert len(c["workloads"]) == 8 and pairs == 25
    systems = matrix.q1_systems(c)
    labels = [s[0] for s in systems]
    assert len(systems) == 21 and sum(l.startswith("recal_") for l in labels) == 8
    for system in ("aker", "proximity-a", "proximity-b", "qvcache"):
        assert {f"{system}_exact", f"{system}_diskann"} <= set(labels)
    assert sum(l.startswith("qvcache-d") for l in labels) == 5
    medrag = c["workloads"]["medrag"]["budgets"]
    assert medrag == [0.0001, 0.0003, 0.001, 0.003]


def test_q1_lines_carry_backend_flags_history_default_and_stored_precision():
    lines = matrix.q1_lines(cfg())
    one = [x for x in lines if "/spacev-0.3/" in x and "_b0.01_s0" in x]
    assert len(one) == 21 and all("--coord-bytes 1" in x and "--history-frac" not in x for x in one)
    diskann = [x for x in one if "aker_diskann" in x][0]
    assert "--backend file --backend-file results/backends/spacev-aker-0.3/diskann.npz --backend-target 0.95" in diskann
    assert any("--prox-percentile 0.1 --tag _p0.1" in x and "proximity_p0.1.json" in x for x in one)
    assert sum("--idealized" in x for x in one) == 4
    sweep = [x for x in one if "qvcache-d" in x]
    assert len(sweep) == 5 and all("--backend exact" in x and "--deviation-factor" in x for x in sweep)
    assert all(ju.is_job_line(x) for x in lines)


def test_q2_has_the_default_and_every_ablation_at_the_q2_budget():
    c = cfg()
    lines = matrix.q2_lines(c)
    medrag = [x for x in lines if "/q2/medrag/" in x]
    assert len(medrag) == 16 and all("--budget 0.001 " in x for x in medrag)
    labels = {x.split("/")[3].rsplit("_s", 1)[0] for x in lines}
    assert labels == {"default", "no-union", "no-expansion", "no-shells", "lru", "no-calibration",
                      "no-recalibration", "no-mini-cache"}


def test_q3_lines_cover_splits_the_chosen_split_and_five_throughput_systems():
    c = cfg()
    lines = matrix.q3_lines(c)
    mine = [x for x in lines if "/q3/spacev-0.3/b0.05/" in x]
    assert sum("recal_s" in x and "recal.cells" in x for x in mine) == 12 and any("recal_chosen_s1" in x for x in mine)
    assert any("recal.system_benefit.choose run" in x for x in mine)
    throughput = [x for x in mine if "throughput_replay.py" in x]
    assert len(throughput) == 5 and all("--clients 1 2 4 8 16 32 64" in x and "--repetitions 3" in x for x in throughput)
    assert any("--arm none" in x for x in throughput)
    assert any("recal=" in x and "--nodes $(" in x for x in throughput)
    assert any("--budget $(" in x and "recal_budget_fraction" in x for x in mine)
    assert sum("pgvector.json" in x and "depth_latency_pgvector" in x for x in lines) == 8


def test_q4_dimensions_and_the_update_job_contract():
    c = cfg()
    lines = matrix.q4_lines(c)
    assert not [x for x in lines if "/q4/medrag/" in x and "updates" not in x]
    assert sum("--no-rebuild-signals" in x for x in lines) == 30
    updates = [x for x in lines if "/updates/" in x]
    assert len(updates) == 144
    live = [x for x in updates if "--backend pgvector" in x]
    assert len(live) == 36 and all("--pg-dsn @$PG_CLONES/" in x for x in live)
    ids = [x.split("/updates/")[1].split("/")[0] for x in updates]
    assert len(set(ids)) == 144
    one = [x for x in live if "spacev-0.99_uniform1_recal_s0" in x][0]
    assert "$PG_CLONES/spacev-0.99_uniform1_recal_s0.dsn" in one and "results/updates/preps/spacev-0.99_uniform1" in one
    assert all("--budget 0.001 " in x for x in updates if "/medrag/" in x)
    bursts = [x for x in lines if "burst_b" in x]
    assert len(bursts) == 12 and sum("--no-mini-cache" in x for x in bursts) == 6
    grid = [x for x in lines if "/q4/tripclick/" in x and "recal_b" in x]
    assert len(grid) == 16


def test_cli_table_writes_files_manifest_and_counts(tmp_path, capsys):
    gen_matrix.main(["--config", str(CONFIG), "--table"])
    out = capsys.readouterr().out
    assert "q1: 1050 cells = 25 (workload, budget) x 21 systems x 2 seeds" in out and "total: 2014 cells" in out
    gen_matrix.main(["--config", str(CONFIG), "--out", str(tmp_path), "--only", "q2"])
    assert json.loads((tmp_path / "manifest.json").read_text())["cells"] == {"q2": 128}
    assert len((tmp_path / "q2.txt").read_text().splitlines()) == 128


def test_a_generated_line_has_a_config_hash_that_follows_its_arguments():
    lines = matrix.q2_lines(cfg())
    a = ju.jobline_hash(lines[0])
    b = ju.jobline_hash(lines[0].replace("--seed 0", "--seed 1"))
    assert a and b and a != b
