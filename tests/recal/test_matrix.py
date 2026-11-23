import json
from pathlib import Path

from recal import matrix
from recal.measure import jobs_util as ju
from scripts import gen_matrix

CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "matrix.example.json"


def cfg():
    return json.loads(CONFIG.read_text())


def test_example_config_counts_per_experiment():
    lines = matrix.all_lines(cfg())
    assert len(lines["q1"]) == 25 * 2 * 11 and len(lines["q2"]) == 3 * 8 * 2
    assert len(lines["q3"]) == 6 + 8 + 2 + 1 + 1
    assert all(ju.is_job_line(line) for line in lines["q1"] + lines["q2"] + lines["q4"])


def test_q1_cells_cover_targets_modes_and_baselines():
    lines = matrix.q1_lines(cfg())
    first = [x for x in lines if "/spacev/zipf0.3/" in x and "_b0.01_s0" in x]
    assert len(first) == 11
    assert sum("--idealized" in x for x in first) == 4 and sum("--method aker " in x for x in first) == 1
    assert any("--prox-percentile 0.1 --tag _p0.1" in x and "proximity_p0.1.json" in x for x in first)
    assert all("--history-frac" not in x and "--coord-bytes 1" in x for x in first)
    assert {f"--target {t}" for t in ("0.8", "0.85", "0.9", "0.95")} <= {
        f"--target {t}" for x in first for t in ("0.8", "0.85", "0.9", "0.95") if f"--target {t} " in x}


def test_q2_has_default_and_every_ablation():
    lines = matrix.q2_lines(cfg())
    one = [x for x in lines if "/spacev/zipf0.99/" in x and "_s0" in x]
    labels = {x.split("/")[4].split(" ")[0] for x in one}
    assert len(one) == 8 and "default_s0" in labels and "lru_s0" in labels and "no-mini-cache_s0" in labels


def test_q3_and_q4_reference_their_inputs():
    c = cfg()
    q3 = matrix.q3_lines(c)
    assert any("--nodes 100000" in x and "--arm nodecache" not in x for x in q3)
    assert any("recal_s50=" in x and "--nodes 100000" in x for x in q3)
    q4 = matrix.q4_lines(c)
    assert any("--no-rebuild-signals" in x for x in q4) and any("--no-mini-cache" in x for x in q4)
    assert any("--updates data/upd_targeted5.npz" in x for x in q4)


def test_cli_writes_files_manifest_and_counts(tmp_path, capsys):
    gen_matrix.main(["--config", str(CONFIG), "--out", str(tmp_path), "--only", "q2"])
    out = capsys.readouterr().out
    assert "q2: 48 cells" in out and "total: 48 cells" in out
    assert len((tmp_path / "q2.txt").read_text().splitlines()) == 48
    assert json.loads((tmp_path / "manifest.json").read_text())["counts"] == {"q2": 48}
    gen_matrix.main(["--config", str(CONFIG), "--dry-run"])
    assert not (tmp_path / "q1.txt").exists()


def test_a_generated_line_has_a_config_hash_that_follows_its_arguments():
    lines = matrix.q2_lines(cfg())
    a = ju.jobline_hash(lines[0])
    b = ju.jobline_hash(lines[0].replace("--seed 0", "--seed 1"))
    assert a and b and a != b
