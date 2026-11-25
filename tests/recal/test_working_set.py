import json
from pathlib import Path

import numpy as np

from recal import matrix, working_set as ws
from recal.backend import ExactBackend
from recal.params import Params
from scripts import medrag_working_set

ROOT = Path(__file__).resolve().parents[2]


def test_working_set_is_the_smallest_budget_after_which_the_rate_grows_under_a_point():
    budgets = [1, 2, 4, 8, 16, 32]
    assert ws.working_set(budgets, [0.1, 0.3, 0.6, 0.8, 0.805, 0.806]) == 8
    assert ws.working_set(budgets, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) == 32
    assert ws.working_set(budgets, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) == 1
    assert ws.working_set(budgets, [0.1, 0.1, 0.6, 0.6, 0.8, 0.8]) == 16


def test_budgets_file_holds_the_q1_budgets_and_the_stand_in():
    out = ws.budgets_file([1.0, 2.0], [0.1, 0.2], 0.01)
    assert out["q1_budgets"] == [0.001, 0.003, 0.01] and out["standin_5pct"] == 0.003
    assert out["sweep"][1] == {"budget": 2.0, "history_hit_rate": 0.2}


def test_geometric_sweep_includes_both_ends():
    assert ws.geometric_sweep(1.0, 8.0, 2.0) == [1.0, 2.0, 4.0, 8.0]


def synthetic(n_questions=12, n=2400, d=12, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(n_questions, d))
    data = (centers[rng.integers(0, n_questions, 5000)] + rng.normal(size=(5000, d))).astype(np.float32)
    which = rng.integers(0, n_questions, n)
    vectors = (centers[which] + 0.05 * rng.normal(size=(n, d))).astype(np.float32)
    return data, vectors


def test_the_sweep_finds_where_the_hit_rate_stops_growing_on_a_known_stream(tmp_path):
    data, vectors = synthetic()
    ks = np.full(len(vectors), 5, dtype=np.int32)
    params = Params(K=5, k_max=20, H=3000, cold_start=1500, min_ess=30.0, seed=1)
    budgets = ws.geometric_sweep(0.01, 0.64, 2.0)
    rates = ws.sweep_rates(ExactBackend(data), vectors, ks, params, len(data), data.shape[1], budgets, 4.0)
    w = ws.working_set(budgets, rates)
    assert rates[-1] > 0.5 and rates[0] < rates[-1]
    assert max(rates) - rates[budgets.index(w)] < 0.03
    assert w < budgets[-1]
    assert all(rates[i + 1] - rates[i] < 0.01 for i in range(budgets.index(w), len(budgets) - 1))


def test_script_writes_the_budget_file_from_a_prep_folder(tmp_path):
    data, vectors = synthetic()
    uniq, inverse = np.unique(vectors, axis=0, return_inverse=True)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq)
    np.savez(tmp_path / "streams.npz", trace=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"),
                                                    "q": str(tmp_path / "q.npy")}))
    out = tmp_path / "medrag_budgets.json"
    medrag_working_set.main(["--prep", str(tmp_path), "--backend", "exact", "--low", "0.02", "--high", "0.32",
                             "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["q1_budgets"][2] == data["working_set_fraction"] and len(data["sweep"]) == 5


def test_generator_refuses_medrag_cells_without_the_file_and_uses_it_when_present(tmp_path):
    cfg = json.loads((ROOT / "experiments" / "matrix.json").read_text())
    cfg["medrag_budgets_file"] = str(tmp_path / "missing.json")
    jobs, skipped = matrix.cell_jobs(cfg)
    assert skipped == ["medrag"] and not [j for v in jobs.values() for j in v if j.workload == "medrag"]
    (tmp_path / "missing.json").write_text(json.dumps(ws.budgets_file([1.0], [0.5], 0.002)))
    jobs, skipped = matrix.cell_jobs(cfg)
    medrag_q1 = [j.line for j in jobs["q1"] if j.workload == "medrag"]
    assert not skipped and len(medrag_q1) == 48
    assert {b for b in (0.0002, 0.0006, 0.002) if any(f"--budget {b:g} " in x for x in medrag_q1)} == {0.0002, 0.0006, 0.002}
