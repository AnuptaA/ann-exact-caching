import json
import sys
from pathlib import Path

import numpy as np
import pytest

from recal import artifacts, prepare
from recal.measure import jobs_util
from simulations.approximate_hits import eval_real_backend as erb

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "experiments" / "matrix.json").read_text())
N_Q, DEPTH = 60, 20


def commands(dataset: str, stage: str) -> list[str]:
    return next(s for s in prepare.diskann_stages(CFG, dataset) if s.name == stage).commands


def test_pq_sizes_come_from_the_configuration():
    assert {d: CFG["prepare"][d]["diskann_pq_bytes"] for d in ("spacev", "triviaqa", "medrag")} == {
        "spacev": 32, "triviaqa": 256, "medrag": 256}
    assert CFG["prepare"]["laion"]["diskann_pq_bytes"] == [64, 128, 256]
    for dataset, pq in (("spacev", 32), ("triviaqa", 256), ("medrag", 256)):
        for stage in ("diskann_index", "diskann_answers"):
            assert all(f"--diskann-pq-bytes {pq}" in c and "--pq-candidates" not in c for c in commands(dataset, stage))


def test_laion_tries_sizes_in_order_in_the_build_stage_only():
    build = commands("laion", "diskann_index")[0]
    assert "--pq-candidates 64 128 256" in build and "--require-recall 0.95" in build
    assert "--max-search-list 256" in build and "--diskann-pq-bytes" not in build
    assert all("--pq-candidates" not in c and "--diskann-pq-bytes" not in c for c in commands("laion", "diskann_answers"))


def test_a_dataset_without_a_setting_gets_the_default():
    cfg = json.loads(json.dumps(CFG))
    del cfg["prepare"]["spacev"]["diskann_pq_bytes"]
    assert prepare.pq_setting(cfg, "spacev") == (prepare.DEFAULT_PQ_BYTES, [])


def test_choose_pq_keeps_the_smallest_size_that_reaches_the_target():
    seen = []

    def measure(pq):
        seen.append(pq)
        return 200, {64: 0.91, 128: 0.96, 256: 0.99}[pq], 1.0

    chosen, trials = erb.choose_pq([256, 64, 128], measure, 0.95)
    assert chosen["pq_bytes"] == 128 and seen == [64, 128] and [t["reached"] for t in trials] == [False, True]
    chosen, trials = erb.choose_pq([64, 128], lambda pq: (256, 0.9, 1.0), 0.95)
    assert chosen is None and len(trials) == 2


class FakeIndex:
    """Stands in for the DiskANN index: recall depends on the PQ size and the search list."""
    truth = None
    recall_by_pq = {64: 0.80, 128: 0.96, 256: 0.99}

    def __init__(self, kind, db, args, out):
        self.pq_bytes = erb.index_pq_bytes(args.index_dir, args.diskann_pq_bytes) if getattr(args, "index_dir", None) \
            else args.diskann_pq_bytes
        self.build_s = 1.0
        if getattr(args, "index_dir", None):
            Path(args.index_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.index_dir) / erb.PQ_FILE).write_text(json.dumps({"pq_bytes": self.pq_bytes}))

    def search(self, q, k, effort):
        ids = np.array(self.truth[:len(q), :k])
        wrong = int(round((1 - self.recall_by_pq.get(self.pq_bytes, 0.99)) * k))
        ids[:, :wrong] = -1
        return ids


@pytest.fixture
def backend(tmp_path, monkeypatch):
    truth = np.tile(np.arange(DEPTH), (N_Q, 1))
    FakeIndex.truth = truth
    monkeypatch.setattr(erb, "load", lambda prep: (np.zeros((5, 2), np.float32), np.zeros((N_Q, 2), np.float32), truth))
    monkeypatch.setattr(erb, "Index", FakeIndex)
    return tmp_path


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["eval_real_backend.py", "--index", "diskann", "--tune-queries", "20", "--k", "10",
                                      "--R", str(DEPTH), "--answer-depth", "10", *argv])
    erb.main()


def test_pick_pq_keeps_the_chosen_index_and_deletes_the_others(backend, monkeypatch):
    out, index = backend / "out", backend / "out" / "diskann"
    run_main(monkeypatch, "--prep", str(backend), "--out-dir", str(out), "--index-dir", str(index), "--build-only",
             "--pq-candidates", "64", "128", "256", "--require-recall", "0.95", "--max-search-list", "64")
    assert erb.index_pq_bytes(index, 0) == 128
    assert not (out / "diskann_pq64").exists() and not (out / "diskann_pq128").exists()
    trials = json.loads((out / "pq_trials.json").read_text())["trials"]
    assert [(t["pq_bytes"], t["reached"]) for t in trials] == [(64, False), (128, True)]
    run_main(monkeypatch, "--prep", str(backend), "--out-dir", str(out), "--index-dir", str(index), "--build-only",
             "--pq-candidates", "64", "128", "256", "--require-recall", "0.95", "--max-search-list", "64")
    assert erb.index_pq_bytes(index, 0) == 128


def test_pick_pq_fails_with_the_table_when_no_size_reaches_the_target(backend, monkeypatch):
    out = backend / "out"
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--prep", str(backend), "--out-dir", str(out), "--index-dir", str(out / "diskann"),
                 "--build-only", "--pq-candidates", "64", "--require-recall", "0.95")
    assert "pq 64: recall 0.8000" in str(exc.value) and not (out / "diskann").exists()


def fixed_run(monkeypatch, backend, pq, target="0.95"):
    out = backend / "out"
    index = out / "diskann"
    index.mkdir(parents=True, exist_ok=True)
    (index / erb.PQ_FILE).write_text(json.dumps({"pq_bytes": pq}))
    run_main(monkeypatch, "--prep", str(backend), "--out-dir", str(out), "--index-dir", str(index),
             "--diskann-pq-bytes", "32", "--fixed", f"{target}:100")
    return out


def test_fixed_fails_below_the_target_and_writes_nothing(backend, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        fixed_run(monkeypatch, backend, 64)
    message = str(exc.value)
    assert "0.8000" in message and "64-byte PQ" in message and "search list 100" in message
    assert not (backend / "out" / "diskann.npz").exists() and not (backend / "out" / "diskann.json").exists()


def test_fixed_succeeds_at_the_target_and_records_pq_search_list_and_recall(backend, monkeypatch):
    out = fixed_run(monkeypatch, backend, 128)
    meta = json.loads((out / "diskann.json").read_text())
    stats = meta["targets"]["0.95"]
    assert meta["pq_bytes"] == 128 and stats["pq_bytes"] == 128 and stats["search_list"] == 100
    assert stats["recall_at_10"] >= 0.95
    assert (out / "diskann.npz").exists()


def test_the_recall_check_boundary():
    assert not erb.below_target(0.95, 0.95) and erb.below_target(0.9499, 0.95)


def write_answers(directory: Path, pq=128, effort=700, recall=0.953):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "diskann.npz", **{"answer_0.95": np.zeros((2, 10), np.int32)})
    (directory / "diskann.json").write_text(json.dumps({"pq_bytes": pq, "targets": {"0.95": {
        "effort": effort, "search_list": effort, "recall_at_10": recall, "recall_answer_at_k": recall}}}))
    return directory / "diskann.npz"


def test_results_carry_the_backend_settings(tmp_path):
    answers = write_answers(tmp_path / "b")
    jobs_util._backend_meta.clear()
    jobs_util.register_backend_file(answers, "0.95")
    result = jobs_util.stamp_result({})
    jobs_util._backend_meta.clear()
    assert result["diskann_backend"] == {"file": str(answers), "target": "0.95", "pq_bytes": 128, "search_list": 700,
                                         "measured_recall_at_10": 0.953}


def test_results_of_older_answer_files_record_nulls(tmp_path):
    answers = tmp_path / "old.npz"
    np.savez(answers, x=np.zeros(1))
    jobs_util._backend_meta.clear()
    jobs_util.register_backend_file(answers, "0.95")
    meta = dict(jobs_util._backend_meta)
    jobs_util._backend_meta.clear()
    assert meta["pq_bytes"] is None and meta["search_list"] is None


def test_the_manifest_entry_records_pq_search_list_and_recall(tmp_path):
    answers = write_answers(tmp_path / "results" / "b")
    index = tmp_path / "results" / "b" / "diskann"
    index.mkdir()
    (index / "ann_disk.index").write_bytes(b"x")
    (index / erb.PQ_FILE).write_text(json.dumps({"pq_bytes": 128}))
    paths = [str(answers.relative_to(tmp_path)), str(answers.with_suffix(".json").relative_to(tmp_path))]
    entry = artifacts.describe("diskann_answers", paths, "cmd", "abc", tmp_path)
    info = entry["backend"][paths[1]]
    assert info["pq_bytes"] == 128 and info["targets"]["0.95"] == {"search_list": 700, "recall_at_10": 0.953}
    entry = artifacts.describe("diskann_index", [str(index.relative_to(tmp_path))], "cmd", "abc", tmp_path)
    assert list(entry["backend"].values()) == [{"pq_bytes": 128}]


def test_the_answer_json_is_part_of_the_answer_artifact():
    paths = artifacts.local_paths(CFG, "laion")["diskann_answers"]
    assert paths == ["results/backends/laion-prompts/diskann.json", "results/backends/laion-prompts/diskann.npz"]


def test_laion_is_in_the_backend_selection():
    from simulations.approximate_hits import select_backend
    assert select_backend.CANDIDATES["laion-prompts"] == ["laion"]
