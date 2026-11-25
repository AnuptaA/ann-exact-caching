import json
from pathlib import Path

import numpy as np
import pytest

from recal import matrix, prepare
from scripts import data_fetch, laion_data

ROOT = Path(__file__).resolve().parents[2]
SHARDS, PER_SHARD, REQUESTS, QUERIES = 3, 10, 40, 25
MEDRAG = {"q1_budgets": [0.0001, 0.0003, 0.001], "standin_5pct": 0.0003}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(data_fetch, "LAION_SHARDS", SHARDS)
    monkeypatch.setattr(data_fetch, "LAION_ROWS", SHARDS * PER_SHARD)
    monkeypatch.setattr(data_fetch, "LAION_REQUESTS", REQUESTS)
    monkeypatch.setattr(data_fetch, "LAION_QUERIES", QUERIES)
    monkeypatch.setattr(laion_data, "TOP_DEPTH", 10)


def unit(rng, rows):
    x = rng.normal(size=(rows, laion_data.DIM)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make_folder(root: Path, seed: int = 0) -> Path:
    """A LAION-like folder: float16 shards, request files, shuffled stream, exact top-k."""
    rng = np.random.default_rng(seed)
    folder = root / "laion"
    (folder / "shards").mkdir(parents=True)
    corpus = unit(rng, SHARDS * PER_SHARD)
    ids = rng.permutation(10**6)[:len(corpus)].astype(np.int64)
    for i in range(SHARDS):
        sel = slice(i * PER_SHARD, (i + 1) * PER_SHARD)
        np.savez(folder / "shards" / data_fetch.laion_shard_name(i), embeddings=corpus[sel].astype(np.float16),
                 passage_ids=ids[sel])
    queries = unit(rng, QUERIES)
    stream = rng.permutation(np.concatenate([np.arange(QUERIES), rng.integers(0, QUERIES, REQUESTS - QUERIES)]))
    np.save(folder / "q.npy", queries)
    np.save(folder / "rc-diffusiondb-requests.npy", queries[stream])
    np.savez(folder / "streams.npz", trace=stream.astype(np.int64))
    top = np.argsort(-(queries @ corpus.astype(np.float16).astype(np.float32).T), axis=1)[:, :laion_data.TOP_DEPTH]
    np.save(folder / "top128.npy", top.astype(np.int32))
    np.save(folder / "row_to_passage_id.npy", ids)
    for name in ("requests.json", "build.json", "checks.json"):
        (folder / name).write_text("{}")
    np.save(folder / "shuffle_perm.npy", np.arange(REQUESTS))
    return folder


def test_verify_accepts_a_consistent_folder(small, tmp_path, capsys):
    folder = make_folder(tmp_path)
    assert laion_data.verify(folder) == 0
    assert "ids and norms consistent" in capsys.readouterr().out


def test_verify_flags_each_kind_of_damage(small, tmp_path, capsys):
    folder = make_folder(tmp_path)
    ids = np.load(folder / "row_to_passage_id.npy")
    ids[PER_SHARD + 2] += 1
    np.save(folder / "row_to_passage_id.npy", ids)
    assert laion_data.verify(folder) == 1
    assert "passage ids differ" in capsys.readouterr().out
    (folder / "shards" / data_fetch.laion_shard_name(1)).unlink()
    assert laion_data.verify(folder) == 1
    assert "missing 1 shards" in capsys.readouterr().out


def test_verify_flags_stream_and_request_problems(small, tmp_path, capsys):
    folder = make_folder(tmp_path)
    stream = np.load(folder / "streams.npz")["trace"]
    stream[0] = QUERIES + 5
    np.savez(folder / "streams.npz", trace=stream)
    assert laion_data.verify(folder) == 1
    assert "outside" in capsys.readouterr().out


def test_build_writes_float32_in_shard_order_and_is_rerunnable(small, tmp_path, capsys):
    folder = make_folder(tmp_path)
    out = tmp_path / "aker" / "laion-ip-docs.npy"
    assert laion_data.main(["build", "--dir", str(folder), "--out", str(out)]) == 0
    base = np.load(out, mmap_mode="r")
    expected = np.concatenate([laion_data.shard_vectors(p) for p in laion_data.shard_paths(folder)])
    assert base.dtype == np.float32 and np.array_equal(base, expected.astype(np.float32))
    info = json.loads(out.with_suffix(".json").read_text())
    assert info["rows"] == len(base) and len(info["sha256"]) == 64
    assert (tmp_path / "aker" / "laion-requests.npy").is_symlink()
    assert (tmp_path / "aker" / "laion-ip-docs-f32.npy").resolve() == out.resolve()
    capsys.readouterr()
    assert laion_data.main(["build", "--dir", str(folder), "--out", str(out)]) == 0
    assert "already built" in capsys.readouterr().out
    assert not out.with_name(out.name + ".tmp").exists()


def test_truth_check_matches_prep_queries_to_the_shipped_top128(small, tmp_path):
    folder = make_folder(tmp_path)
    queries = np.load(folder / "q.npy")
    base = np.concatenate([laion_data.shard_vectors(p) for p in laion_data.shard_paths(folder)]).astype(np.float32)
    order = np.random.default_rng(1).permutation(QUERIES)
    prep = tmp_path / "prep"
    prep.mkdir()
    np.save(prep / "q.npy", queries[order])
    top = np.argsort(-(queries[order] @ base.T), axis=1).astype(np.int32)
    np.save(prep / "top500.npy", top)
    assert laion_data.check_truth(prep, folder, count=10) == []
    np.save(prep / "top500.npy", np.flip(top, axis=1).copy())
    assert laion_data.check_truth(prep, folder, count=10)


def test_fetch_lists_prompts_and_shards_from_their_own_folders(small, tmp_path, capsys):
    prompts = [{"path": n, "id": n, "size": None} for n in data_fetch.LAION_PROMPTS]
    shards = [{"path": f"data_vectors/{data_fetch.laion_shard_name(i)}", "id": str(i), "size": 5}
              for i in range(SHARDS)]
    listing = tmp_path / "listing.json"
    listing.write_text(json.dumps({"laion_prompts": prompts, "laion": shards}))
    code = data_fetch.main(["--laion", "--dry-run", "--root", str(tmp_path), "--listing", str(listing)])
    out = capsys.readouterr().out
    assert code == 0 and "missing" not in out.split("needs")[0]
    assert "datasets/laion/shards/shard_002.npz" in out and "datasets/laion/README_prompts.md" in out
    listing.write_text(json.dumps({"laion_prompts": prompts, "laion": shards[:-1]}))
    code = data_fetch.main(["--laion", "--dry-run", "--root", str(tmp_path), "--listing", str(listing)])
    assert code == 3 and "RECAL_DRIVE_LAION)" in capsys.readouterr().out


def test_no_corpus_skips_the_shards_and_all_leaves_laion_out(small):
    assert {i.folder for i in data_fetch.items_for("laion", corpus=False)} == {"laion_prompts"}
    assert {i.folder for i in data_fetch.items_for("laion")} == {"laion_prompts", "laion"}
    assert "laion" not in data_fetch.selected_modes(data_fetch.parse_args(["--all"]))


def test_missing_variables_name_both_folders(tmp_path):
    assert data_fetch.missing_variables(["laion_prompts", "laion"], {}) == [
        "RECAL_DRIVE_LAION_PROMPTS", "RECAL_DRIVE_LAION"]
    assert "RECAL_DRIVE_LAION_PROMPTS=" in (ROOT / "env.example").read_text()


def laion_cfg():
    return json.loads((ROOT / "experiments" / "matrix.json").read_text())


def test_registry_knows_the_workload():
    from simulations.approximate_hits import run_e2e_oracle_trace, workloads
    assert run_e2e_oracle_trace.WORKLOADS["laion-prompts"]["base"] == "laion-ip-docs.npy"
    assert workloads.get("laion-prompts").prep == "laion-prompts"
    w = laion_cfg()["workloads"]["laion"]
    assert (w["n_vectors"], w["dim"], w["replay_workload"]) == (9_646_707, 512, "laion-prompts")


def test_prepare_stages_build_the_base_before_prep_and_check_the_truth():
    names = [s.name for s in prepare.stages(laion_cfg(), "laion")]
    assert names[:4] == ["fetch", "base", "prep", "diskann_index"]
    prep = prepare.prep_stage(laion_cfg(), "laion")
    assert "--R 500" in prep.commands[0] and any("laion_data check-truth" in c for c in prep.commands)
    assert prepare.laion_base_stage(laion_cfg(), "laion").commands[0].endswith("datasets/aker/laion-ip-docs.npy")
    assert prepare.DISK_NEED_GB["laion"] == 146.2


def test_laion_config_replaces_tripclick_and_example_keeps_it():
    text = (ROOT / "experiments" / "matrix.json").read_text()
    assert "tripclick" not in text
    q4 = laion_cfg()["q4"]
    assert "laion" in q4["grid_workloads"] and "laion" in q4["k_workloads"] and "laion" in q4["update_workloads"]
    example = json.loads((ROOT / "experiments" / "matrix.tripclick-legacy.json").read_text())
    assert "tripclick" in example["workloads"]
    counts = [{k: len(v) for k, v in matrix.cell_jobs(c, MEDRAG)[0].items()} for c in (example, laion_cfg())]
    assert counts[0] == counts[1] and counts[0]["q1"] > 0
    laion_lines = [j.line for j in matrix.cell_jobs(laion_cfg(), MEDRAG)[0]["q1"]]
    assert any("/laion/" in x for x in laion_lines) and not any("/tripclick/" in x for x in laion_lines)


def test_a_question_can_name_its_own_workloads():
    cfg = laion_cfg()
    full = len(matrix.cell_jobs(cfg, MEDRAG)[0]["q1"])
    cfg["q1"]["workloads"] = ["laion"]
    only = matrix.cell_jobs(cfg, MEDRAG)[0]["q1"]
    assert 0 < len(only) < full and all("/laion/" in j.line for j in only)
