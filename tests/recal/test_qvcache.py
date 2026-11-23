import json
import sys
import types

import numpy as np
import pytest

from recal import accounting, cells, qvcache_driver as qd, sim
from recal import records as rec
from recal.backend import ExactBackend
from recal.qvcache_backends import StoredBackend
from tests.recal.test_system import make_stream, make_world


class FakeQVCache:
    """The artifact's interface: a hit when the query was seen, else a backend search and an insert through
    fetch_vectors_by_ids, up to memory_index_max_points vectors."""

    instances = []

    def __init__(self, **kw):
        self.kw, self.backend = kw, kw["backend"]
        self.seen, self.vectors = {}, set()
        FakeQVCache.instances.append(self)

    def set_search_strategy(self, s):
        self.strategy = s

    def search(self, q, K):
        key = (q.tobytes(), K)
        if key in self.seen:
            tags = self.seen[key]
            return True, tags, np.zeros(K, np.float32)
        tags, dists = self.backend.search(q, K)
        self.seen[key] = tags
        take = [int(t) for t in tags if int(t) not in self.vectors]
        if take:
            self.backend.fetch_vectors_by_ids(take)
            self.vectors.update(take)
            while len(self.vectors) > self.kw["memory_index_max_points"]:
                self.vectors.pop()
        return False, tags, dists

    def get_number_of_vectors_in_memory_index(self):
        return len(self.vectors)

    def get_number_of_active_pca_regions(self):
        return max(1, len(self.seen) // 8)


def fake_module():
    mod = types.SimpleNamespace(QVCache=FakeQVCache, SearchStrategy=types.SimpleNamespace(SEQUENTIAL_LRU_ADAPTIVE="adaptive"))
    return mod


@pytest.fixture
def prep(tmp_path):
    rng, data, centers = make_world(seed=5)
    stream = make_stream(rng, centers, 3000, repeat_share=0.5)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq.astype(np.float32))
    np.save(tmp_path / "top500.npy", ExactBackend(data).search(uniq.astype(np.float32), 100)[0].astype(np.int32))
    np.savez(tmp_path / "streams.npz", random=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"), "q": str(tmp_path / "q.npy")}))
    return tmp_path


def args_for(prep, tmp_path, *extra):
    parser = sim.parser()
    parser.add_argument("--deviation-factor", type=float, default=None)
    parser.add_argument("--pca-points", type=int, default=10_000_000)
    parser.add_argument("--qvcache-threads", type=int, default=None)
    return parser.parse_args(["--prep", str(prep), "--stream", "random", "--k", "5", "--backend", "exact", "--budget", "0.6",
                              "--out", str(tmp_path / "o"), *extra])


def test_adapters_return_squared_distances_and_count_rereads():
    data = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    backend = StoredBackend(ExactBackend(data))
    q = data[3] + 0.01
    tags, dists = backend.search(q, 5)
    exact_ids, exact_d = ExactBackend(data).search(q, 5)
    assert tags.dtype == np.uint32 and tags.tolist() == exact_ids.tolist()
    np.testing.assert_allclose(dists, exact_d ** 2, rtol=1e-5)
    vectors = backend.fetch_vectors_by_ids([1, 2, 3])
    assert len(vectors) == 3 and backend.rereads == 3 and backend.searches == 1 and backend.stats()["miss"]["depth_sum"] == 5


def test_default_deviation_follows_the_paper():
    assert qd.default_deviation(1) == 0.25 and qd.default_deviation(4) == 0.075


def test_capacity_makes_a_full_cache_cost_m_with_points_split_over_the_mini_indexes():
    for coord in (1, 4):
        cap = accounting.qvcache_capacity_for_budget(5e6, 100, coord, 64, 16, 500, 4)
        full = accounting.qvcache_full_capacity_bytes(cap, 100, coord, 64, 16, 500)
        assert cap % 4 == 0 and full <= 5e6 and full > 5e6 - 4 * (coord * 100 + 4 + 256) - 1
    assert accounting.capacity_for_budget("qvcache", 5e6, 100, 1, R=64, pca_dim=16, regions=500) == \
        accounting.qvcache_capacity_for_budget(5e6, 100, 1, 64, 16, 500)


def test_cell_runs_the_artifact_through_the_harness(prep, tmp_path):
    FakeQVCache.instances.clear()
    a = args_for(prep, tmp_path)
    result, records = qd.run_cell(a, tmp_path / "o", fake_module())
    assert result["history_len"] == int(0.3 * 3000) == result["first_scored"] and result["n_scored"] == 2100
    q = result["qvcache"]
    assert q["repo"] == "https://github.com/qvcache/qvcache-vldb26" and len(q["commit"]) == 40
    assert q["deviation_factor"] == 0.075 and q["settings"]["number_of_mini_indexes"] == 4 and q["differences"]
    cal = result["calibration"]
    assert abs(cal["bytes_at_full_capacity"] - result["M"]) <= 0.01 * result["M"] and cal["regions_on_history"] >= 1
    assert result["configured_capacity"] == cal["configured_capacity"] and result["bytes_used_end"] <= result["M"]
    assert result["within_M"] and result["summary"]["queries"] == 2100
    cols = records.arrays()
    assert len(cols["index"]) == 2100 and cols["index"][0] == 900
    hits, misses = cols["path"] == rec.MAIN, cols["path"] == rec.MISS
    assert hits.sum() + misses.sum() == 2100 and hits.sum() > 0
    assert (cols["backend_kind"][misses] == rec.BACKEND_MISS).all() and cols["rereads"].sum() > 0
    assert (cols["recall"][hits] == 1.0).all() or cols["recall"][hits].min() >= 0.0
    assert FakeQVCache.instances[-1].kw["deviation_factor"] == 0.075


def test_deviation_override_and_thread_setting_reach_the_artifact(prep, tmp_path):
    FakeQVCache.instances.clear()
    a = args_for(prep, tmp_path, "--deviation-factor", "0.15", "--qvcache-threads", "4")
    result, _ = qd.run_cell(a, tmp_path / "o", fake_module())
    kw = FakeQVCache.instances[-1].kw
    assert kw["deviation_factor"] == 0.15 and kw["search_threads"] == kw["build_threads"] == 4
    assert result["qvcache"]["threads"] == 4


def test_pca_training_file_is_a_diskann_bin(tmp_path):
    db = np.random.default_rng(0).normal(size=(500, 6)).astype(np.float32)
    path = qd.pca_base(db, tmp_path, 200)
    raw = path.read_bytes()
    n, d = np.frombuffer(raw[:8], dtype=np.int32)
    assert (n, d) == (200, 6) and len(raw) == 8 + 200 * 6 * 4


def test_cells_main_writes_qvcache_result_and_records(prep, tmp_path, monkeypatch):
    monkeypatch.setattr(qd, "load_module", fake_module)
    out = tmp_path / "res"
    monkeypatch.setattr(sys, "argv", ["x", "--method", "qvcache", "--prep", str(prep), "--stream", "random", "--k", "5",
                                      "--backend", "exact", "--budget", "0.6", "--out", str(out)])
    cells.main()
    data = json.loads((out / "qvcache.json").read_text())
    assert data["method"] == "qvcache" and data["records_file"] == "qvcache_records.npz" and "config_hash" in data
    assert (out / "qvcache_records.npz").exists() and not list(out.glob(".*tmp"))


def test_simulator_stays_available_only_as_a_cross_check():
    assert cells.SIM_METHOD["qvcache_sim"] == "qvcache" and "qvcache" not in cells.SIM_METHOD


def test_validation_helpers_parse_the_artifact_output_and_scripts():
    from scripts import validate_qvcache as v
    text = 'noise\n{"event": "window_metrics", "hit_ratio": 0.5, "recall_all": 0.9, "recall_cache_hits": 0.95}\n' \
           '{"event": "window_metrics", "hit_ratio": 0.7, "recall_all": 0.88, "recall_cache_hits": null}\n'
    m = v.parse_window_metrics(text)
    assert m["windows"] == 2 and m["hit_ratio"] == pytest.approx(0.6) and m["recall_cache_hits"] == 0.95
    with pytest.raises(ValueError):
        v.parse_window_metrics("nothing")
    script = 'DATASET="siftsmall"\nR=64\nMEMORY_L=32  \nALPHA=1.2\nDEVIATION_FACTOR=0.25 # comment\nK=10\n'
    vars_ = v.script_variables(script)
    assert vars_["DATASET"] == "siftsmall" and vars_["MEMORY_L"] == 32 and vars_["ALPHA"] == 1.2
    assert vars_["DEVIATION_FACTOR"] == 0.25 and vars_["K"] == 10
    ok = v.compare({"hit_ratio": 0.6, "recall_all": 0.9}, {"hit_ratio": 0.62, "recall_all": 0.89}, 0.03)
    bad = v.compare({"hit_ratio": 0.6, "recall_all": 0.9}, {"hit_ratio": 0.7, "recall_all": 0.9}, 0.03)
    assert ok["ok"] and not bad["ok"]


def test_adapter_stage_runs_on_the_script_data_with_a_fake_artifact(tmp_path):
    from scripts import validate_qvcache as v
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 8)).astype(np.float32)
    queries = np.repeat(base[:40] + 0.01, 3, axis=0).astype(np.float32)
    truth = ExactBackend(base).search(queries, 10)[0].astype(np.uint32)
    data = tmp_path / "data" / "tiny"
    data.mkdir(parents=True)
    (data / "tiny_base.bin").write_bytes(b"x")
    mod = fake_module()
    mod.load_aligned_binary_data = lambda path: (base if "base" in path else queries, 8, 8)
    mod.load_ground_truth_data = lambda path: (truth, None)
    out = v.run_adapter(tmp_path, {"DATASET": "tiny", "K": 10, "MEMORY_INDEX_MAX_POINTS": 400}, qvc=mod)
    assert out["queries"] == 120 and out["hit_ratio"] == pytest.approx(2 / 3, abs=0.02) and out["recall_all"] > 0.9
