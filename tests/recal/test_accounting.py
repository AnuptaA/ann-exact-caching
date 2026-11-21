import numpy as np
import pytest

from recal import accounting
from recal.backend import ExactBackend
from recal.stored import StoredAnswers


def test_budget_is_a_share_of_the_stored_bytes():
    assert accounting.budget_bytes(1000, 8, 0.05) == pytest.approx(0.05 * 1000 * 8 * 4)
    assert accounting.budget_bytes(1000, 8, 0.05, coord_bytes=1) == pytest.approx(0.05 * 1000 * 8)


def test_every_structure_is_charged():
    a = accounting.aker_bytes(1000, 100, 10, 0, 16)
    assert a["total"] == pytest.approx(sum(v for k, v in a.items() if k != "total"))
    assert a["representatives"] == 100 * 4 * 16 and a["neighbor_pool"] == 1000 * (4 * 16 + 8)
    q = accounting.qvcache_bytes(500, 16, graph_degree=32, regions=40)
    assert q["graphs"] == 500 * 32 * 4 and q["total"] > q["vectors"]
    assert accounting.qvcache_bytes(500, 16)["graphs"] == 0
    p = accounting.prototype_bytes(900, 300, 90, 12.0, 16)
    assert p["total"] == pytest.approx(sum(v for k, v in p.items() if k != "total"))


def test_scale_pool_brings_a_run_under_budget():
    pool = accounting.scale_pool(1000, 1.2e6, 1.0e6)
    assert 0 < pool < 1000 and pool * 1.2e6 / 1000 <= 1.0e6


def test_stored_answers_match_exact_search_and_reject_unknown_queries():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(300, 6)).astype(np.float32)
    queries = rng.normal(size=(20, 6)).astype(np.float32)
    exact = ExactBackend(data)
    ids, dists = exact.search(queries, 10)
    stored = StoredAnswers(data, queries, ids)
    got_ids, got_d = stored.search(queries[3], 5)
    assert got_ids.tolist() == ids[3, :5].tolist()
    np.testing.assert_allclose(got_d, dists[3, :5], rtol=1e-5)
    with pytest.raises(ValueError):
        stored.search(queries[3], 11)
    with pytest.raises(KeyError):
        stored.search(rng.normal(size=6).astype(np.float32), 5)


def test_pgvector_answer_table_adapter(tmp_path):
    import json
    from recal.stored import from_pgvector
    rng = np.random.default_rng(1)
    data = rng.normal(size=(200, 5)).astype(np.float32)
    queries = rng.normal(size=(30, 5)).astype(np.float32)
    ids, dist = ExactBackend(data).search(queries, 20)
    (tmp_path / "meta.json").write_text(json.dumps({"n": 30, "block": 20}))
    np.savez(tmp_path / "block_0000000.npz", ids=ids[:20].astype(np.int32), dist=dist[:20])
    np.savez(tmp_path / "block_0000001.npz", ids=ids[20:].astype(np.int32), dist=dist[20:])
    stored = from_pgvector(data, queries, tmp_path)
    got, d = stored.search(queries[25], 10)
    assert got.tolist() == ids[25, :10].tolist()
