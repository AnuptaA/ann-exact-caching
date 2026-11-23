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


STATUS = "  Total entries: 40\n  Physical entries: 30\n    Entries: 0.003 MB\n    Pool: 0.100 MB\n"


def test_aker_status_is_parsed_and_recharged_at_stored_precision():
    s = accounting.parse_aker_status(STATUS)
    assert s == {"total_entries": 40, "physical_entries": 30, "pool_mb": 0.1}
    float_bytes = accounting.aker_bytes_from_status(STATUS, 100, 4)
    int8_bytes = accounting.aker_bytes_from_status(STATUS, 100, 1)
    assert float_bytes["pool_vectors"] == int8_bytes["pool_vectors"] == 223
    assert int8_bytes["neighbor_pool"] == 223 * (48 + 100)
    assert int8_bytes["representatives"] == 30 * 100 and int8_bytes["total"] < float_bytes["total"]
    with pytest.raises(ValueError):
        accounting.parse_aker_status("nothing")


def test_aker_pool_for_budget_fits_when_entries_hold_k_vectors():
    d, k, M = 100, 10, 2.0e6
    for coord in (1, 4):
        pool = accounting.aker_pool_for_budget(M, d, k, coord)
        entries = pool // k
        used = (pool * (accounting.AKER_POOL_META_BYTES + coord * d)
                + entries * (accounting.AKER_ENTRY_BYTES + coord * d + accounting.AKER_FILTER_LINK_BYTES))
        assert used <= M and used > 0.98 * M


def test_coord_bytes_changes_every_vector_charge():
    one = accounting.aker_bytes(100, 10, 5, 0, 8, coord_bytes=1)
    four = accounting.aker_bytes(100, 10, 5, 0, 8)
    assert one["neighbor_pool"] == 100 * (8 + 8) and four["neighbor_pool"] == 100 * (32 + 8)
    assert accounting.qvcache_bytes(10, 8, coord_bytes=1)["vectors"] == 10 * 16
    assert accounting.prototype_bytes(10, 0, 2, 5.0, 8, coord_bytes=1)["vectors"] == 10 * 16
