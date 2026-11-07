"""Backend tests: fake backends and a fake connection, plus optional real PostgreSQL."""

import os

import numpy as np
import pytest

from harness.backend import answer_table, build_index, ef_tuning, pgvector

RNG = np.random.default_rng(0)


class ExactBackend:
    """Exact search whose quality degrades below ef_search 50 (swaps in far ids)."""
    name = "fake"

    def __init__(self, base):
        self.base = base

    def search(self, queries, list_size, ef_search):
        d = ((queries[:, None, :] - self.base[None]) ** 2).sum(2)
        order = np.argsort(d, axis=1)[:, :list_size].copy()
        if ef_search < 50:
            order[:, : max(1, (50 - ef_search) // 8)] = len(self.base) - 1
        return order


class FakeCursor:
    def __init__(self, log, plan):
        self.log, self.plan, self._rows = log, plan, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def execute(self, sql, params=None):
        self.log.append(sql)
        self._rows = [(line,) for line in self.plan] if sql.startswith("EXPLAIN") else [(1,), (2,)]

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, plan):
        self.log, self.plan = [], plan

    def cursor(self):
        return FakeCursor(self.log, self.plan)


def test_expected_result_count_is_capped_by_ef_search():
    assert pgvector.expected_result_count(40, 50, 1000) == 40
    assert pgvector.expected_result_count(100, 50, 1000) == 50
    assert pgvector.expected_result_count(100, 50, 20) == 20


def test_session_disables_seqscan_and_iterative_scans_and_checks_the_plan():
    conn = FakeConn(["Limit", "  ->  Index Scan using base_embedding_hnsw on base"])
    be = pgvector.PgvectorBackend(conn)
    out = be.search(np.zeros((2, 3), dtype=np.float32), 2, 40)
    assert out.tolist() == [[1, 2], [1, 2]]
    first = conn.log[:4]
    assert first[0] == "SET enable_seqscan = off" and first[1] == "SET hnsw.iterative_scan = off"
    assert first[2] == "SET hnsw.ef_search = 40" and first[3].startswith("EXPLAIN")
    with pytest.raises(ValueError):
        be.search(np.zeros((1, 3), dtype=np.float32), 2, pgvector.EF_SEARCH_MAX + 1)
    seq = pgvector.PgvectorBackend(FakeConn(["Limit", "  ->  Seq Scan on base"]))
    with pytest.raises(RuntimeError):
        seq.search(np.zeros((1, 3), dtype=np.float32), 2, 40)


def test_build_memory_sizing_and_notice_guard():
    assert build_index.hnsw_graph_bytes(10_000_000, 768, 32) == pytest.approx(41e9, rel=0.01)
    assert build_index.maintenance_work_mem_mb(10_000_000, 768, 32) > 41_000
    sql = build_index.create_index_sql("base", "embedding", "idx", 16, 128, 900, 4)
    assert "m = 16, ef_construction = 128" in sql[2] and "900MB" in sql[0]
    watcher = build_index.NoticeWatcher()
    watcher("hnsw graph no longer fits into maintenance_work_mem after 1000 tuples")
    with pytest.raises(RuntimeError):
        watcher.check()
    assert watcher.check(fail=False) is False
    assert build_index.BUILD_PARAMS["sphere"] == {"m": 32, "ef_construction": 256}
    assert build_index.BUILD_PARAMS["spacev"] == {"m": 16, "ef_construction": 128}


def test_sweep_choice_and_list_size_rule():
    base = RNG.normal(size=(200, 4)).astype(np.float32)
    queries = RNG.normal(size=(30, 4)).astype(np.float32)
    from workloads.q1.groundtruth import exact_topk_chunk
    _, dist = exact_topk_chunk(queries, base, k=10)
    rows = ef_tuning.sweep(ExactBackend(base), queries, base, dist[:, 9],
                           grid=(10, 20, 30, 50, 75, 100))
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls) and recalls[0] < 1.0 and recalls[-1] == 1.0
    assert ef_tuning.choose_ef(rows, list_size=10)["ef_search"] == 50
    bound = ef_tuning.choose_ef(rows, list_size=75)
    assert bound["ef_search"] == 75 and bound["list_size_binds"]
    with pytest.raises(ValueError):
        ef_tuning.choose_ef(rows[:2], list_size=10)


def test_answer_table_resumes_shards_and_rejects_short_answers(tmp_path):
    base = RNG.normal(size=(120, 4)).astype(np.float32)
    vectors = RNG.normal(size=(25, 4)).astype(np.float32)
    be = ExactBackend(base)
    for shard in ((0, 2), (1, 2)):
        answer_table.build_answer_table(be, vectors, base, tmp_path / "t", 20, 100,
                                        block=10, shard=shard)
    ids, dist = answer_table.load_answer_table(tmp_path / "t")
    assert ids.shape == (25, 20) and ids.dtype == np.int32 and (np.diff(dist, axis=1) >= 0).all()
    (tmp_path / "t" / "block_0000001.npz").unlink()
    answer_table.build_answer_table(be, vectors, base, tmp_path / "t", 20, 100, block=10)
    assert np.array_equal(answer_table.load_answer_table(tmp_path / "t")[0], ids)
    with pytest.raises(ValueError):
        answer_table.build_answer_table(be, vectors, base, tmp_path / "t", 19, 100, block=10)

    class Short(ExactBackend):
        def search(self, q, k, ef):
            out = super().search(q, k, ef)
            out[:, -1] = -1
            return out
    with pytest.raises(ValueError):
        answer_table.build_answer_table(Short(base), vectors, base, tmp_path / "s", 20, 100)


@pytest.mark.skipif(not os.environ.get("PGVECTOR_TEST_DSN"), reason="needs PostgreSQL + pgvector")
def test_real_pgvector_behaviors():
    """ef_search caps results, has an upper limit, top-50's first 10 equal top-10."""
    import psycopg
    rng = np.random.default_rng(1)
    base = rng.normal(size=(3000, 16)).astype(np.float32)
    with psycopg.connect(os.environ["PGVECTOR_TEST_DSN"], autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS q1_test")
            cur.execute("CREATE TABLE q1_test (id bigint PRIMARY KEY, embedding vector(16))")
            for i, v in enumerate(base):
                cur.execute("INSERT INTO q1_test VALUES (%s, %s::vector)", (i, pgvector.vector_literal(v)))
            cur.execute("CREATE INDEX q1_test_embedding_hnsw ON q1_test USING hnsw "
                        "(embedding vector_l2_ops) WITH (m = 16, ef_construction = 64)")
        be = pgvector.PgvectorBackend(conn, table="q1_test")
        q = rng.normal(size=(1, 16)).astype(np.float32)
        out = be.search(q, 50, 20)
        assert (out >= 0).sum() == pgvector.expected_result_count(20, 50, len(base))
        top50 = be.search(q, 50, 60)[0]
        top10 = be.search(q, 10, 60)[0]
        assert top50[:10].tolist() == top10.tolist()
        with pytest.raises(Exception):
            be.search(q, 10, pgvector.EF_SEARCH_MAX + 1)


def test_copy_lines_format():
    from harness.backend import load
    rows = np.array([[1.0, 2.5], [3.0, -4.0]], dtype=np.float32)
    assert list(load.copy_lines(rows, 7)) == ["7\t[1,2.5]\n", "8\t[3,-4]\n"]
