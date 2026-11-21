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
    assert build_index.hnsw_graph_bytes(10_000_000, 768, 32) == pytest.approx(43.04e9, rel=0.01)
    # measured bytes per tuple (pgvector 0.8.0 notices): dim 100/m 16, 128/16, 384/16, 768/32
    for dim, m, measured in ((100, 16, 1115), (128, 16, 1228), (384, 16, 2250), (768, 32, 4303)):
        assert build_index.hnsw_graph_bytes(1, dim, m) == pytest.approx(measured, rel=0.01)
    assert build_index.maintenance_work_mem_mb(10_000_000, 768, 32) > 41_000
    sql = build_index.create_index_sql("base", "embedding", "idx", 16, 128, 900, 4)
    assert "m = 16, ef_construction = 128" in sql[3] and "900MB" in sql[1] and sql[0] == "SET statement_timeout = 0"
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


def test_dsn_gets_keepalives_unless_it_sets_its_own():
    dsn = pgvector.with_keepalives("host=127.0.0.1 port=54320 user=postgres", "q1_esci")
    assert "dbname=q1_esci" in dsn and "keepalives_idle=60" in dsn and "connect_timeout=30" in dsn
    own = pgvector.with_keepalives("host=h keepalives_idle=5")
    assert "keepalives_idle=5" in own and own.count("keepalives_idle") == 1


def test_answer_table_requires_ef_search_between_list_size_and_the_pgvector_limit(tmp_path):
    base = RNG.normal(size=(120, 4)).astype(np.float32)
    vectors = RNG.normal(size=(5, 4)).astype(np.float32)
    for ef in (19, 1001):
        with pytest.raises(ValueError, match="ef_search"):
            answer_table.build_answer_table(ExactBackend(base), vectors, base, tmp_path / "t", 20, ef)
    assert not (tmp_path / "t").exists()


def test_validate_blocks_catches_each_kind_of_corruption(tmp_path):
    base = RNG.normal(size=(120, 4)).astype(np.float32)
    vectors = RNG.normal(size=(25, 4)).astype(np.float32)
    table = tmp_path / "t"
    answer_table.build_answer_table(ExactBackend(base), vectors, base, table, 20, 100, block=10)
    answer_table.validate_blocks(table, 120, 20, 25)
    with pytest.raises(ValueError, match="meta"):
        answer_table.validate_blocks(table, 120, 20, 26)
    good = dict(np.load(table / "block_0000001.npz"))

    def rewrite(**changes):
        np.savez(table / "block_0000001.npz", **{**good, **changes})

    for changes, message in (({"ids": np.where(good["ids"] == good["ids"][0, 0], 120, good["ids"])}, "outside"),
                             ({"ids": np.concatenate([good["ids"][:, :1], good["ids"][:, :1],
                                                      good["ids"][:, 2:]], axis=1)}, "repeats"),
                             ({"dist": good["dist"][:, ::-1].copy()}, "sorted"),
                             ({"dist": np.where(np.arange(20) == 3, np.nan, good["dist"]).astype(np.float32)}, "finite"),
                             ({"ids": good["ids"][:-1], "dist": good["dist"][:-1]}, "shape")):
        rewrite(**changes)
        with pytest.raises(ValueError, match=message):
            answer_table.validate_blocks(table, 120, 20, 25)
    rewrite()
    (table / "block_0000002.npz").unlink()
    with pytest.raises(ValueError, match="blocks present"):
        answer_table.validate_blocks(table, 120, 20, 25)


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


def _pg_table(conn, base, name="t_q1"):
    from harness.backend import load
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"DROP TABLE IF EXISTS {name}")
        cur.execute(f"CREATE TABLE {name} (id bigint PRIMARY KEY, embedding vector({base.shape[1]}))")
        with cur.copy(f"COPY {name} (id, embedding) FROM STDIN") as cp:
            for line in load.copy_lines(base, 0):
                cp.write(line)


@pytest.mark.skipif(not os.environ.get("PGVECTOR_TEST_DSN"), reason="needs PostgreSQL + pgvector")
def test_real_pgvector_l2_support_proc_is_squared_and_version_is_080():
    import psycopg
    with psycopg.connect(os.environ["PGVECTOR_TEST_DSN"], autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SELECT p.proname FROM pg_amproc ap JOIN pg_opclass oc ON ap.amprocfamily = oc.opcfamily "
                    "JOIN pg_proc p ON p.oid = ap.amproc JOIN pg_am am ON am.oid = oc.opcmethod "
                    "WHERE am.amname = 'hnsw' AND oc.opcname = 'vector_l2_ops'")
        assert [r[0] for r in cur.fetchall()] == ["vector_l2_squared_distance"]
        assert pgvector.extension_version(conn) == pgvector.EXPECTED_VERSION


@pytest.mark.skipif(not os.environ.get("PGVECTOR_TEST_DSN"), reason="needs PostgreSQL + pgvector")
def test_real_small_ef_search_is_not_exact_and_sized_build_does_not_warn():
    import psycopg
    from workloads.q1.groundtruth import exact_topk_chunk
    rng = np.random.default_rng(0)
    base = rng.normal(size=(6000, 32)).astype(np.float32)
    with psycopg.connect(os.environ["PGVECTOR_TEST_DSN"], autocommit=True) as conn:
        _pg_table(conn, base)
        small, sized = build_index.NoticeWatcher(), build_index.NoticeWatcher()
        conn.add_notice_handler(small)
        cur = conn.cursor()
        for sql in build_index.create_index_sql("t_q1", "embedding", "t_q1_embedding_hnsw", 16, 64, 1, 2):
            cur.execute(sql)
        assert small.check(fail=False) is False            # 1 MB: "graph no longer fits"
        with pytest.raises(RuntimeError):
            small.check()
        cur.execute("DROP INDEX t_q1_embedding_hnsw")
        conn.add_notice_handler(sized)
        mem = build_index.maintenance_work_mem_mb(6000, 32, 16)
        for sql in build_index.create_index_sql("t_q1", "embedding", "t_q1_embedding_hnsw", 16, 64, mem, 2):
            cur.execute(sql)
        assert sized.check(fail=False)
        queries = rng.normal(size=(50, 32)).astype(np.float32)
        _, dist = exact_topk_chunk(queries, base, k=10)
        rows = ef_tuning.sweep(pgvector.PgvectorBackend(conn, table="t_q1"), queries, base, dist[:, 9],
                               grid=(10, 40, 400))
        assert rows[0]["recall"] < 1.0 and rows[0]["recall"] < rows[-1]["recall"]


def test_binary_rows_layout():
    from harness.backend import load
    rows = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    blob = load.binary_rows(rows, 7)
    per_row = 2 + 4 + 8 + 4 + 4 + 8
    assert len(blob) == 2 * per_row
    first = blob[:per_row]
    assert first[:2] == (2).to_bytes(2, "big") and first[2:6] == (8).to_bytes(4, "big")
    assert int.from_bytes(first[6:14], "big") == 7 and int.from_bytes(first[14:18], "big") == 12
    assert first[18:20] == (2).to_bytes(2, "big") and np.frombuffer(first[22:], dtype=">f4").tolist() == [1.0, -2.0]


@pytest.mark.skipif(not os.environ.get("PGVECTOR_TEST_DSN"), reason="needs PostgreSQL + pgvector")
def test_real_binary_copy_matches_text_copy_and_checks_the_count():
    import psycopg
    from harness.backend import load
    rng = np.random.default_rng(3)
    base = rng.normal(size=(2500, 12)).astype(np.float32)
    with psycopg.connect(os.environ["PGVECTOR_TEST_DSN"]) as conn:
        load.load_base(conn, base, table="t_bin", chunk=700, binary=True)
        load.load_base(conn, base, table="t_txt", chunk=700, binary=False)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM t_bin b JOIN t_txt t USING (id) WHERE b.embedding = t.embedding")
        assert cur.fetchone()[0] == 2500
        cur.execute("SELECT embedding::text FROM t_bin WHERE id = 5")
        got = np.array([float(x) for x in cur.fetchone()[0].strip("[]").split(",")], dtype=np.float32)
        assert np.array_equal(got, base[5])
        cur.execute("DROP TABLE t_bin"); cur.execute("DROP TABLE t_txt")
        conn.commit()
