"""Index build helpers: SQL, memory sizing and the "graph no longer fits" guard."""

from dataclasses import dataclass

# Build params: Aker's published values for SPACEV and SPHERE; SIFT-10M and ESCI use SPACEV's.
BUILD_PARAMS = {
    "spacev": {"m": 16, "ef_construction": 128},
    "sphere": {"m": 32, "ef_construction": 256},
    "sift10m": {"m": 16, "ef_construction": 128},
    "esci": {"m": 16, "ef_construction": 128},
}

MEMORY_MARGIN = 1.15
NOT_FITTING = "no longer fits"


def hnsw_graph_bytes(n: int, dim: int, m: int) -> int:
    """Estimated size of the in-memory HNSW graph while building.

    Roughly the float32 vectors plus level-0 neighbor lists; calibrated against
    the ~41 GB reported for 10M x 768 with m = 32 (4112 bytes per vector).
    """
    return n * (4 * dim + 32 * m + 16)


def maintenance_work_mem_mb(n: int, dim: int, m: int, margin: float = MEMORY_MARGIN) -> int:
    return int(hnsw_graph_bytes(n, dim, m) * margin / 2 ** 20) + 1


def create_index_sql(table: str, column: str, index_name: str, m: int, ef_construction: int,
                     maintenance_mb: int, workers: int) -> list[str]:
    return [f"SET maintenance_work_mem = '{maintenance_mb}MB'",
            f"SET max_parallel_maintenance_workers = {int(workers)}",
            f"CREATE INDEX {index_name} ON {table} USING hnsw ({column} vector_l2_ops) "
            f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"]


@dataclass
class NoticeWatcher:
    """Collects server notices; a build fails when pgvector says the graph outgrew memory."""
    notices: list

    def __init__(self):
        self.notices = []

    def __call__(self, notice) -> None:
        self.notices.append(str(getattr(notice, "message_primary", notice)))

    def check(self, fail: bool = True) -> bool:
        bad = [n for n in self.notices if NOT_FITTING in n]
        if bad and fail:
            raise RuntimeError("pgvector graph outgrew maintenance_work_mem: " + bad[0])
        return not bad


def build_index(conn, table, column, dataset, n, dim, workers=8, fail_on_notice=True):
    """CREATE INDEX with sized memory; raises when the graph no longer fits."""
    params = BUILD_PARAMS[dataset]
    watcher = NoticeWatcher()
    conn.add_notice_handler(watcher)
    mem = maintenance_work_mem_mb(n, dim, params["m"])
    with conn.cursor() as cur:
        for sql in create_index_sql(table, column, f"{table}_{column}_hnsw", params["m"],
                                    params["ef_construction"], mem, workers):
            cur.execute(sql)
    conn.commit()
    watcher.check(fail_on_notice)
    return {"maintenance_work_mem_mb": mem, **params, "notices": watcher.notices}
