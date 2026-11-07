"""Load the float32 base into PostgreSQL (id = 0-based row position).

Binary COPY of the pgvector `vector` wire format (int16 dim, int16 unused, float4s in
network byte order) is the default; the text path stays as a fallback and for tests.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

CHUNK = 20_000
LOG_EVERY = 500_000
BINARY_HEADER = b"PGCOPY\n\xff\r\n\x00" + (0).to_bytes(4, "big") + (0).to_bytes(4, "big")
BINARY_TRAILER = (-1).to_bytes(2, "big", signed=True)


def copy_lines(rows: np.ndarray, start_id: int):
    """Text COPY lines "id<TAB>[v1,v2,...]" for consecutive rows."""
    for offset, row in enumerate(rows):
        yield f"{start_id + offset}\t[" + ",".join(f"{float(x):.9g}" for x in row) + "]\n"


def binary_rows(rows: np.ndarray, start_id: int) -> bytes:
    """Binary COPY tuples (bigint id, vector) for consecutive rows."""
    n, d = rows.shape
    row_type = np.dtype([("nfields", ">i2"), ("id_len", ">i4"), ("id", ">i8"), ("vec_len", ">i4"),
                         ("dim", ">i2"), ("unused", ">i2"), ("vec", ">f4", (d,))])
    out = np.empty(n, dtype=row_type)
    out["nfields"], out["id_len"], out["vec_len"] = 2, 8, 4 + 4 * d
    out["id"] = np.arange(start_id, start_id + n)
    out["dim"], out["unused"] = d, 0
    out["vec"] = rows
    return out.tobytes()


def load_base(conn, base, table: str = "base", column: str = "embedding", chunk: int = CHUNK,
              binary: bool = True) -> None:
    n, d = base.shape
    started = time.time()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"CREATE TABLE {table} (id bigint PRIMARY KEY, {column} vector({int(d)}))")
        fmt = "(FORMAT BINARY)" if binary else ""
        with cur.copy(f"COPY {table} (id, {column}) FROM STDIN {fmt}") as copy:
            if binary:
                copy.write(BINARY_HEADER)
            for start in range(0, n, chunk):
                rows = np.asarray(base[start:start + chunk], dtype=np.float32)
                if binary:
                    copy.write(binary_rows(rows, start))
                else:
                    for line in copy_lines(rows, start):
                        copy.write(line)
                done = min(start + chunk, n)
                if done % LOG_EVERY < chunk or done == n:
                    rate = done / max(time.time() - started, 1e-6)
                    logger.info(f"pg load {done}/{n} rows ({rate:.0f} rows/s, "
                                f"ETA {(n - done) / max(rate, 1e-6) / 60:.1f} min)")
            if binary:
                copy.write(BINARY_TRAILER)
    conn.commit()
    with conn.cursor() as cur:
        count = cur.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    if count != n:
        raise RuntimeError(f"loaded {count} rows into {table}, expected {n}")
