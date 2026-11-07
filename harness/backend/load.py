"""Load the float32 base into PostgreSQL (id = 0-based row position)."""

import numpy as np

CHUNK = 20_000


def copy_lines(rows: np.ndarray, start_id: int):
    """Text COPY lines "id<TAB>[v1,v2,...]" for consecutive rows."""
    for offset, row in enumerate(rows):
        yield f"{start_id + offset}\t[" + ",".join(f"{float(x):.9g}" for x in row) + "]\n"


def load_base(conn, base, table: str = "base", column: str = "embedding", chunk: int = CHUNK) -> None:
    n, d = base.shape
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"CREATE TABLE {table} (id bigint PRIMARY KEY, {column} vector({int(d)}))")
        with cur.copy(f"COPY {table} (id, {column}) FROM STDIN") as copy:
            for start in range(0, n, chunk):
                for line in copy_lines(np.asarray(base[start:start + chunk], dtype=np.float32), start):
                    copy.write(line)
    conn.commit()
