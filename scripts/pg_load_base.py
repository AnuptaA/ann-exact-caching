#!/usr/bin/env python3
"""Load a base-vector .npy file into its PostgreSQL database and build the HNSW index (pgvector).

  PYTHONPATH=. .venv-core/bin/python -m scripts.pg_load_base --dataset tripclick --base datasets/aker/tripclick-ip-docs.npy

For datasets the harness pipeline does not know (TripClick, MedRAG); SPACEV and the others go through
`make q1-stage`. The database is q1_<dataset>, table base(id, embedding), index base_embedding_hnsw, as
harness.backend.pgvector_live expects. Safe to rerun: a loaded table is kept, a valid index is kept.
Needs a running server (make q1-pg-start); the index memory is sized by harness.backend.build_index.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from harness.backend import build_index  # noqa: E402

logger = logging.getLogger("pg_load_base")
GIB = 2 ** 30
INDEX_NAME = "base_embedding_hnsw"


def default_dsn() -> str:
    return os.environ.get("Q1_PG_DSN") or f"host=127.0.0.1 port={os.environ.get('Q1_PG_PORT', '54320')} user=postgres"


def default_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 15))


def build_key(dataset: str) -> str:
    """Index parameters: the dataset's own when the harness has them, else SPACEV's."""
    return dataset if dataset in build_index.BUILD_PARAMS else "spacev"


def load_base_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"base file {path} does not exist (run scripts/fetch_data.sh first)")
    base = np.load(path, mmap_mode="r")
    if base.ndim != 2:
        raise SystemExit(f"{path} is not a 2-D array (shape {base.shape})")
    return base


def total_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def resource_problems(n: int, dim: int, key: str, ram: int | None, free_disk: int | None) -> list[str]:
    """What stops the load and build on this machine, in words with the numbers; empty when it fits."""
    params = build_index.BUILD_PARAMS[key]
    graph = build_index.hnsw_graph_bytes(n, dim, params["m"])
    table = n * (4 * dim + 40)
    problems = []
    if ram is not None and ram < graph * build_index.MEMORY_MARGIN:
        problems.append(f"the HNSW build for {n} x {dim} needs about {graph * build_index.MEMORY_MARGIN / GIB:.0f} GiB "
                        f"of memory (maintenance_work_mem); this machine has {ram / GIB:.0f} GiB")
    need_disk = int(2 * table + graph) + GIB
    if free_disk is not None and free_disk < need_disk:
        problems.append(f"the PostgreSQL data directory needs about {need_disk / GIB:.0f} GiB free (table, index, "
                        f"WAL); {free_disk / GIB:.0f} GiB are free")
    return problems


def table_state(conn, n: int) -> tuple[bool, bool]:
    """(table loaded with n rows, valid index present)."""
    exists = conn.execute("SELECT to_regclass('base') IS NOT NULL").fetchone()[0]
    loaded = bool(exists) and conn.execute("SELECT count(*) FROM base").fetchone()[0] == n
    has_index = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (INDEX_NAME,)).fetchone()[0]
    indexed = bool(loaded and has_index and build_index.index_is_valid(conn, INDEX_NAME))
    return loaded, indexed


def run(args: argparse.Namespace) -> int:
    import psycopg
    from psycopg.conninfo import make_conninfo
    from harness.backend import load, pgvector

    base = load_base_file(Path(args.base))
    n, dim = base.shape
    key = build_key(args.dataset)
    pgdata = Path(args.pgdata) if args.pgdata else None
    free = None
    if pgdata is not None and pgdata.exists():
        usage = os.statvfs(pgdata)
        free = usage.f_bavail * usage.f_frsize
    problems = resource_problems(n, dim, key, total_ram_bytes(), free)
    if problems:
        raise SystemExit("; ".join(problems))
    db_name = f"q1_{args.dataset}"
    try:
        admin = psycopg.connect(make_conninfo(args.dsn, dbname="postgres"), autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        raise SystemExit(f"cannot reach PostgreSQL ({exc}); start it with: make q1-pg-start")
    with admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)).fetchone():
            admin.execute(f'CREATE DATABASE "{db_name}"')
    dsn = pgvector.with_keepalives(args.dsn, db_name)
    with psycopg.connect(dsn) as conn:
        loaded, indexed = table_state(conn, n)
        if loaded and indexed and not args.force:
            logger.info(f"{db_name}: {n} rows loaded and the HNSW index is valid; nothing to do")
            return 0
        if not loaded or args.force:
            load.load_base(conn, base)
            logger.info(f"{db_name}: pgvector {pgvector.check_version(conn)}")
        conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        conn.commit()
        info = build_index.build_index(conn, "base", "embedding", key, n, dim, workers=args.workers, dsn=dsn)
        logger.info(f"{db_name}: index built: {info}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="database q1_<dataset>")
    ap.add_argument("--base", required=True, help="float32 base vectors, .npy")
    ap.add_argument("--dsn", default=default_dsn(), help="connection string of the server (database name ignored)")
    ap.add_argument("--workers", type=int, default=default_workers(), help="parallel index build workers")
    ap.add_argument("--pgdata", default=os.environ.get("Q1_PGDATA", "/data/q1/pgdata"),
                    help="PostgreSQL data directory, for the free-disk check")
    ap.add_argument("--force", action="store_true", help="reload the table and rebuild the index")
    return ap


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
