#!/usr/bin/env bash
# Manage the Q1 PostgreSQL cluster.
#   pg.sh start | stop | status | snapshot <name> | restore <name>
# Environment: Q1_PGDATA (data directory), Q1_PG_PORT, PG_BIN (directory with pg_ctl),
# Q1_DATA (snapshots go to $Q1_DATA/snapshots).
set -euo pipefail

# shellcheck source-path=SCRIPTDIR source=lib.sh
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"

PG_BIN="${PG_BIN:-/usr/lib/postgresql/16/bin}"
PGDATA="${Q1_PGDATA:?set Q1_PGDATA}"
guard_path "$PGDATA"
PORT="${Q1_PG_PORT:-54320}"
SNAP_DIR="${Q1_DATA:?set Q1_DATA}/snapshots"
LOG="${PGDATA%/}.log"

opts="-p $PORT -c listen_addresses=127.0.0.1 -c unix_socket_directories="

case "${1:-}" in
    start)
        "$PG_BIN/pg_ctl" -D "$PGDATA" -o "$opts" -l "$LOG" -w start ;;
    stop)
        "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop ;;
    status)
        "$PG_BIN/pg_ctl" -D "$PGDATA" status ;;
    snapshot)
        name="${2:?snapshot name}"
        mkdir -p "$SNAP_DIR"
        "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop
        tar -C "$(dirname "$PGDATA")" -cf "$SNAP_DIR/$name.tar" "$(basename "$PGDATA")"
        "$PG_BIN/pg_ctl" -D "$PGDATA" -o "$opts" -l "$LOG" -w start
        echo "snapshot written to $SNAP_DIR/$name.tar" ;;
    restore)
        name="${2:?snapshot name}"
        [ -f "$SNAP_DIR/$name.tar" ] || { echo "no such snapshot: $SNAP_DIR/$name.tar" >&2; exit 1; }
        if "$PG_BIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
            "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop
        fi
        guard_path "$PGDATA"
        rm -rf -- "${PGDATA:?}"
        tar -C "$(dirname "$PGDATA")" -xf "$SNAP_DIR/$name.tar"
        "$PG_BIN/pg_ctl" -D "$PGDATA" -o "$opts" -l "$LOG" -w start
        echo "restored $name" ;;
    *)
        echo "usage: pg.sh start|stop|status|snapshot <name>|restore <name>" >&2
        exit 2 ;;
esac
