#!/usr/bin/env bash
# Manage the Q1 PostgreSQL cluster. start and stop can be repeated.
#   pg.sh start | stop | status | snapshot <name> | restore <name>
# Environment: Q1_PGDATA (data directory), Q1_PG_PORT, PG_BIN (directory with pg_ctl),
# Q1_DATA (snapshots go to $Q1_DATA/snapshots).
set -euo pipefail

# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"

PG_BIN="${PG_BIN:-/usr/lib/postgresql/16/bin}"
PGDATA="${Q1_PGDATA:?set Q1_PGDATA}"
guard_path "$PGDATA"
PORT="${Q1_PG_PORT:-54320}"
SNAP_DIR="${Q1_DATA:?set Q1_DATA}/snapshots"
LOG="${PGDATA%/}.log"

opts="-p $PORT -c listen_addresses=127.0.0.1 -c unix_socket_directories="

# running_port: port of the server running on this data directory; empty when none runs. A stale pid file
# counts as not running (the postmaster removes it on start).
running_port() {
    "$PG_BIN/pg_ctl" -D "$PGDATA" status > /dev/null 2>&1 || return 0
    sed -n 4p "$PGDATA/postmaster.pid"
}

# start_server: no-op when already running on the wanted port; shows the end of the server log when it fails.
start_server() {
    local current last
    current=$(running_port)
    if [ -n "$current" ]; then
        [ "$current" = "$PORT" ] || die "a server for $PGDATA already runs on port $current, not $PORT; stop it first (make q1-pg-stop)"
        echo "already running on port $PORT"
    elif ! "$PG_BIN/pg_ctl" -D "$PGDATA" -o "$opts" -l "$LOG" -w -t 120 start; then
        echo "pg_ctl start failed; last lines of $LOG:" >&2
        tail -n 20 "$LOG" >&2 || true
        last=$(tail -n 1 "$LOG" 2> /dev/null || true)
        die "PostgreSQL did not start: ${last:-no log output} (see $LOG)"
    fi
    if [ -x "$PG_BIN/pg_isready" ]; then
        "$PG_BIN/pg_isready" -h 127.0.0.1 -p "$PORT" -t 30 > /dev/null || die "PostgreSQL runs but does not accept connections on 127.0.0.1:$PORT (see $LOG)"
    fi
}

stop_server() {
    if "$PG_BIN/pg_ctl" -D "$PGDATA" status > /dev/null 2>&1; then
        "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop
    else
        echo "not running"
    fi
}

case "${1:-}" in
    start)
        start_server ;;
    stop)
        stop_server ;;
    status)
        "$PG_BIN/pg_ctl" -D "$PGDATA" status ;;
    snapshot)
        name="${2:?snapshot name}"
        mkdir -p "$SNAP_DIR"
        stop_server
        tar -C "$(dirname "$PGDATA")" -cf "$SNAP_DIR/$name.tar" "$(basename "$PGDATA")"
        start_server
        echo "snapshot written to $SNAP_DIR/$name.tar" ;;
    restore)
        name="${2:?snapshot name}"
        [ -f "$SNAP_DIR/$name.tar" ] || { echo "no such snapshot: $SNAP_DIR/$name.tar" >&2; exit 1; }
        stop_server
        guard_path "$PGDATA"
        rm -rf -- "${PGDATA:?}"
        tar -C "$(dirname "$PGDATA")" -xf "$SNAP_DIR/$name.tar"
        start_server
        echo "restored $name" ;;
    *)
        echo "usage: pg.sh start|stop|status|snapshot <name>|restore <name>" >&2
        exit 2 ;;
esac
