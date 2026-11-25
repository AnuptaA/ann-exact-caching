#!/bin/bash
# Per-job pgvector clones on ZFS (set up by instance_setup.sh --zfs-size-gb, base built by the prepare's pgvector_db stage).
#
#   bash scripts/pg_clones.sh create <job>     clone pgpool/base@built to pgpool/<job>, start a server on a free port,
#                                              write its connection string to $PG_CLONES/<job>.dsn
#   bash scripts/pg_clones.sh destroy <job>    stop the server and destroy the clone
#   bash scripts/pg_clones.sh run <job> -- <command>   create, run the command (a single argument runs through sh -c),
#                                              destroy on exit, also when the command fails
#   bash scripts/pg_clones.sh list             clones with port and state
#   bash scripts/pg_clones.sh gc               destroy clones whose job is not running
#
# Environment: PG_CLONES (default /data/pg_clones), PG_POOL (pgpool), PG_BIN (directory of pg_ctl and psql),
# PG_CLONE_PORT_RANGE (default 55000-55999), PG_CLONE_SHARED_BUFFERS (default 256MB; small and fixed because update
# jobs measure hit rate and recall, not timing), PG_CLONE_WAIT (seconds to wait for a server, default 120).
set -eu -o pipefail

POOL="${PG_POOL:-pgpool}"
BASE="$POOL/base@built"
CLONES="${PG_CLONES:-/data/pg_clones}"
PG_BIN="${PG_BIN:-/usr/lib/postgresql/16/bin}"
PORT_RANGE="${PG_CLONE_PORT_RANGE:-55000-55999}"
SHARED_BUFFERS="${PG_CLONE_SHARED_BUFFERS:-256MB}"
WAIT="${PG_CLONE_WAIT:-120}"
MOUNT_ROOT="${PG_POOL_MOUNT:-/$POOL}"

die() { echo "[pg_clones] $*" >&2; exit 1; }
log() { echo "[pg_clones] $*"; }

pg() {
    local tool="$1"
    shift
    if [ -x "$PG_BIN/$tool" ]; then "$PG_BIN/$tool" "$@"; else "$tool" "$@"; fi
}

check_job() {
    case "$1" in "" | *[!A-Za-z0-9._:-]* | .*) die "bad job id: '$1' (letters, digits, . _ - : only)" ;; esac
}

datadir() { echo "$MOUNT_ROOT/$1/pgdata"; }

lock() {
    local tries=0
    until mkdir "$CLONES/.lock" 2> /dev/null; do
        tries=$((tries + 1))
        [ "$tries" -lt 600 ] || die "could not take $CLONES/.lock (remove it if no pg_clones.sh runs)"
        sleep 0.1
    done
}

unlock() { rmdir "$CLONES/.lock" 2> /dev/null || true; }

port_busy() { (exec 3<> "/dev/tcp/127.0.0.1/$1") 2> /dev/null; }

# free_port: first port of the range that no other clone reserved and nothing listens on.
free_port() {
    local low="${PORT_RANGE%-*}" high="${PORT_RANGE#*-}" p
    for p in $(seq "$low" "$high"); do
        if ! grep -qx "$p" "$CLONES"/*.port 2> /dev/null && ! port_busy "$p"; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

answers() { pg psql -h 127.0.0.1 -p "$1" -U postgres -d postgres -Atc 'select 1' > /dev/null 2>&1; }

start_clone() {
    local job="$1" port="$2" data
    data=$(datadir "$job")
    if ! zfs list "$POOL/$job" > /dev/null 2>&1; then
        zfs list "$BASE" > /dev/null 2>&1 || die "no snapshot $BASE: run make prepare for the dataset on an instance set up with --zfs-size-gb"
        zfs clone "$BASE" "$POOL/$job"
    fi
    pg pg_ctl -D "$data" -l "$CLONES/$job.log" -w -t "$WAIT" \
        -o "-p $port -c listen_addresses=127.0.0.1 -c unix_socket_directories= -c shared_buffers=$SHARED_BUFFERS" start > /dev/null
    local waited=0
    until answers "$port"; do
        waited=$((waited + 1))
        [ "$waited" -lt "$WAIT" ] || return 1
        sleep 1
    done
    printf '%s\n' "host=127.0.0.1 port=$port user=postgres" > "$CLONES/$job.dsn.tmp"
    mv "$CLONES/$job.dsn.tmp" "$CLONES/$job.dsn"
    printf '{"job": "%s", "dataset": "%s/%s", "snapshot": "%s", "port": %s, "shared_buffers": "%s"}\n' \
        "$job" "$POOL" "$job" "$BASE" "$port" "$SHARED_BUFFERS" > "$CLONES/$job.json"
}

create() {
    local job="${1:-}" port
    check_job "$job"
    mkdir -p "$CLONES"
    if [ -f "$CLONES/$job.dsn" ] && answers "$(sed -n 's/.*port=\([0-9]*\).*/\1/p' "$CLONES/$job.dsn")"; then
        log "clone $job already runs"
        return 0
    fi
    lock
    port=$(free_port) || { unlock; die "no free port in $PORT_RANGE"; }
    echo "$port" > "$CLONES/$job.port"
    unlock
    if ! start_clone "$job" "$port"; then
        echo "[pg_clones] clone $job failed to start (see $CLONES/$job.log); destroying it" >&2
        destroy "$job" || true
        return 1
    fi
    log "clone $job on port $port ($CLONES/$job.dsn)"
}

destroy() {
    local job="${1:-}" data
    check_job "$job"
    data=$(datadir "$job")
    if [ -d "$data" ] && pg pg_ctl -D "$data" status > /dev/null 2>&1; then
        pg pg_ctl -D "$data" -m immediate -w stop > /dev/null 2>&1 || true
    fi
    if zfs list "$POOL/$job" > /dev/null 2>&1; then
        zfs destroy -f "$POOL/$job" || die "could not destroy $POOL/$job"
    fi
    rm -f "$CLONES/$job".{dsn,port,json,pid,log,dsn.tmp}
    log "clone $job destroyed"
}

run() {
    local job="${1:-}" rc=0
    check_job "$job"
    shift
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || die "run needs a command after --"
    mkdir -p "$CLONES"
    echo "$$" > "$CLONES/$job.pid"
    trap 'destroy "$job" || true' EXIT
    trap 'exit 143' INT TERM HUP
    create "$job" || exit 1
    export PG_CLONES="$CLONES"
    if [ $# -eq 1 ]; then sh -c "$1" || rc=$?; else "$@" || rc=$?; fi
    exit "$rc"
}

jobs_of_clones() {
    local f
    for f in "$CLONES"/*.port; do
        [ -e "$f" ] || continue
        basename "$f" .port
    done
}

alive() {
    local pidfile="$CLONES/$1.pid"
    [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2> /dev/null
}

list() {
    local job state
    for job in $(jobs_of_clones); do
        state=stopped
        if answers "$(cat "$CLONES/$job.port")"; then state=running; fi
        echo "$job port=$(cat "$CLONES/$job.port") $state runner=$(alive "$job" && echo alive || echo none)"
    done
}

gc() {
    local job
    for job in $(jobs_of_clones); do
        if alive "$job"; then continue; fi
        # a clone still being made by a runner that has not written its pid file yet
        if [ -n "$(find "$CLONES/$job.port" -mmin -1 2> /dev/null)" ] && ! [ -f "$CLONES/$job.pid" ]; then continue; fi
        destroy "$job"
    done
}

case "${1:-}" in
    create) create "${2:-}" ;;
    destroy) destroy "${2:-}" ;;
    run) shift; run "$@" ;;
    list) list ;;
    gc) gc ;;
    *) sed -n '2,/^set /p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
