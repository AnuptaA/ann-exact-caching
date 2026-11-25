#!/bin/bash
# Build the update database of a dataset in pgpool/base and snapshot it (pgpool/base@built), the source of the
# per-job clones (scripts/pg_clones.sh).
#
#   bash scripts/pg_update_base.sh build <dataset> -- <loader command>
#
# Without the pool pgpool (instance_setup.sh --zfs-size-gb) it says so, records the skip and exits 0. With the snapshot
# already present it only refreshes the record. Otherwise: initdb in pgpool/base/pgdata with the settings of the main
# cluster (harness/deploy/setup_instance.sh) plus maintenance_work_mem and max_parallel_maintenance_workers from this
# machine, start the server on UPDATE_PG_PORT (default 54330), run the loader with Q1_PG_PORT, Q1_PG_DSN and Q1_PGDATA
# pointing at it, stop the server cleanly and take the snapshot. The record goes to RECORD
# (default results/pgvector/<dataset>_base.json). Rerunning after a failed load continues in the same data directory.
set -eu -o pipefail

POOL="${PG_POOL:-pgpool}"
MOUNT_ROOT="${PG_POOL_MOUNT:-/$POOL}"
PG_BIN="${PG_BIN:-/usr/lib/postgresql/16/bin}"
PORT="${UPDATE_PG_PORT:-54330}"
SNAPSHOT="$POOL/base@built"

die() { echo "[pg_update_base] $*" >&2; exit 1; }
log() { echo "[pg_update_base] $*"; }

pg() {
    local tool="$1"
    shift
    if [ -x "$PG_BIN/$tool" ]; then "$PG_BIN/$tool" "$@"; else "$tool" "$@"; fi
}

write_record() {
    local state="$1" extra="${2:-}"
    mkdir -p "$(dirname "$RECORD")"
    printf '{"dataset": "%s", "state": "%s"%s}\n' "$DATASET" "$state" "$extra" > "$RECORD.tmp"
    mv "$RECORD.tmp" "$RECORD"
}

server_settings() {
    local ram_mb cpus workers maint
    ram_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2> /dev/null || echo 8192)
    [ -n "$ram_mb" ] || ram_mb=8192
    cpus=$(nproc 2> /dev/null || echo 8)
    workers=$((cpus > 8 ? 7 : cpus - 1))
    [ "$workers" -ge 1 ] || workers=1
    maint=$((ram_mb / 2))
    SETTINGS="shared_buffers=$((ram_mb / 4))MB maintenance_work_mem=${maint}MB max_parallel_maintenance_workers=$workers"
    cat <<CONF
# update base settings
port = $PORT
listen_addresses = '127.0.0.1'
unix_socket_directories = ''
shared_buffers = $((ram_mb / 4))MB
effective_cache_size = $((ram_mb * 7 / 10))MB
maintenance_work_mem = ${maint}MB
max_connections = 200
max_worker_processes = 16
max_parallel_workers = 16
max_parallel_maintenance_workers = $workers
max_wal_size = 64GB
checkpoint_timeout = 30min
synchronous_commit = off
wal_level = minimal
max_wal_senders = 0
statement_timeout = 0
idle_in_transaction_session_timeout = 0
tcp_keepalives_idle = 60
tcp_keepalives_interval = 10
tcp_keepalives_count = 6
CONF
}

build() {
    DATASET="${1:-}"
    [ -n "$DATASET" ] || die "usage: pg_update_base.sh build <dataset> -- <loader command>"
    shift
    [ "${1:-}" = "--" ] && shift
    local loader="$*"
    RECORD="${RECORD:-results/pgvector/${DATASET}_base.json}"
    if ! zpool list "$POOL" > /dev/null 2>&1; then
        log "no ZFS pool $POOL: update runs need --zfs-size-gb (scripts/instance_setup.sh); the update database is skipped"
        write_record skipped ', "reason": "no ZFS pool pgpool; run instance_setup.sh with --zfs-size-gb"'
        return 0
    fi
    if zfs list "$SNAPSHOT" > /dev/null 2>&1; then
        log "$SNAPSHOT exists; keeping it (destroy it and its clones to rebuild)"
        [ -f "$RECORD" ] || write_record built ", \"snapshot\": \"$SNAPSHOT\""
        return 0
    fi
    [ -n "$loader" ] || die "no loader command given"
    local data="$MOUNT_ROOT/base/pgdata" settings log_file
    log_file="${TMPDIR:-/tmp}/pg_update_base_$DATASET.log"
    if [ ! -f "$data/PG_VERSION" ]; then
        mkdir -p "$data"
        pg initdb -D "$data" -U postgres --auth=trust > /dev/null
        server_settings >> "$data/postgresql.conf"
    else
        server_settings > /dev/null
    fi
    settings="$SETTINGS"
    stopped=0
    trap 'if [ "$stopped" -eq 0 ]; then pg pg_ctl -D "$data" -m fast -w stop > /dev/null 2>&1 || true; fi' EXIT
    pg pg_ctl -D "$data" -l "$log_file" -w -t 120 start > /dev/null
    log "loading $DATASET into $data on port $PORT"
    Q1_PG_PORT="$PORT" Q1_PGDATA="$data" Q1_PG_DSN="host=127.0.0.1 port=$PORT user=postgres" sh -c "$loader" \
        || die "the loader failed; the server was stopped and the data directory kept (see $log_file)"
    pg pg_ctl -D "$data" -m fast -w stop > /dev/null
    stopped=1
    zfs snapshot "$SNAPSHOT"
    local created used
    created=$(zfs get -Hp -o value creation "$SNAPSHOT" 2> /dev/null || echo "")
    used=$(zfs list -Hp -o used "$POOL/base" 2> /dev/null || echo "")
    write_record built ", \"snapshot\": \"$SNAPSHOT\", \"created\": \"${created:-unknown}\", \"used_bytes\": \"${used:-unknown}\", \"server_settings\": \"$settings\", \"loader\": \"$(printf '%s' "$loader" | sed 's/"/\\"/g')\""
    log "snapshot $SNAPSHOT taken"
}

case "${1:-}" in
    build) shift; build "$@" ;;
    *) sed -n '2,/^set /p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
