#!/usr/bin/env bash
# One-time setup of a Linux x86_64 (Ubuntu 22.04 or 24.04) instance for Q1. Idempotent. `make q1-setup`.
#   setup_instance.sh [--ci]     --ci skips the data-volume, GPU and encoder-venv steps (CI runners)
#
# Builds from pinned, verified sources whatever stock Ubuntu lacks or has too old (Boost, CMake, FAISS,
# PostgreSQL 16 from PGDG, pgvector), creates its own venvs (never the AMI's conda or default python),
# initializes an isolated PostgreSQL cluster on its own port, builds the replay drivers and runs preflight.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
here="$root/harness/deploy"
# shellcheck source-path=SCRIPTDIR source=lib.sh
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$here/lib.sh"
# shellcheck source-path=SCRIPTDIR source=pins.env
# shellcheck source-path=SCRIPTDIR source=pins.env
. "$here/pins.env"
# shellcheck source-path=SCRIPTDIR source=toolchain.sh
# shellcheck source-path=SCRIPTDIR source=toolchain.sh
. "$here/toolchain.sh"

CI=0
for arg in "$@"; do
    case "$arg" in
        --ci) CI=1 ;;
        *) die "unknown argument: $arg" ;;
    esac
done

Q1_DATA="${Q1_DATA:-/data/q1}"
Q1_TOOLS="${Q1_TOOLS:-/opt/q1}"
Q1_PGDATA="${Q1_PGDATA:-$Q1_DATA/pgdata}"
Q1_PG_PORT="${Q1_PG_PORT:-54320}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
PG_BIN="/usr/lib/postgresql/$PG_MAJOR/bin"
MIN_DATA_GB="${Q1_MIN_DATA_GB:-400}"

# Do not inherit the AMI's conda or any active virtualenv: our venvs are created from $PYTHON_BIN.
PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -viE 'conda|miniforge|mambaforge' | paste -sd: -)"
export PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL VIRTUAL_ENV PYTHONPATH PYTHONHOME PIP_REQUIRE_VIRTUALENV

if [ -z "${Q1_SETUP_ALLOW_NON_LINUX:-}" ] && { [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; }; then
    die "expected Linux x86_64, found $(uname -s) $(uname -m)"
fi
if [ -z "${Q1_DRYRUN:-}" ] && [ -n "$(git -C "$root" status --porcelain)" ]; then
    die "git tree is dirty; commit or stash first"
fi
[ -x "$PYTHON_BIN" ] || die "$PYTHON_BIN not found (set PYTHON_BIN to a system python >= 3.10, not conda)"
py_ver=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
version_ge "$py_ver" 3.10 || die "$PYTHON_BIN is Python $py_ver, need >= 3.10"
log "python $py_ver at $PYTHON_BIN; data $Q1_DATA; tools $Q1_TOOLS; postgres port $Q1_PG_PORT"
log "PATH after removing conda entries: $PATH"

# --- data volume -------------------------------------------------------------------------------------
if [ "$CI" -eq 0 ]; then
    run sudo mkdir -p "$Q1_DATA"
    run sudo chown "$(id -u):$(id -g)" "$Q1_DATA"
    if [ -z "${Q1_DRYRUN:-}" ]; then
        mount_target=$(findmnt -no TARGET --target "$Q1_DATA" 2>/dev/null || df --output=target "$Q1_DATA" | tail -1)
        if [ "$mount_target" = "/" ] && [ -z "${Q1_ALLOW_ROOT_VOLUME:-}" ]; then
            die "$Q1_DATA is on the root volume; mount the large data volume there (or set Q1_ALLOW_ROOT_VOLUME=1)"
        fi
        total_gb=$(df --output=size -BG "$Q1_DATA" | tail -1 | tr -dc '0-9')
        free_gb=$(df --output=avail -BG "$Q1_DATA" | tail -1 | tr -dc '0-9')
        log "data volume $mount_target: ${total_gb} GB total, ${free_gb} GB free"
        [ "$total_gb" -ge "$MIN_DATA_GB" ] || die "data volume is ${total_gb} GB, need at least ${MIN_DATA_GB} GB for all four datasets (Q1_MIN_DATA_GB to override)"
    fi
else
    run mkdir -p "$Q1_DATA"
fi
run sudo mkdir -p "$Q1_TOOLS"
run sudo chown "$(id -u):$(id -g)" "$Q1_TOOLS"

# --- toolchain ---------------------------------------------------------------------------------------
[ -z "${Q1_DRYRUN:-}" ] || log "dry run: state-changing commands are only printed"
[ -n "${Q1_DRYRUN:-}" ] || wait_for_apt
ensure_build_packages
ensure_cmake
ensure_boost
ensure_faiss
ensure_postgres

# --- isolated PostgreSQL cluster ---------------------------------------------------------------------
guard_path "$Q1_PGDATA"
if [ -n "${Q1_DRYRUN:-}" ] || [ ! -f "$Q1_PGDATA/PG_VERSION" ]; then
    run mkdir -p "$Q1_PGDATA"
    run "$PG_BIN/initdb" -D "$Q1_PGDATA" -U postgres --auth=trust
    ram_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 8192)
    cpus=$(nproc 2>/dev/null || echo 8)
    workers=$(( cpus > 8 ? 7 : cpus - 1 ))
    [ "$workers" -ge 1 ] || workers=1
    run bash -c "cat >> '$Q1_PGDATA/postgresql.conf' <<CONF
# q1 settings
port = $Q1_PG_PORT
listen_addresses = '127.0.0.1'
unix_socket_directories = ''
shared_buffers = $((ram_mb / 4))MB
effective_cache_size = $((ram_mb * 7 / 10))MB
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
CONF"
fi

# --- python environments (own venvs from the system python) -------------------------------------------
if [ "$CI" -eq 1 ]; then
    lock="$root/requirements-ci.lock"
else
    lock="$root/requirements-q1.lock"
fi
run "$PYTHON_BIN" -m venv "$root/.venv"
if [ -f "$lock" ]; then
    run "$root/.venv/bin/pip" install --quiet --require-hashes -r "$lock"
else
    die "missing $lock (generate with uv pip compile; see harness/deploy/README or EC2_FIRST_RUN.md)"
fi
if [ "$CI" -eq 0 ]; then
    run "$PYTHON_BIN" -m venv "$root/.venv-encode"
    run "$root/.venv-encode/bin/pip" install --quiet --require-hashes -r "$root/requirements-encode.lock"
    # the GPU must be usable by torch in the main venv, otherwise ground truth would fall back to the CPU
    if [ -z "${Q1_DRYRUN:-}" ]; then
        "$root/.venv/bin/python" -c "import torch, sys; torch.zeros(1).cuda(); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" \
            || die "torch cannot use the GPU (driver too old for this torch build, or no GPU); fix before generating"
    fi
fi

# --- replay drivers -----------------------------------------------------------------------------------
CMAKE_BIN="$CMAKE_BIN" BOOST_ROOT="$BOOST_ROOT" FAISS_ROOT="$FAISS_ROOT" run bash "$here/build_linux.sh"

# --- start the cluster and check it ---------------------------------------------------------------------
Q1_DATA="$Q1_DATA" Q1_PGDATA="$Q1_PGDATA" Q1_PG_PORT="$Q1_PG_PORT" PG_BIN="$PG_BIN" run bash "$here/pg.sh" start
PYTHONPATH="$root" run "$root/.venv/bin/python" -m harness preflight --data-dir "$Q1_DATA" --dsn "host=127.0.0.1 port=$Q1_PG_PORT user=postgres"
log "setup done. Next: make q1-smoke"
