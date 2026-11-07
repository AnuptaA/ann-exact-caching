#!/usr/bin/env bash
# One-time setup of a Linux x86_64 (Ubuntu 22.04 or 24.04) GPU instance for Q1.
# Idempotent: every step checks before acting. Run via `make q1-setup`.
#
# Installs system packages, PostgreSQL 16 (PGDG) with pgvector v0.8.0 built from
# source, FAISS (system package or built from source), the Python venvs, the
# replay drivers, and initializes a PostgreSQL cluster in $Q1_PGDATA.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
Q1_DATA="${Q1_DATA:-/data/q1}"
Q1_PGDATA="${Q1_PGDATA:-$Q1_DATA/pgdata}"
Q1_PG_PORT="${Q1_PG_PORT:-5432}"
PGVECTOR_TAG="v0.8.0"
FAISS_TAG="v1.8.0"
PG_BIN=/usr/lib/postgresql/16/bin

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "setup_instance.sh: expected Linux x86_64, found $(uname -s) $(uname -m)" >&2
    exit 1
fi
if [ -n "$(git -C "$root" status --porcelain)" ]; then
    echo "setup_instance.sh: git tree is dirty; commit or stash first" >&2
    exit 1
fi

sudo apt-get update
sudo apt-get install -y build-essential cmake git curl ca-certificates gnupg lsb-release \
    python3-venv python3-dev libboost-all-dev libopenblas-dev libomp-dev zstd

# PostgreSQL 16 from the PGDG repository.
if [ ! -x "$PG_BIN/pg_ctl" ]; then
    sudo install -d /usr/share/postgresql-common/pgdg
    sudo curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
        https://www.postgresql.org/media/keys/ACCC4CF8.asc
    echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        | sudo tee /etc/apt/sources.list.d/pgdg.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y postgresql-16 postgresql-server-dev-16
    sudo systemctl disable --now postgresql || true   # we run our own cluster below
fi

# pgvector at an exact tag.
if [ ! -f "$("$PG_BIN/pg_config" --sharedir)/extension/vector.control" ] || \
   ! grep -q "default_version = '0.8.0'" "$("$PG_BIN/pg_config" --sharedir)/extension/vector.control"; then
    tmp="$(mktemp -d)"
    git clone --depth 1 --branch "$PGVECTOR_TAG" https://github.com/pgvector/pgvector "$tmp/pgvector"
    make -C "$tmp/pgvector" PG_CONFIG="$PG_BIN/pg_config"
    sudo make -C "$tmp/pgvector" PG_CONFIG="$PG_BIN/pg_config" install
    rm -rf "$tmp"
fi

# FAISS: system package if available, else built from source (CPU only).
FAISS_ROOT="${FAISS_ROOT:-/usr/local}"
if [ ! -f "$FAISS_ROOT/include/faiss/Index.h" ] && [ ! -f /usr/include/faiss/Index.h ]; then
    if apt-cache show libfaiss-dev >/dev/null 2>&1; then
        sudo apt-get install -y libfaiss-dev
        FAISS_ROOT=/usr
    else
        tmp="$(mktemp -d)"
        git clone --depth 1 --branch "$FAISS_TAG" https://github.com/facebookresearch/faiss "$tmp/faiss"
        cmake -S "$tmp/faiss" -B "$tmp/build" -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
            -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
        cmake --build "$tmp/build" -j "$(nproc)"
        sudo cmake --install "$tmp/build"
        rm -rf "$tmp"
    fi
fi

# PostgreSQL cluster on the big disk, tuned for bulk loads and HNSW builds.
mkdir -p "$Q1_DATA" "$Q1_PGDATA"
if [ ! -f "$Q1_PGDATA/PG_VERSION" ]; then
    "$PG_BIN/initdb" -D "$Q1_PGDATA" -U postgres --auth=trust
    ram_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
    cat >> "$Q1_PGDATA/postgresql.conf" <<CONF
shared_buffers = $((ram_mb / 4))MB
effective_cache_size = $((ram_mb * 7 / 10))MB
max_connections = 200
max_parallel_maintenance_workers = 7
max_wal_size = 64GB
checkpoint_timeout = 30min
synchronous_commit = off
CONF
fi

# Python environments.
[ -d "$root/.venv" ] || python3 -m venv "$root/.venv"
"$root/.venv/bin/pip" install --upgrade pip
"$root/.venv/bin/pip" install -r "$root/requirements-q1.txt"
[ -d "$root/.venv-encode" ] || python3 -m venv "$root/.venv-encode"
"$root/.venv-encode/bin/pip" install -r "$root/requirements-encode.txt"

# Replay drivers (standard and proximity).
FAISS_ROOT="$FAISS_ROOT" bash "$root/harness/deploy/build_linux.sh"

Q1_PGDATA="$Q1_PGDATA" Q1_PG_PORT="$Q1_PG_PORT" Q1_DATA="$Q1_DATA" bash "$root/harness/deploy/pg.sh" start || true
PYTHONPATH="$root" "$root/.venv/bin/python" -m harness preflight --data-dir "$Q1_DATA"
echo "setup done. Smoke test: make q1-smoke"
