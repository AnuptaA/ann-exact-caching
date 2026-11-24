#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 x86_64 instance for the evaluation. Run as root. Idempotent.
#   sudo bash scripts/ec2_bootstrap.sh [--role prepare|run] [--dry-run] [--only STEP ...] [--skip STEP ...] [--no-data] [--no-smoke]
#
# Roles: prepare (default) fetches the data and builds the shared artifacts afterwards (make prepare); run skips the data
# fetch and every build of data or indexes: it pulls the artifacts of its shard from the store (make run SHARD=...).
#
# Steps, in order:
#   base        harness/deploy/setup_instance.sh: system packages, pinned CMake, Boost and FAISS, PostgreSQL 16 with
#               pgvector 0.8.0 (isolated cluster), the harness venv, the Q1 drivers, preflight
#   limits      raises fs.aio-max-nr (DiskANN and the replay open many AIO contexts)
#   venvs       .venv-core (Python 3.11) and .venv-ann from envs/ (diskannpy in .venv-ann)
#   core        builds the ReCal core and the cache core
#   aker        the vendored Aker library, standard and Proximity modes, against the pinned Boost and FAISS
#   qvcache     the authors' QVCache artifact (pinned commit in harness/deploy/pins.env), built natively against
#               the .venv-ann interpreter so DiskANN and pgvector stay in the same process (no extra hop)
#   links       points datasets/{aker,medrag,tripclick} and simulations/approximate_hits/results at the data volume
#   store       boto3 for the S3 store (credentials from the instance role; set RECAL_STORE=s3://bucket/prefix)
#   data        scripts/fetch_data.sh --all (skipped with --no-data and with --role run)
#   smoke       tests, matrix dry run, a small ReCal cell with every ablation, library and backend checks
#               (skipped with --no-smoke)
#
# Environment:
#   DATA_DIR      data volume (default /data); datasets and answer tables go to $DATA_DIR/datasets
#   RESULTS_DIR   results (default $DATA_DIR/results); linked to simulations/approximate_hits/results
#   TP            third-party build prefix (default $DATA_DIR/third_party); Aker builds there
#   RECAL_DRIVE_* dataset Drive folder links for the data step (or .env.local in the checkout)
#   PYTHON311     interpreter for .venv-core (default python3.11; installed from the deadsnakes PPA if missing)
# Every step prints PASS or FAIL in the summary; a failed step does not stop independent ones.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/data}"
RESULTS_DIR="${RESULTS_DIR:-$DATA_DIR/results}"
TP="${TP:-$DATA_DIR/third_party}"
PYTHON311="${PYTHON311:-python3.11}"
DRY=0
ROLE=prepare
SMOKE=1
DATA=1
ONLY=()
SKIP=()
while [ $# -gt 0 ]; do
    case "$1" in
        --role) shift; ROLE="$1"; case "$ROLE" in prepare|run) ;; *) echo "--role must be prepare or run" >&2; exit 2 ;; esac ;;
        --dry-run) DRY=1 ;;
        --no-data) DATA=0 ;;
        --no-smoke) SMOKE=0 ;;
        --only) shift; ONLY+=("$1") ;;
        --skip) shift; SKIP+=("$1") ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

[ "$ROLE" = run ] && DATA=0

log() { printf '[bootstrap] %s\n' "$*"; }
run() {
    if [ "$DRY" -eq 1 ]; then printf '[dry-run] %s\n' "$*"; return 0; fi
    "$@"
}
selected() {
    local s
    for s in "${SKIP[@]:-}"; do [ "$s" = "$1" ] && return 1; done
    [ "${#ONLY[@]}" -eq 0 ] && return 0
    for s in "${ONLY[@]}"; do [ "$s" = "$1" ] && return 0; done
    return 1
}

step_preconditions() {
    [ "$DRY" -eq 1 ] && return 0
    [ "$(uname -s)" = "Linux" ] || { log "not Linux"; return 1; }
    [ "$(id -u)" -eq 0 ] || { log "run as root"; return 1; }
    grep -q 'VERSION_ID="24.04"' /etc/os-release || log "warning: not Ubuntu 24.04"
}

step_base() {
    run mkdir -p "$DATA_DIR" "$RESULTS_DIR"
    Q1_DATA="${Q1_DATA:-$DATA_DIR/q1}" run bash "$root/harness/deploy/setup_instance.sh"
}

step_limits() {
    run sysctl -w fs.aio-max-nr=1048576
    run bash -c "echo 'fs.aio-max-nr = 1048576' > /etc/sysctl.d/99-recal-aio.conf"
}

step_venvs() {
    if ! command -v "$PYTHON311" > /dev/null 2>&1; then
        run add-apt-repository -y ppa:deadsnakes/ppa
        run apt-get update -qq
        run apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    fi
    run "$PYTHON311" -m venv "$root/.venv-core"
    run "$root/.venv-core/bin/pip" install --quiet -r "$root/envs/core.txt" pybind11 pytest
    run "$PYTHON311" -m venv "$root/.venv-ann"
    run "$root/.venv-ann/bin/pip" install --quiet -r "$root/envs/ann.txt" pytest
}

step_core() {
    run sh "$root/recal/core/build.sh"
    run sh "$root/simulations/approximate_hits/cachecore/build.sh"
}

step_aker() {
    # the pinned Boost and FAISS of the harness toolchain, installed where aker_real/build.sh looks for them
    run mkdir -p "$TP"
    local tools="${Q1_TOOLS:-/opt/q1}"
    # the toolchain leaves Boost empty when the system one is new enough: then the system prefix is used
    local boost_prefix="$tools/boost"
    [ -d "$boost_prefix" ] || boost_prefix=/usr
    [ -e "$TP/boost-install" ] || run ln -s "$boost_prefix" "$TP/boost-install"
    [ -e "$TP/faiss-install" ] || run ln -s "$tools/faiss" "$TP/faiss-install"
    TP="$TP" AKER_SRC=vendored run sh "$root/simulations/approximate_hits/aker_real/build.sh"
}

step_qvcache() {
    # shellcheck source=../harness/deploy/pins.env
    . "$root/harness/deploy/pins.env"
    run apt-get install -y -qq libaio-dev libgoogle-perftools-dev libunwind-dev libboost-program-options-dev \
        libmkl-full-dev libcpprest-dev libeigen3-dev libspdlog-dev libnuma-dev libtbb-dev
    local src="$TP/qvcache"
    if [ ! -d "$src/.git" ]; then
        run mkdir -p "$TP"
        run git init -q "$src"
        run git -C "$src" remote add origin "$QVCACHE_REPO"
    fi
    run git -C "$src" fetch -q --depth 1 origin "$QVCACHE_COMMIT"
    run git -C "$src" checkout -q "$QVCACHE_COMMIT"
    run "$root/.venv-ann/bin/pip" install --quiet pybind11 numpy
    run cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$root/.venv-ann/bin/python"
    run cmake --build "$src/build" -j "$(nproc 2>/dev/null || echo 8)"
    # make the binding importable from .venv-ann
    run bash -c "d=\$(dirname \$(find '$src/build' -name 'qvcache*.so' | head -n1)) && \
        echo \$d > \$('$root/.venv-ann/bin/python' -c 'import site; print(site.getsitepackages()[0])')/qvcache.pth"
}

step_links() {
    local rel
    for rel in datasets/aker datasets/medrag datasets/tripclick; do
        run mkdir -p "$DATA_DIR/$rel"
        [ -e "$root/$rel" ] || run ln -s "$DATA_DIR/$rel" "$root/$rel"
    done
    run mkdir -p "$RESULTS_DIR"
    [ -e "$root/simulations/approximate_hits/results" ] || \
        run ln -s "$RESULTS_DIR" "$root/simulations/approximate_hits/results"
}

step_store() {
    run "$root/.venv-core/bin/pip" install --quiet boto3
    [ "$DRY" -eq 1 ] || [ -n "${RECAL_STORE:-}${RECAL_S3_BUCKET:-}" ] || log "note: set RECAL_STORE before make prepare / make run"
}

step_data() {
    [ "$DATA" -eq 1 ] || { log "data fetch skipped (--no-data)"; return 0; }
    RECAL_REPO="$root" run sh "$root/scripts/fetch_data.sh" --all
}

step_smoke() {
    [ "$SMOKE" -eq 1 ] || { log "smoke test skipped (--no-smoke)"; return 0; }
    run "$root/.venv-core/bin/python" -m pytest -q "$root/tests/recal" "$root/tests/test_cachecore.py"
    run "$root/.venv-core/bin/python" -m scripts.gen_matrix --config "$root/experiments/matrix.example.json" --dry-run
    run env PYTHONPATH="$root" "$root/.venv-core/bin/python" -m scripts.smoke_cells --out "$RESULTS_DIR/smoke"
}

summary=()
failed=0
for name in preconditions base limits venvs core aker qvcache links store data smoke; do
    if ! selected "$name"; then summary+=("$name: NOT RUN"); continue; fi
    log "step $name"
    if (cd "$root" && "step_$name"); then summary+=("$name: PASS"); else summary+=("$name: FAIL"); failed=1; fi
done
printf '\n[bootstrap] summary\n'
printf '  %s\n' "${summary[@]}"
exit "$failed"
