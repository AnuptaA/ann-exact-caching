#!/usr/bin/env bash
# One-time setup of a Linux x86_64 (Ubuntu 22.04 or 24.04) instance for Q1. Idempotent. `make q1-setup`.
#   setup_instance.sh [--ci]     --ci skips the data-volume, GPU and encoder-venv steps (CI runners)
#
# Builds from pinned, verified sources whatever stock Ubuntu lacks or has too old (Boost, CMake, FAISS,
# PostgreSQL 16 from PGDG, pgvector), creates its own venvs (never the AMI's conda or default python),
# initializes an isolated PostgreSQL cluster on its own port, builds the replay drivers and runs preflight.
#
# Each stage is a step. A failed step does not stop the steps that do not depend on it, the GPU check runs
# last, and a summary lists every step as PASS, FAIL, SKIPPED (a dependency failed), N/A or NOT RUN.
# Exit code is 1 if any step failed. No GPU is a failure unless Q1_ALLOW_NO_GPU=1 (then a loud warning).
# Disk and RAM shortfalls in preflight are failures unless Q1_ALLOW_SHORTFALL=1 (dry run on a small machine).
#
# Environment overrides:
#   Q1_DATA                     data volume directory (default /data/q1)
#   Q1_TOOLS                    pinned toolchain prefix (default /opt/q1)
#   Q1_PGDATA, Q1_PG_PORT       PostgreSQL data directory and port (default $Q1_DATA/pgdata, 54320)
#   PYTHON_BIN                  system python for the venvs (default /usr/bin/python3)
#   Q1_MIN_DATA_GB              minimum size of the data volume (default 400; 10 is enough for the smoke test)
#   Q1_ALLOW_ROOT_VOLUME=1      accept a data directory on the root volume
#   Q1_SETUP_DATASETS           datasets preflight sizes disk and RAM for (default "spacev sift10m sphere esci";
#                               "none" skips the sizing, for a CPU dry run)
#   Q1_ALLOW_NO_GPU=1           a missing or unusable GPU is a warning, not a failure (CPU dry run)
#   Q1_ALLOW_SHORTFALL=1        preflight disk and RAM shortfalls are warnings with the numbers (small dry-run
#                               machine; combine with Q1_SETUP_DATASETS=tiny or one dataset name). Other preflight
#                               problems (dirty tree, root volume, PostgreSQL) still fail.
#   Q1_APT_WAIT_SECONDS, Q1_CLOUD_INIT_WAIT_SECONDS    limits of the apt lock and cloud-init waits (900, 600)
#   Q1_DRYRUN                   print state-changing commands only (tests)
#   Q1_SETUP_ALLOW_NON_LINUX    run the decision logic on a non-Linux machine (tests)
#   Q1_GPU_CHECK_CMD, Q1_PREFLIGHT_CMD    replace the GPU check and preflight commands (tests)
#   Q1_CHECK_CLEAN_TREE=1       run the clean-tree step in a dry run too (tests)
# shellcheck disable=SC2317,SC2329
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
here="$root/harness/deploy"
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$here/lib.sh"
# shellcheck source-path=SCRIPTDIR source=pins.env
. "$here/pins.env"
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
SETUP_DATASETS="${Q1_SETUP_DATASETS:-spacev sift10m sphere esci}"

# Do not inherit the AMI's conda or any active virtualenv: our venvs are created from $PYTHON_BIN.
PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -viE 'conda|miniforge|mambaforge' | paste -sd: -)"
export PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL VIRTUAL_ENV PYTHONPATH PYTHONHOME PIP_REQUIRE_VIRTUALENV
# Nothing may prompt: a failed credential or password request must fail the step, not hang it.
export GIT_TERMINAL_PROMPT=0 PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1

log "data $Q1_DATA; tools $Q1_TOOLS; postgres port $Q1_PG_PORT"
log "PATH after removing conda entries: $PATH"
[ -z "${Q1_DRYRUN:-}" ] || log "dry run: state-changing commands are only printed"

# --- step runner ---------------------------------------------------------------------------------------
STEP_ORDER=(preconditions data_volume tools_dir first_boot_wait system_packages cmake boost faiss postgres
    pg_cluster python_env encode_env drivers pg_start preflight clean_tree gpu)
STATE_DIR="$(mktemp -d)"
SUMMARY_PRINTED=0
NO_GPU_WARNING=0
CMAKE_BIN=""
BOOST_ROOT=""
FAISS_ROOT=""

set_status() {
    printf '%s\n' "$2" > "$STATE_DIR/$1.status"
    printf '%s\n' "${3:-}" > "$STATE_DIR/$1.note"
}

status_of() {
    if [ -f "$STATE_DIR/$1.status" ]; then cat "$STATE_DIR/$1.status"; else echo "NOT RUN"; fi
}

# mark_na REASON: called by a step that does not apply here; the step then ends as N/A.
mark_na() { printf '%s\n' "$*" > "$STATE_DIR/$STEP_NAME.na"; }

# mark_warn NOTE: the step passed with a warning (it counts as passed for its dependents).
mark_warn() { printf '%s\n' "$*" > "$STATE_DIR/$STEP_NAME.warn"; }

# step NAME [DEPENDENCY...]: run step_NAME in a subshell with errexit on, unless a dependency did not pass.
step() {
    local name="$1" dep status rc note
    shift
    for dep in "$@"; do
        status=$(status_of "$dep")
        case "$status" in
            PASS | WARN | N/A) ;;
            *)
                set_status "$name" SKIPPED "needs $dep ($status)"
                log "step $name: SKIPPED (needs $dep: $status)"
                return 0
                ;;
        esac
    done
    log "step $name: start"
    rm -f "$STATE_DIR/$name.na" "$STATE_DIR/$name.warn" "$STATE_DIR/$name.err"
    set +e
    (
        set -euo pipefail
        STEP_NAME="$name"
        Q1_ERR_FILE="$STATE_DIR/$name.err"
        export Q1_ERR_FILE
        "step_$name"
        printf 'CMAKE_BIN=%q\nBOOST_ROOT=%q\nFAISS_ROOT=%q\n' "$CMAKE_BIN" "$BOOST_ROOT" "$FAISS_ROOT" > "$STATE_DIR/env"
    )
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        note=$(head -n1 "$STATE_DIR/$name.err" 2>/dev/null || true)
        set_status "$name" FAIL "${note:-exit code $rc}"
        log "step $name: FAIL (${note:-exit code $rc})"
        return 0
    fi
    # shellcheck source=/dev/null
    . "$STATE_DIR/env"
    if [ -f "$STATE_DIR/$name.na" ]; then
        set_status "$name" N/A "$(cat "$STATE_DIR/$name.na")"
        log "step $name: N/A ($(cat "$STATE_DIR/$name.na"))"
    elif [ -f "$STATE_DIR/$name.warn" ]; then
        set_status "$name" WARN "$(cat "$STATE_DIR/$name.warn")"
        log "step $name: WARN ($(cat "$STATE_DIR/$name.warn"))"
    else
        set_status "$name" PASS
        log "step $name: PASS"
    fi
}

hard_stop() { [ "$(status_of "$1")" = PASS ] || finish; }

# finish: print the summary and exit. A failed GPU check is a warning only when Q1_ALLOW_NO_GPU is exactly 1.
finish() {
    local name status note failed="" rc=0
    if [ "$(status_of gpu)" = FAIL ] && [ "${Q1_ALLOW_NO_GPU:-}" = "1" ]; then
        set_status gpu WARN "$(cat "$STATE_DIR/gpu.note") (ignored: Q1_ALLOW_NO_GPU=1)"
        NO_GPU_WARNING=1
    fi
    SUMMARY_PRINTED=1
    echo
    log "setup summary"
    [ "${Q1_ALLOW_SHORTFALL:-}" != "1" ] || log "mode: DRY RUN (Q1_ALLOW_SHORTFALL=1): disk and RAM shortfalls are warnings"
    for name in "${STEP_ORDER[@]}"; do
        status=$(status_of "$name")
        note=$(cat "$STATE_DIR/$name.note" 2>/dev/null || true)
        printf '  %-16s %-8s %s\n' "$name" "$status" "$note"
        case "$status" in
            FAIL | "NOT RUN")
                failed="$failed $name"
                rc=1
                ;;
        esac
    done
    if [ "$NO_GPU_WARNING" -eq 1 ]; then
        {
            echo "!!! WARNING: no usable GPU (Q1_ALLOW_NO_GPU=1). Ground truth, embedding and encoding will refuse"
            echo "!!! to run on the CPU except for the tiny smoke dataset. Do not generate real datasets here."
        } >&2
    fi
    if [ "${Q1_ALLOW_SHORTFALL:-}" = "1" ]; then
        {
            echo "!!! DRY-RUN MODE (Q1_ALLOW_SHORTFALL=1): disk and RAM shortfalls were reported as warnings, not failures."
            echo "!!! This machine is not sized for the real datasets. Do not generate them here."
        } >&2
    fi
    if [ "$rc" -ne 0 ]; then
        log "setup FAILED:$failed"
    elif [ "$NO_GPU_WARNING" -eq 1 ] || [ "${Q1_ALLOW_SHORTFALL:-}" = "1" ]; then
        local notes=""
        [ "$NO_GPU_WARNING" -eq 0 ] || notes=" a GPU warning (Q1_ALLOW_NO_GPU=1)"
        [ "${Q1_ALLOW_SHORTFALL:-}" != "1" ] || notes="$notes${notes:+ and} dry-run shortfall mode (Q1_ALLOW_SHORTFALL=1)"
        log "setup OK with$notes. Next: make q1-smoke"
    else
        log "setup done. Next: make q1-smoke"
    fi
    rm -rf -- "${STATE_DIR:?}"
    exit "$rc"
}
trap 'if [ "$SUMMARY_PRINTED" -eq 0 ]; then finish; fi' EXIT

# --- steps -----------------------------------------------------------------------------------------------
step_preconditions() {
    if [ -z "${Q1_SETUP_ALLOW_NON_LINUX:-}" ] && { [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; }; then
        die "expected Linux x86_64, found $(uname -s) $(uname -m)"
    fi
    if [ -z "${Q1_DRYRUN:-}" ] && [ -n "$(git -C "$root" status --porcelain)" ]; then
        die "git tree is dirty; commit or stash first"
    fi
    [ -x "$PYTHON_BIN" ] || die "$PYTHON_BIN not found (set PYTHON_BIN to a system python >= 3.10, not conda)"
    local py_ver
    py_ver=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    version_ge "$py_ver" 3.10 || die "$PYTHON_BIN is Python $py_ver, need >= 3.10"
    log "python $py_ver at $PYTHON_BIN"
    [ -n "${Q1_DRYRUN:-}" ] || require_sudo
}

step_data_volume() {
    if [ "$CI" -eq 1 ]; then
        run mkdir -p "$Q1_DATA"
        return 0
    fi
    run sudo mkdir -p "$Q1_DATA"
    run sudo chown "$(id -u):$(id -g)" "$Q1_DATA"
    [ -z "${Q1_DRYRUN:-}" ] || return 0
    local mount_target total_gb free_gb
    mount_target=$(findmnt -no TARGET --target "$Q1_DATA" 2>/dev/null || df --output=target "$Q1_DATA" | tail -1)
    if [ "$mount_target" = "/" ] && [ -z "${Q1_ALLOW_ROOT_VOLUME:-}" ]; then
        die "$Q1_DATA is on the root volume; mount the large data volume there (or set Q1_ALLOW_ROOT_VOLUME=1)"
    fi
    total_gb=$(df --output=size -BG "$Q1_DATA" | tail -1 | tr -dc '0-9')
    free_gb=$(df --output=avail -BG "$Q1_DATA" | tail -1 | tr -dc '0-9')
    log "data volume $mount_target: ${total_gb} GB total, ${free_gb} GB free"
    [ "$total_gb" -ge "$MIN_DATA_GB" ] || die "data volume is ${total_gb} GB, need at least ${MIN_DATA_GB} GB for all four datasets (Q1_MIN_DATA_GB to override; 10 is enough for the smoke test)"
}

step_tools_dir() {
    run sudo mkdir -p "$Q1_TOOLS"
    run sudo chown "$(id -u):$(id -g)" "$Q1_TOOLS"
}

step_first_boot_wait() {
    if [ -n "${Q1_DRYRUN:-}" ]; then
        log "dry run: cloud-init and apt lock waits skipped"
        return 0
    fi
    wait_for_cloud_init
    wait_for_apt
}

step_system_packages() { ensure_build_packages; }
step_cmake() { ensure_cmake; }
step_boost() { ensure_boost; }
step_faiss() { ensure_faiss; }
step_postgres() { ensure_postgres; }

step_pg_cluster() {
    guard_path "$Q1_PGDATA"
    if [ -n "${Q1_DRYRUN:-}" ] || [ ! -f "$Q1_PGDATA/PG_VERSION" ]; then
        run mkdir -p "$Q1_PGDATA"
        run "$PG_BIN/initdb" -D "$Q1_PGDATA" -U postgres --auth=trust
        local ram_mb cpus workers
        ram_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 8192)
        cpus=$(nproc 2>/dev/null || echo 8)
        workers=$((cpus > 8 ? 7 : cpus - 1))
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
}

step_python_env() {
    local lock
    if [ "$CI" -eq 1 ]; then
        lock="$root/requirements-ci.lock"
    else
        lock="$root/requirements-q1.lock"
    fi
    run "$PYTHON_BIN" -m venv "$root/.venv"
    [ -f "$lock" ] || die "missing $lock (generate with uv pip compile; see harness/deploy/README or EC2_FIRST_RUN.md)"
    run "$root/.venv/bin/pip" install --quiet --require-hashes -r "$lock"
}

step_encode_env() {
    if [ "$CI" -eq 1 ]; then
        mark_na "--ci"
        return 0
    fi
    run "$PYTHON_BIN" -m venv "$root/.venv-encode"
    run "$root/.venv-encode/bin/pip" install --quiet --require-hashes -r "$root/requirements-encode.lock"
}

step_drivers() {
    CMAKE_BIN="$CMAKE_BIN" BOOST_ROOT="$BOOST_ROOT" FAISS_ROOT="$FAISS_ROOT" run bash "$here/build_linux.sh"
}

step_pg_start() {
    Q1_DATA="$Q1_DATA" Q1_PGDATA="$Q1_PGDATA" Q1_PG_PORT="$Q1_PG_PORT" PG_BIN="$PG_BIN" run bash "$here/pg.sh" start
}

step_preflight() {
    local -a datasets=() extra=() cmd=("$root/.venv/bin/python" -m harness preflight)
    [ "$SETUP_DATASETS" = none ] || read -ra datasets <<< "$SETUP_DATASETS"
    [ "${Q1_ALLOW_SHORTFALL:-}" != "1" ] || extra=(--allow-shortfall)
    local dsn="host=127.0.0.1 port=$Q1_PG_PORT user=postgres"
    if [ -n "${Q1_PREFLIGHT_CMD:-}" ]; then
        read -ra cmd <<< "$Q1_PREFLIGHT_CMD"
    elif [ -n "${Q1_DRYRUN:-}" ]; then
        PYTHONPATH="$root" run "${cmd[@]}" --datasets ${datasets[@]+"${datasets[@]}"} --data-dir "$Q1_DATA" \
            --dsn "$dsn" ${extra[@]+"${extra[@]}"}
        return 0
    fi
    PYTHONPATH="$root" "${cmd[@]}" --datasets ${datasets[@]+"${datasets[@]}"} --data-dir "$Q1_DATA" \
        --dsn "$dsn" ${extra[@]+"${extra[@]}"} | tee "$STATE_DIR/preflight.out"
    local warned
    warned=$(grep -c '^PREFLIGHT WARN' "$STATE_DIR/preflight.out" || true)
    if [ "${warned:-0}" -gt 0 ]; then
        mark_warn "$warned shortfall(s): $(grep -m1 '^PREFLIGHT WARN' "$STATE_DIR/preflight.out" | cut -c17-140)"
    fi
}

# Setup, the builds and the smoke test must leave no untracked files: every target refuses a dirty tree.
step_clean_tree() {
    if [ -n "${Q1_DRYRUN:-}" ] && [ -z "${Q1_CHECK_CLEAN_TREE:-}" ]; then
        mark_na "dry run"
        return 0
    fi
    local dirty
    dirty=$(git -C "$root" status --short)
    [ -z "$dirty" ] || die "git status --short is not empty after setup: $(printf '%s' "$dirty" | head -n5 | tr '\n' ';')"
}

# Last on purpose: a missing GPU must not hide whether the builds, PostgreSQL and pgvector work.
step_gpu() {
    if [ "$CI" -eq 1 ]; then
        mark_na "--ci"
        return 0
    fi
    local -a cmd=("$root/.venv/bin/python" -m harness.deploy.gpu_check)
    if [ -n "${Q1_GPU_CHECK_CMD:-}" ]; then
        read -ra cmd <<< "$Q1_GPU_CHECK_CMD"
    elif [ -n "${Q1_DRYRUN:-}" ]; then
        mark_na "dry run"
        return 0
    fi
    local rc=0
    PYTHONPATH="$root" "${cmd[@]}" || rc=$?
    case "$rc" in
        0) ;;
        10) die "no NVIDIA driver or GPU found" ;;
        11) die "NVIDIA driver too old for the installed torch CUDA build" ;;
        12) die "torch missing or a CPU-only build" ;;
        13) die "NVIDIA driver present but torch cannot use the GPU" ;;
        *) die "GPU check failed (exit code $rc)" ;;
    esac
}

# --- run -------------------------------------------------------------------------------------------------
step preconditions
hard_stop preconditions
step data_volume preconditions
hard_stop data_volume
step tools_dir preconditions
step first_boot_wait preconditions
step system_packages first_boot_wait
step cmake system_packages
step boost system_packages tools_dir
step faiss system_packages tools_dir cmake
step postgres system_packages
step pg_cluster postgres data_volume
step python_env system_packages
step encode_env python_env
step drivers cmake boost faiss
step pg_start pg_cluster
step preflight pg_start python_env
step clean_tree
step gpu python_env
finish
