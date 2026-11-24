#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 x86_64 instance for the evaluation. Safe to rerun.
#   sudo -E bash scripts/ec2_bootstrap.sh [--role prepare|run] [--datasets LIST] [--dry-run]
#       [--only STEP ...] [--skip STEP ...] [--from STEP] [--clean] [--check] [--no-data] [--no-smoke]
#
# Run it through sudo from the normal user (-E keeps the RECAL_* variables). System steps (apt, sysctl, the
# data directories) run as root; everything else (the harness setup with its PostgreSQL cluster, virtual
# environments, builds, data fetch, tests) runs as the invoking user (SUDO_USER), so nothing in the user's tree
# is created as root. At the end the repository, the data directory and the build and queue directories are
# owned by that user.
#
# Roles: prepare (default) fetches the data and builds the shared artifacts afterwards (make prepare); run skips
# the data fetch and every build of data or indexes: it pulls the artifacts of its shard from the store.
#
# Options:
#   --datasets LIST  datasets this instance prepares, comma separated (spacev, triviaqa, medrag, tripclick;
#                    default all four). Sets the disk requirement (the free space of the data volume must cover
#                    the sum of their needs plus 20%), the harness volume check and the data fetch.
#   --from STEP      resume at STEP (earlier steps are skipped, STEP is redone)
#   --clean          remove the state file, .venv-core, .venv-ann and the third-party builds ($TP), then exit
#   --check          verify the result of every step without changing anything (run it as the normal user, no
#                    sudo): one PASS or FAIL line per check, with the fix for each FAIL; exit 1 if any failed
#
# Steps, in order. A step runs only if the previous ones passed; on a failure the script prints the step, the
# failed command and the last 30 lines of its output, then the summary, and stops.
#   preconditions  checks Linux and the disk requirement; creates the data directories
#   base           harness/deploy/setup_instance.sh: system packages, pinned CMake, Boost and FAISS, PostgreSQL 16
#                  with pgvector 0.8.0 (isolated cluster), the harness venv, the Q1 drivers, preflight
#   limits         raises fs.aio-max-nr (DiskANN and the replay open many AIO contexts)
#   venvs          .venv-core (Python 3.11) and .venv-ann from envs/ (diskannpy in .venv-ann)
#   core           builds the ReCal core and the cache core
#   aker           the vendored Aker library, standard and Proximity modes, against the pinned Boost and FAISS
#   qvcache        the authors' QVCache artifact (pinned commit in harness/deploy/pins.env), built natively against
#                  the .venv-ann interpreter so DiskANN and pgvector stay in the same process (no extra hop)
#   links          points datasets/{aker,medrag,tripclick} and simulations/approximate_hits/results at the data volume
#   store          boto3 for the S3 store (credentials from the instance role; set RECAL_STORE=s3://bucket/prefix)
#   data           scripts/fetch_data.sh for the selected datasets (skipped with --no-data and with --role run)
#   smoke          tests, matrix dry run, a small ReCal cell with every ablation, library and backend checks
#                  (skipped with --no-smoke)
#
# State: $RECAL_TMP/recal_bootstrap/state records each step as started or done with the code commit. A rerun skips
# done steps of the same commit; a step left started (interrupted or failed) is cleaned (partial virtual
# environments and build directories are removed) and redone. --dry-run neither reads nor writes the state.
#
# Environment:
#   DATA_DIR         data volume (default /data); datasets and answer tables go to $DATA_DIR/datasets
#   RESULTS_DIR      results (default $DATA_DIR/results); linked to simulations/approximate_hits/results
#   TP               third-party build prefix (default $RECAL_THIRD_PARTY, else $DATA_DIR/third_party); Aker and
#                    QVCache build there
#   RECAL_TMP        scratch and state directory (default /tmp); the state is in $RECAL_TMP/recal_bootstrap
#   RECAL_QUEUE_DIR  job queue directory (default /tmp/mx)
#   RECAL_DRIVE_*    dataset Drive folder links for the data step (or .env.local in the checkout)
#   PYTHON311        interpreter for .venv-core (default python3.11; installed from the deadsnakes PPA if missing)
set -uo pipefail
# nothing may prompt: git, pip and apt fail instead of waiting for input
export GIT_TERMINAL_PROMPT=0 PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/data}"
RESULTS_DIR="${RESULTS_DIR:-$DATA_DIR/results}"
TP="${TP:-${RECAL_THIRD_PARTY:-$DATA_DIR/third_party}}"
PYTHON311="${PYTHON311:-python3.11}"
RECAL_TMP="${RECAL_TMP:-/tmp}"
RECAL_QUEUE_DIR="${RECAL_QUEUE_DIR:-/tmp/mx}"
STATE_DIR="$RECAL_TMP/recal_bootstrap"
ALL_DATASETS="spacev triviaqa medrag tripclick"
STEPS=(preconditions base limits venvs core aker qvcache links store data smoke)
CORE_MODULES="numpy scipy sklearn faiss pybind11 gdown boto3 psycopg"
ANN_MODULES="numpy scipy faiss diskannpy"

DRY=0
ROLE=prepare
SMOKE=1
DATA=1
CLEAN=0
CHECK=0
FROM=""
DATASETS="$ALL_DATASETS"
ONLY=()
SKIP=()
RUN_USER="${SUDO_USER:-}"
COMMIT="${RECAL_BOOTSTRAP_COMMIT:-}"
FAIL_FILE=/dev/null
LOG_DIR=""
REQUIRED_GB=0
summary=()

log() { printf '[bootstrap] %s\n' "$*"; }

# run CMD...: print in a dry run; otherwise run and note the failed command for the failure report.
run() {
    if [ "$DRY" -eq 1 ]; then printf '[dry-run] %s\n' "$*"; return 0; fi
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -ne 0 ]; then printf '%s\n' "$*" > "$FAIL_FILE"; fi
    return "$rc"
}

# apt_run CMD...: root apt call that never prompts (debconf, needrestart).
apt_run() { run env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 "$@"; }
APT_OPTS=(-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

# preseed_mkl: answer the Intel MKL debconf question "use libmkl_rt.so as the default BLAS/LAPACK" with no.
preseed_mkl() {
    run bash -c "echo 'libmkl-rt libmkl-rt/use-as-default-blas-lapack boolean false' | debconf-set-selections"
}

# as_user CMD...: run as the invoking user when the script runs as root through sudo.
as_user() {
    if [ "$DRY" -eq 1 ]; then printf '[dry-run] (as %s) %s\n' "${RUN_USER:-current user}" "$*"; return 0; fi
    if [ "$(id -u)" -eq 0 ] && [ -n "$RUN_USER" ]; then
        run sudo -u "$RUN_USER" -H -E "$@"
    else
        run "$@"
    fi
}

selected() {
    local s
    for s in "${SKIP[@]:-}"; do [ "$s" = "$1" ] && return 1; done
    [ "${#ONLY[@]}" -eq 0 ] && return 0
    for s in "${ONLY[@]}"; do [ "$s" = "$1" ] && return 0; done
    return 1
}

# --- disk requirement ------------------------------------------------------------------------------------
# Per dataset, in 0.1 GB: base vectors + DiskANN index + pgvector table and index + answer tables (README sizes
# table) + scratch for conversions and builds (2 x base vectors; an estimate).
dataset_tenths() {
    case "$1" in
        spacev) echo $((10 + 40 + 160 + 14 + 20)) ;;
        triviaqa) echo $((310 + 340 + 0 + 14 + 620)) ;;
        medrag) echo $((740 + 800 + 1650 + 5 + 1480)) ;;
        tripclick) echo $((47 + 50 + 100 + 5 + 94)) ;;
        laion) echo $((198 + 193 + 250 + 420 + 5 + 396)) ;;
        *) return 1 ;;
    esac
}

parse_datasets() {
    local list="$1" d out=""
    for d in ${list//,/ }; do
        dataset_tenths "$d" > /dev/null || { echo "unknown dataset: $d (spacev, triviaqa, medrag, tripclick, laion)" >&2; return 1; }
        case " $out " in *" $d "*) ;; *) out="$out $d" ;; esac
    done
    [ -n "$out" ] || { echo "--datasets needs at least one dataset" >&2; return 1; }
    DATASETS="${out# }"
}

# Required free space in GB: the sum of the selected datasets plus 20%, rounded up.
required_gb() {
    local total=0 d
    for d in $DATASETS; do total=$((total + $(dataset_tenths "$d"))); done
    echo $(((total * 12 + 99) / 100))
}

# Datasets the harness preflight can size (it knows spacev and sphere, TriviaQA's base); "none" skips its sizing.
harness_datasets() {
    local d out=""
    for d in $DATASETS; do
        case "$d" in spacev) out="$out spacev" ;; triviaqa) out="$out sphere" ;; esac
    done
    echo "${out:-none}" | sed 's/^ //'
}

fetch_flags() {
    local d flags=""
    if [ "$DATASETS" = "$ALL_DATASETS" ]; then echo "--all"; return; fi
    for d in $DATASETS; do
        case "$d" in
            triviaqa) flags="$flags --triviaqa --sphere" ;;
            *) flags="$flags --$d" ;;
        esac
    done
    echo "${flags# }"
}

existing_parent() {
    local p="$1"
    while [ ! -e "$p" ] && [ "$p" != "/" ]; do p="$(dirname "$p")"; done
    echo "$p"
}

check_disk() {
    local d t free_gb
    for d in $DATASETS; do
        t=$(dataset_tenths "$d")
        log "disk need $d: $((t / 10)).$((t % 10)) GB"
    done
    log "disk requirement: $REQUIRED_GB GB free on $DATA_DIR for: $DATASETS (sum plus 20%)"
    if [ "$DRY" -eq 1 ]; then return 0; fi
    free_gb=$(df --output=avail -BG "$(existing_parent "$DATA_DIR")" | tail -n 1 | tr -dc '0-9')
    log "free on the data volume: ${free_gb:-?} GB"
    if [ -z "$free_gb" ] || [ "$free_gb" -lt "$REQUIRED_GB" ]; then
        log "not enough free space on the data volume: need $REQUIRED_GB GB for $DATASETS"
        return 1
    fi
}

# --- ownership ---------------------------------------------------------------------------------------------
# Refuse system directories as targets of a recursive chown or removal.
unsafe_path() {
    case "$1" in
        "" | / | /tmp | /var | /usr | /etc | /bin | /sbin | /lib | /boot | /opt | /home | /root | /proc | /sys | /dev | /mnt | /srv) return 0 ;;
        /*) return 1 ;;
        *) return 0 ;;
    esac
}

make_dirs() {
    run mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$TP" "$RECAL_QUEUE_DIR" "$STATE_DIR"
    if [ -n "$RUN_USER" ]; then
        run chown "$RUN_USER:" "$DATA_DIR" "$RESULTS_DIR" "$TP" "$RECAL_QUEUE_DIR" "$STATE_DIR"
    fi
}

finish_ownership() {
    local p
    if [ "$DRY" -eq 1 ] || [ "$(id -u)" -ne 0 ] || [ -z "$RUN_USER" ]; then return 0; fi
    for p in "$root" "$DATA_DIR" "$RESULTS_DIR" "$TP" "$RECAL_QUEUE_DIR" "$STATE_DIR"; do
        if unsafe_path "$p"; then log "ownership of $p left as it is (system directory)"; continue; fi
        if [ -e "$p" ]; then chown -R "$RUN_USER:" "$p"; fi
    done
    return 0
}

# --- steps -------------------------------------------------------------------------------------------------
# shellcheck source=bootstrap_toolchain.sh
. "$root/scripts/bootstrap_toolchain.sh"

step_preconditions() {
    if [ "$DRY" -eq 0 ]; then
        [ "$(uname -s)" = "Linux" ] || { log "not Linux"; return 1; }
        grep -q 'VERSION_ID="24.04"' /etc/os-release || log "warning: not Ubuntu 24.04"
    fi
    check_disk || return 1
    make_dirs
}

step_base() {
    Q1_DATA="${Q1_DATA:-$DATA_DIR/q1}" Q1_MIN_DATA_GB="$REQUIRED_GB" Q1_SETUP_DATASETS="$(harness_datasets)" \
        as_user bash "$root/harness/deploy/setup_instance.sh"
}

step_limits() {
    run sysctl -w fs.aio-max-nr=1048576
    run bash -c "echo 'fs.aio-max-nr = 1048576' > /etc/sysctl.d/99-recal-aio.conf"
}

step_venvs() {
    if ! command -v "$PYTHON311" > /dev/null 2>&1; then
        apt_run add-apt-repository -y ppa:deadsnakes/ppa
        apt_run apt-get "${APT_OPTS[@]}" update -qq
        apt_run apt-get "${APT_OPTS[@]}" install -y -qq python3.11 python3.11-venv python3.11-dev
    fi
    as_user "$PYTHON311" -m venv "$root/.venv-core"
    as_user "$root/.venv-core/bin/pip" install --quiet -r "$root/envs/core.txt" pybind11 pytest
    as_user "$PYTHON311" -m venv "$root/.venv-ann"
    as_user "$root/.venv-ann/bin/pip" install --quiet -r "$root/envs/ann.txt" pytest
    if [ "$DRY" -eq 0 ]; then
        # shellcheck disable=SC2086
        run python_imports "$root/.venv-core/bin/python" $CORE_MODULES
        # shellcheck disable=SC2086
        run python_imports "$root/.venv-ann/bin/python" $ANN_MODULES
    fi
}

clean_venvs() { run rm -rf "$root/.venv-core" "$root/.venv-ann"; }

step_core() {
    as_user sh "$root/recal/core/build.sh"
    as_user sh "$root/simulations/approximate_hits/cachecore/build.sh"
}

clean_core() {
    run rm -f "$root"/recal/core/_recal_core*.so* "$root"/simulations/approximate_hits/cachecore/_cachecore*.so*
}

step_aker() {
    # the step installs what the harness does not (rsync, used by the Aker build), then checks every tool and library
    apt_run apt-get "${APT_OPTS[@]}" install -y -qq rsync libopenblas-dev
    if [ "$DRY" -eq 0 ]; then aker_missing | require_missing aker; fi
    as_user mkdir -p "$TP"
    local boost
    boost="$(boost_prefix)"
    # Boost: the harness copy, or the system packages (prefix /usr); the link follows what the harness installed
    if [ -L "$TP/boost-install" ] || [ ! -e "$TP/boost-install" ]; then as_user ln -sfn "$boost" "$TP/boost-install"; fi
    ensure_aker_faiss
    if [ "$DRY" -eq 0 ]; then faiss_missing | require_missing aker; fi
    TP="$TP" AKER_SRC=vendored as_user env "PATH=$(tool_path)" "CMAKE_PREFIX_PATH=$TP/faiss-install:$boost" \
        sh "$root/simulations/approximate_hits/aker_real/build.sh"
}

clean_aker() {
    run rm -rf "$TP/aker_vendored/build" "$TP/aker_vendored_prox/build" "$TP/src/faiss-shared-build"
    if [ ! -f "$TP/faiss-install/.installed-$FAISS_COMMIT-shared" ]; then run rm -rf "$TP/faiss-install"; fi
}

step_qvcache() {
    # a complete build whose module imports is only registered again: no packages, no checks, no rebuild
    if [ "$DRY" -eq 0 ] && qvcache_module > /dev/null && register_qvcache; then
        log "qvcache: build already complete, module registered"
        return 0
    fi
    preseed_mkl
    # shellcheck disable=SC2086
    apt_run apt-get "${APT_OPTS[@]}" install -y -qq $QVCACHE_PACKAGES
    if [ "$DRY" -eq 0 ]; then
        { qvcache_missing; apt_missing $QVCACHE_PACKAGES; } | require_missing qvcache
    fi
    local src="$TP/qvcache" cmake_path
    cmake_path="$(cmake_bin)"
    if [ ! -d "$src/.git" ]; then
        as_user mkdir -p "$TP"
        as_user git init -q "$src"
        as_user git -C "$src" remote add origin "$QVCACHE_REPO"
    fi
    as_user git -C "$src" fetch -q --depth 1 origin "$QVCACHE_COMMIT"
    as_user git -C "$src" checkout -q "$QVCACHE_COMMIT"
    as_user "$root/.venv-ann/bin/pip" install --quiet pybind11 numpy
    as_user env "PATH=$(tool_path)" "$cmake_path" -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE="$root/.venv-ann/bin/python"
    as_user env "PATH=$(tool_path)" "$cmake_path" --build "$src/build" -j "$(nproc 2>/dev/null || echo 8)"
    # make the binding importable from .venv-ann
    if [ "$DRY" -eq 1 ]; then
        printf '[dry-run] register the qvcache module found under %s in .venv-ann (qvcache.pth) and import it\n' "$src"
    else
        register_qvcache
    fi
}

clean_qvcache() { run rm -rf "$TP/qvcache/build"; }

step_links() {
    local rel
    for rel in datasets/aker datasets/medrag datasets/tripclick; do
        as_user mkdir -p "$DATA_DIR/$rel"
        [ -e "$root/$rel" ] || as_user ln -s "$DATA_DIR/$rel" "$root/$rel"
    done
    as_user mkdir -p "$RESULTS_DIR"
    # make prepare writes under <repo>/results; keep it on the data volume
    [ -e "$root/results" ] || as_user ln -s "$RESULTS_DIR" "$root/results"
    [ -e "$root/simulations/approximate_hits/results" ] || \
        as_user ln -s "$RESULTS_DIR" "$root/simulations/approximate_hits/results"
}

step_store() {
    # boto3 is in envs/core.txt; install it only when an older venv lacks it
    if ! as_user "$root/.venv-core/bin/python" -c "import boto3" > /dev/null 2>&1; then
        as_user "$root/.venv-core/bin/pip" install --quiet boto3
    fi
    if [ "$DRY" -eq 0 ] && [ -z "${RECAL_STORE:-}${RECAL_S3_BUCKET:-}" ]; then
        log "note: set RECAL_STORE before make prepare / make run"
    fi
}

step_data() {
    [ "$DATA" -eq 1 ] || { log "data fetch skipped (--no-data)"; return 0; }
    local py="$root/.venv-core/bin/python"
    # the fetch needs gdown and numpy in the core venv (not the system python)
    if ! as_user "$py" -c "import gdown, numpy" > /dev/null 2>&1; then
        as_user "$root/.venv-core/bin/pip" install --quiet gdown
    fi
    # shellcheck disable=SC2046
    RECAL_REPO="$root" FETCH_PY="$py" as_user sh "$root/scripts/fetch_data.sh" $(fetch_flags)
}

step_smoke() {
    [ "$SMOKE" -eq 1 ] || { log "smoke test skipped (--no-smoke)"; return 0; }
    as_user "$root/.venv-core/bin/python" -m pytest -q "$root/tests/recal" "$root/tests/test_cachecore.py"
    as_user "$root/.venv-core/bin/python" -m scripts.gen_matrix --config "$root/experiments/matrix.example.json" --dry-run
    as_user env PYTHONPATH="$root" "LD_LIBRARY_PATH=$(aker_ld_library_path)" "$root/.venv-core/bin/python" \
        -m scripts.smoke_cells --out "$RESULTS_DIR/smoke"
}

# --- --check ---------------------------------------------------------------------------------------------
CHECK_FAILED=0

# check_line NAME FIX CMD...: PASS when the command succeeds, else FAIL with the fix and the last line the
# command printed (what is missing). Nothing is changed.
check_line() {
    local name="$1" fix="$2" out
    shift 2
    if out="$("$@" 2> /dev/null)"; then
        printf 'PASS  %s\n' "$name"
    else
        out="$(printf '%s\n' "$out" | tail -n 1)"
        printf 'FAIL  %s%s -> %s\n' "$name" "${out:+ (missing: $out)}" "$fix"
        CHECK_FAILED=1
    fi
}

# python_imports PYTHON MODULE...: prints the modules that do not import
python_imports() {
    local py="$1"
    shift
    "$py" -I -c "
import importlib, sys
missing = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
print(' '.join(missing))
sys.exit(1 if missing else 0)" "$@"
}

aker_filter_line_present() {
    "$root/.venv-core/bin/python" -I -c "
import sys
data = open(sys.argv[1], 'rb').read()
sys.exit(0 if b'FILTER faiss_M=' in data else 1)" "$TP/aker_vendored/build/lib/libaker.so"
}

shims_load() {
    LD_LIBRARY_PATH="$(aker_ld_library_path)" "$root/.venv-core/bin/python" -I -c "
import ctypes, sys
d = sys.argv[1]
for n in ('libakershim.so', 'libakershim_prox.so'):
    ctypes.CDLL(d + '/' + n)" "$root/simulations/approximate_hits/aker_real"
}

pg_has_vector() {
    "$root/.venv-core/bin/python" -I -c "
import psycopg, sys
with psycopg.connect(host='127.0.0.1', port=int(sys.argv[1]), user='postgres', connect_timeout=5) as c:
    row = c.execute(\"SELECT 1 FROM pg_available_extensions WHERE name = 'vector'\").fetchone()
sys.exit(0 if row else 1)" "${Q1_PG_PORT:-54320}"
}

# with_timeout CMD...: bounded when timeout(1) exists
with_timeout() {
    if command -v timeout > /dev/null 2>&1; then timeout 30 "$@"; else "$@"; fi
}

# the store through recal/store.py (boto3, as the pipeline uses it); `aws s3 ls` fails on an empty prefix
store_reachable() {
    local store="${RECAL_STORE:-}"
    case "$store" in
        s3://*)
            with_timeout env PYTHONPATH="$root" "$root/.venv-core/bin/python" -c "
import sys
from recal import store
store.open_store(sys.argv[1]).keys('.reachable')" "$store" > /dev/null
            ;;
        "") return 1 ;;
        *) [ -d "$store" ] && [ -w "$store" ] ;;
    esac
}

drive_variables_set() {
    "$root/.venv-core/bin/python" -I -c "
import sys
sys.path.insert(0, sys.argv[1])
from scripts import data_fetch
names = {'spacev': [], 'triviaqa': ['triviaqa'], 'medrag': ['medrag', 'medrag100k'], 'tripclick': ['tripclick']}
folders = sorted({f for d in sys.argv[3:] for f in names[d]})
env = data_fetch.drive_env(__import__('pathlib').Path(sys.argv[1]))
missing = data_fetch.missing_variables(folders, env)
print(' '.join(missing))
sys.exit(1 if missing else 0)" "$root" x $DATASETS
}

disk_has_space() {
    local free
    free=$(df --output=avail -BG "$(existing_parent "$DATA_DIR")" | tail -n 1 | tr -dc '0-9')
    [ -n "$free" ] && [ "$free" -ge "$REQUIRED_GB" ]
}

owned_by_me() {
    local p
    for p in "$@"; do [ -d "$p" ] && [ -w "$p" ] || return 1; done
}

run_checks() {
    local core="$root/.venv-core/bin/python" ann="$root/.venv-ann/bin/python" pgbin="/usr/lib/postgresql/${PG_MAJOR:-16}/bin"
    CHECK_FAILED=0
    log "check: datasets $DATASETS, data volume $DATA_DIR (run as the normal user, not through sudo)"
    check_line "disk: $REQUIRED_GB GB free on $DATA_DIR" "free space on the volume (bash scripts/ec2_bootstrap.sh --datasets ...)" disk_has_space
    check_line "directories writable by $(id -un)" "sudo -E bash scripts/ec2_bootstrap.sh --from preconditions (or sudo chown -R $(id -un): $DATA_DIR $root)" \
        owned_by_me "$DATA_DIR" "$RESULTS_DIR" "$TP" "$RECAL_QUEUE_DIR" "$root"
    check_line "tool cmake >= $MIN_CMAKE" "rerun the base step (sudo -E ... --from base)" find_cmake
    check_line "tool $AKER_CXX >= $MIN_GCC" "sudo apt-get install build-essential" test -z "$(compiler_missing "$AKER_CXX")"
    check_line "tools make git rsync taskset" "sudo apt-get install make git rsync util-linux" test -z "$(commands_missing make git rsync taskset)"
    check_line "tool aws (store)" "install the AWS CLI v2" command -v aws
    check_line "postgresql binaries ($pgbin)" "rerun the base step (sudo -E ... --from base)" test -x "$pgbin/pg_ctl" -a -x "$pgbin/psql"
    # shellcheck disable=SC2086
    check_line ".venv-core imports $CORE_MODULES" "rerun the venvs step (sudo -E ... --from venvs)" \
        python_imports "$core" $CORE_MODULES
    # shellcheck disable=SC2086
    check_line ".venv-ann imports $ANN_MODULES" "rerun the venvs step (sudo -E ... --from venvs)" \
        python_imports "$ann" $ANN_MODULES
    check_line "ReCal core and cache core import" "rerun the core step (sudo -E ... --from core)" \
        env PYTHONPATH="$root" "$core" -c "import recal.core, simulations.approximate_hits.cachecore"
    check_line "Aker shims (standard and Proximity) load" "rerun the aker step (sudo -E ... --from aker)" shims_load
    check_line "Aker FILTER line is compiled in" "rerun the aker step (sudo -E ... --from aker)" aker_filter_line_present
    check_line "import qvcache in .venv-ann" "rerun the qvcache step (sudo -E ... --from qvcache)" qvcache_imports
    check_line "fs.aio-max-nr >= 1048576" "sudo -E bash scripts/ec2_bootstrap.sh --from limits" \
        test "$(cat /proc/sys/fs/aio-max-nr 2> /dev/null || echo 0)" -ge 1048576
    check_line "PostgreSQL answers on port ${Q1_PG_PORT:-54320} with pgvector available" "make q1-pg-start (make prepare starts it too)" pg_has_vector
    check_line "store ${RECAL_STORE:-RECAL_STORE unset} reachable" "set RECAL_STORE (s3://bucket/prefix) and check the instance role (aws s3 ls)" store_reachable
    check_line "Drive variables for $DATASETS set" "set the missing RECAL_DRIVE_* variables in the environment or .env.local" drive_variables_set
    check_line "links: datasets/aker and results" "rerun the links step (sudo -E ... --from links)" test -e "$root/datasets/aker" -a -e "$root/results"
    if [ "$CHECK_FAILED" -eq 0 ]; then log "check: all passed"; else log "check: some checks failed"; fi
    return "$CHECK_FAILED"
}

# --- state -------------------------------------------------------------------------------------------------
state_line() { grep "^$1 " "$STATE_DIR/state" 2> /dev/null | tail -n 1; }

state_set() {
    local tmp="$STATE_DIR/state.tmp"
    mkdir -p "$STATE_DIR"
    { grep -v "^$1 " "$STATE_DIR/state" 2> /dev/null; printf '%s %s %s\n' "$1" "$2" "$COMMIT"; } > "$tmp"
    mv -f "$tmp" "$STATE_DIR/state"
}

# --- run ---------------------------------------------------------------------------------------------------
print_summary() {
    printf '\n[bootstrap] summary\n'
    printf '  %s\n' "${summary[@]}"
}

report_failure() {
    local name="$1" log_file="$2"
    log "step $name FAILED"
    if [ -s "$FAIL_FILE" ]; then log "failed command: $(cat "$FAIL_FILE")"; fi
    log "last 30 lines of output:"
    tail -n 30 "$log_file"
}

mark_not_run_after() {
    local s seen=0
    for s in "${STEPS[@]}"; do
        if [ "$s" = "$1" ]; then seen=1; continue; fi
        if [ "$seen" -eq 1 ]; then summary+=("$s: NOT RUN"); fi
    done
}

# run_steps: run the selected steps in order; stop at the first failure (return 1).
run_steps() {
    local name rc line forced reached=0
    summary=()
    if [ "$DRY" -eq 1 ]; then LOG_DIR="$(mktemp -d)"; else LOG_DIR="$STATE_DIR/log"; mkdir -p "$LOG_DIR"; fi
    FAIL_FILE="$LOG_DIR/failed_command"
    for name in "${STEPS[@]}"; do
        forced=0
        if [ -n "$FROM" ] && [ "$reached" -eq 0 ]; then
            if [ "$name" != "$FROM" ]; then summary+=("$name: SKIPPED (before --from)"); continue; fi
            reached=1
            forced=1
        fi
        if ! selected "$name"; then summary+=("$name: NOT RUN"); continue; fi
        if [ "$DRY" -eq 0 ]; then
            line="$(state_line "$name")"
            if [ "$forced" -eq 0 ] && [ "$line" = "$name done $COMMIT" ]; then
                summary+=("$name: SKIPPED (done)")
                continue
            fi
            case "$line" in
                "$name started "*)
                    log "step $name was interrupted: cleaning partial output"
                    if declare -F "clean_$name" > /dev/null; then "clean_$name"; fi
                    ;;
            esac
            state_set "$name" started
        fi
        log "step $name"
        : > "$FAIL_FILE"
        (
            set -eE
            trap '[ -s "$FAIL_FILE" ] || printf "%s\n" "$BASH_COMMAND" > "$FAIL_FILE"' ERR
            cd "$root"
            "step_$name"
        ) 2>&1 | tee "$LOG_DIR/$name.log"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -ne 0 ]; then
            summary+=("$name: FAIL")
            report_failure "$name" "$LOG_DIR/$name.log"
            mark_not_run_after "$name"
            return 1
        fi
        if [ "$DRY" -eq 0 ]; then state_set "$name" done; fi
        summary+=("$name: PASS")
    done
    return 0
}

do_clean() {
    local p
    for p in "$TP" "$STATE_DIR"; do
        if unsafe_path "$p"; then log "refusing to remove $p"; return 1; fi
    done
    run rm -rf "$STATE_DIR" "$root/.venv-core" "$root/.venv-ann" "$TP"
}

main() {
    local original_args="$*"
    while [ $# -gt 0 ]; do
        case "$1" in
            --role) shift; ROLE="${1:-}"; case "$ROLE" in prepare|run) ;; *) echo "--role must be prepare or run" >&2; return 2 ;; esac ;;
            --datasets) shift; parse_datasets "${1:-}" || return 2 ;;
            --dry-run) DRY=1 ;;
            --no-data) DATA=0 ;;
            --no-smoke) SMOKE=0 ;;
            --clean) CLEAN=1 ;;
            --check) CHECK=1 ;;
            --from)
                shift; FROM="${1:-}"
                case " ${STEPS[*]} " in *" $FROM "*) ;; *) echo "--from needs one of: ${STEPS[*]}" >&2; return 2 ;; esac
                ;;
            --only) shift; ONLY+=("${1:-}") ;;
            --skip) shift; SKIP+=("${1:-}") ;;
            *) echo "unknown argument: $1" >&2; return 2 ;;
        esac
        shift
    done
    if [ "$ROLE" = run ]; then DATA=0; fi
    REQUIRED_GB="$(required_gb)"

    if [ "$CHECK" -eq 1 ]; then
        if [ -z "$COMMIT" ]; then COMMIT=check; fi
        run_checks
        return $?
    fi
    if [ "$DRY" -eq 0 ]; then
        if [ "$(id -u)" -ne 0 ]; then
            log "run as root through sudo from the normal user: sudo -E bash scripts/ec2_bootstrap.sh $original_args"
            return 1
        fi
        if [ -z "$RUN_USER" ] || [ "$RUN_USER" = root ]; then
            log "SUDO_USER is not set: run it through sudo from the normal user: sudo -E bash scripts/ec2_bootstrap.sh $original_args"
            return 1
        fi
    fi
    if [ -z "$COMMIT" ]; then
        COMMIT="$(git -c safe.directory="$root" -C "$root" rev-parse HEAD 2> /dev/null || echo unknown)"
    fi

    if [ "$CLEAN" -eq 1 ]; then
        do_clean
        return $?
    fi
    # a plain call: under `run_steps || ...` bash ignores set -e inside the steps
    run_steps
    local rc=$?
    print_summary
    finish_ownership
    return "$rc"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
    exit $?
fi
