#!/usr/bin/env bash
# Toolchain lookup and requirement checks for the aker and qvcache steps of ec2_bootstrap.sh (sourced).
#
# The harness setup (harness/deploy/setup_instance.sh) puts CMake in <repo>/.venv-build (or uses a system one),
# builds Boost under $Q1_TOOLS/boost only when the system Boost is older than MIN_BOOST (otherwise the system
# packages in /usr are used) and builds FAISS as a static archive under $Q1_TOOLS/faiss. None of that is on the
# PATH of the user running the builds. Aker needs a shared libfaiss and Boost shared libraries, so the aker step
# builds its own shared FAISS (same pinned commit) into $TP/faiss-install and points the build at the Boost the
# harness installed.
#
# Expects from the caller: root, TP, and the log, run and as_user helpers. Sources harness/deploy/pins.env.
# shellcheck disable=SC2154

# shellcheck source=../harness/deploy/pins.env
. "$root/harness/deploy/pins.env"

TOOLS_DIR="${Q1_TOOLS:-/opt/q1}"
SYSTEM_BOOST_PREFIX="${SYSTEM_BOOST_PREFIX:-/usr}"
SYSTEM_LIB_DIRS="${SYSTEM_LIB_DIRS:-/usr/lib /usr/lib64 /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu}"
BOOST_COMPONENTS="program_options log log_setup date_time thread system filesystem regex atomic chrono"
AKER_COMMANDS="${AKER_COMMANDS:-make git rsync}"
AKER_CXX="${AKER_CXX:-g++}"
AKER_GCC="${AKER_GCC:-gcc}"
QVCACHE_COMMANDS="${QVCACHE_COMMANDS:-make git}"
QVCACHE_PACKAGES="libaio-dev libgoogle-perftools-dev libunwind-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev libeigen3-dev libspdlog-dev libnuma-dev libtbb-dev"
DPKG_QUERY="${DPKG_QUERY:-dpkg-query}"

bt_first_version() { printf '%s\n' "$1" | grep -oE '[0-9]+(\.[0-9]+)+' | head -n 1 || true; }

bt_version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]; }

# find_cmake: the first CMake of at least MIN_CMAKE among the harness build venv, $Q1_TOOLS/bin and PATH.
find_cmake() {
    local c v
    for c in "$root/.venv-build/bin/cmake" "$TOOLS_DIR/bin/cmake" "$(command -v cmake 2> /dev/null || true)"; do
        if [ -n "$c" ] && [ -x "$c" ]; then
            v=$(bt_first_version "$("$c" --version 2> /dev/null | head -n 1)")
            if [ -n "$v" ] && bt_version_ge "$v" "$MIN_CMAKE"; then
                echo "$c"
                return 0
            fi
        fi
    done
    return 1
}

# cmake_bin: path of the CMake to use; a dry run prints the plain name when none is found.
cmake_bin() {
    find_cmake && return 0
    if [ "$DRY" -eq 1 ]; then echo cmake; return 0; fi
    return 1
}

# cmake_dir: directory to put first on PATH (empty when cmake is only a plain name).
cmake_dir() {
    case "$(cmake_bin)" in
        /*) dirname "$(cmake_bin)" ;;
        *) echo "" ;;
    esac
}

# tool_path: the PATH the builds run with.
tool_path() {
    local d
    d="$(cmake_dir)"
    if [ -n "$d" ]; then echo "$d:$PATH"; else echo "$PATH"; fi
}

# boost_prefix: the harness copy when it built one, else the system prefix.
boost_prefix() {
    local m
    for m in "$TOOLS_DIR"/boost/.installed-*; do
        if [ -e "$m" ]; then echo "$TOOLS_DIR/boost"; return 0; fi
    done
    echo "$SYSTEM_BOOST_PREFIX"
}

boost_version() {
    sed -n 's/.*BOOST_LIB_VERSION "\([0-9]*\)_\([0-9]*\).*/\1.\2/p' "$1/include/boost/version.hpp" 2> /dev/null | head -n 1
}

boost_lib_dirs() {
    local d
    for d in "$1/lib" "$1/lib64" "$1"/lib/*-linux-gnu; do
        if [ -d "$d" ]; then echo "$d"; fi
    done
}

# has_shared_lib NAME DIR...: lib<NAME>.so (the development link, which the linker needs) in one of the directories.
has_shared_lib() {
    local name="$1" d
    shift
    for d in "$@"; do
        if [ -e "$d/lib$name.so" ]; then return 0; fi
    done
    return 1
}

faiss_ready() {
    [ -f "$TP/faiss-install/.installed-$FAISS_COMMIT-shared" ] && [ -f "$TP/faiss-install/lib64/libfaiss.so" ] \
        && [ -f "$TP/faiss-install/include/faiss/Index.h" ]
}

compiler_missing() {
    local cxx="$1" v
    if ! command -v "$cxx" > /dev/null 2>&1; then
        echo "command $cxx (C++17 compiler, version >= $MIN_GCC)"
        return 0
    fi
    v=$(bt_first_version "$("$cxx" --version 2> /dev/null | head -n 1)")
    if [ -z "$v" ] || ! bt_version_ge "$v" "$MIN_GCC"; then
        echo "$cxx version >= $MIN_GCC (found ${v:-unknown})"
    fi
}

cmake_missing() {
    if ! find_cmake > /dev/null; then
        echo "cmake >= $MIN_CMAKE (looked in $root/.venv-build/bin, $TOOLS_DIR/bin and PATH)"
    fi
}

commands_missing() {
    local c
    for c in "$@"; do
        if ! command -v "$c" > /dev/null 2>&1; then echo "command $c"; fi
    done
}

# aker_missing: one line per tool or library the Aker build needs and does not find; nothing when all are there.
aker_missing() {
    local prefix ver comp
    cmake_missing
    compiler_missing "$AKER_CXX"
    # shellcheck disable=SC2086
    commands_missing $AKER_COMMANDS
    if [ ! -f "$root/third_party/aker/CMakeLists.txt" ]; then echo "vendored Aker source third_party/aker"; fi
    prefix="$(boost_prefix)"
    ver="$(boost_version "$prefix")"
    if [ -z "$ver" ]; then
        echo "Boost headers (boost/version.hpp under $prefix/include)"
    elif ! bt_version_ge "$ver" "$MIN_BOOST"; then
        echo "Boost >= $MIN_BOOST (found $ver under $prefix)"
    fi
    for comp in $BOOST_COMPONENTS; do
        # shellcheck disable=SC2046
        if ! has_shared_lib "boost_$comp" $(boost_lib_dirs "$prefix"); then
            echo "Boost shared library libboost_$comp.so under $prefix"
        fi
    done
    if ! faiss_ready; then
        # shellcheck disable=SC2086
        if ! has_shared_lib openblas $SYSTEM_LIB_DIRS; then echo "libopenblas.so (apt: libopenblas-dev), needed to build FAISS"; fi
        if [ "$("$AKER_GCC" -print-file-name=libgomp.so 2> /dev/null || echo libgomp.so)" = "libgomp.so" ]; then
            echo "libgomp.so (OpenMP of $AKER_GCC), needed to build FAISS"
        fi
    fi
}

# faiss_missing: after the shared FAISS is built.
faiss_missing() {
    if [ ! -f "$TP/faiss-install/include/faiss/Index.h" ]; then echo "FAISS headers $TP/faiss-install/include/faiss/Index.h"; fi
    if [ ! -f "$TP/faiss-install/lib64/libfaiss.so" ]; then echo "shared FAISS $TP/faiss-install/lib64/libfaiss.so"; fi
}

apt_missing() {
    local p
    for p in "$@"; do
        if ! "$DPKG_QUERY" -W -f='${Status}' "$p" 2> /dev/null | grep -q 'install ok installed'; then
            echo "apt package $p"
        fi
    done
}

# qvcache_missing: tools of the QVCache build (the apt libraries are checked after they are installed).
qvcache_missing() {
    cmake_missing
    compiler_missing "$AKER_CXX"
    # shellcheck disable=SC2086
    commands_missing $QVCACHE_COMMANDS
    if [ ! -x "$root/.venv-ann/bin/python" ]; then echo "$root/.venv-ann/bin/python (the venvs step)"; fi
}

# require_missing STEP: reads the missing lines on stdin; prints them and fails when there are any.
require_missing() {
    local lines
    lines="$(cat)"
    if [ -z "$lines" ]; then return 0; fi
    log "$1: missing:"
    printf '%s\n' "$lines" | sed 's/^/  - /'
    if [ -n "${FAIL_FILE:-}" ]; then printf '%s\n' "requirement check of step $1" > "$FAIL_FILE"; fi
    return 1
}

# ensure_aker_faiss: shared FAISS of the pinned commit in $TP/faiss-install (lib64, as the run scripts expect). The
# harness's static copy is not usable: Aker links libfaiss into a shared library and the static archive does not
# carry its BLAS and OpenMP dependencies. The harness's source tree is reused when it has the pinned commit.
ensure_aker_faiss() {
    local prefix="$TP/faiss-install" src="$TOOLS_DIR/src/faiss" build="$TP/src/faiss-shared-build" cmake_path nproc_n
    if faiss_ready; then
        log "shared FAISS already in $prefix"
        return 0
    fi
    cmake_path="$(cmake_bin)"
    nproc_n="$(nproc 2> /dev/null || echo 8)"
    if [ -L "$prefix" ] || [ -e "$prefix" ]; then as_user rm -rf "$prefix"; fi
    if [ "$(git -c safe.directory="$src" -C "$src" rev-parse HEAD 2> /dev/null || true)" != "$FAISS_COMMIT" ]; then
        src="$TP/src/faiss"
        as_user rm -rf "$src"
        as_user mkdir -p "$src"
        as_user git -C "$src" init -q
        as_user git -C "$src" fetch -q --depth 1 https://github.com/facebookresearch/faiss "$FAISS_COMMIT"
        as_user git -C "$src" checkout -q FETCH_HEAD
    fi
    as_user rm -rf "$build"
    as_user env "PATH=$(tool_path)" "$cmake_path" -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBLA_VENDOR="$BLAS_VENDOR" \
        -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib64
    as_user env "PATH=$(tool_path)" "$cmake_path" --build "$build" -j "$nproc_n"
    as_user env "PATH=$(tool_path)" "$cmake_path" --install "$build"
    as_user touch "$prefix/.installed-$FAISS_COMMIT-shared"
}

# --- QVCache module -----------------------------------------------------------------------------------------
# The artifact's build puts the Python module in <artifact>/python (qvcache.cpython-311-*.so); search the whole
# artifact directory, prefer that location, and never assume a path that was not found.

# qvcache_module: path of the built module; fails (and prints nothing) when there is none.
qvcache_module() {
    local src="$TP/qvcache" f
    for f in "$src"/python/qvcache*.so; do
        if [ -f "$f" ]; then echo "$f"; return 0; fi
    done
    f="$(find "$src" -name 'qvcache*.so' -type f 2> /dev/null | sort | head -n 1 || true)"
    if [ -n "$f" ]; then echo "$f"; return 0; fi
    return 1
}

# qvcache_imports: the module imports in .venv-ann (prints its file).
qvcache_imports() {
    as_user "$root/.venv-ann/bin/python" -c "import qvcache; print(qvcache.__file__)"
}

# register_qvcache: make the module importable from .venv-ann through a .pth file (written as the invoking user),
# then verify the import.
register_qvcache() {
    local module site
    module="$(qvcache_module)" || {
        log "qvcache: no qvcache*.so found anywhere under $TP/qvcache (the build did not produce the Python module)"
        return 1
    }
    site="$("$root/.venv-ann/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    if [ -z "$site" ] || [ ! -d "$site" ]; then
        log "qvcache: site-packages of $root/.venv-ann not found (got '$site')"
        return 1
    fi
    dirname "$module" | as_user tee "$site/qvcache.pth" > /dev/null
    log "qvcache: registered $(dirname "$module") in $site/qvcache.pth"
    qvcache_imports || {
        log "qvcache: import qvcache fails in $root/.venv-ann although $module exists"
        return 1
    }
}

# aker_ld_library_path: where the Aker shims find libaker, Boost and FAISS at run time.
aker_ld_library_path() {
    printf '%s' "$TP/aker_vendored/build/lib:$TP/aker_vendored_prox/build/lib:$TP/boost-install/lib:$TP/faiss-install/lib64"
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then printf ':%s' "$LD_LIBRARY_PATH"; fi
}
